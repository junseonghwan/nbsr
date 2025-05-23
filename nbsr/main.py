import os
import copy
import shutil
import time
from pathlib import Path

import click
import pandas as pd
import numpy as np
import scipy
import scipy.optimize as so
import scipy.stats as ss
import torch

from nbsr.nbsr_config import NBSRConfig
from nbsr.negbinomial_model import NegativeBinomialRegressionModel
from nbsr.nbsr_dispersion import NBSRTrended
from nbsr.dispersion import DispersionModel
from nbsr.utils import *

torch.set_default_dtype(torch.float64)
torch.set_printoptions(precision=9)

checkpoint_filename = "checkpoint.pth"
model_state_key = "model_state"
fisher_information_filename = "information.npy"

@click.group()
def cli():
	pass

def moving_average(arr, window):
	return np.convolve(arr, np.ones(window), 'valid') / window

def assess_convergence(loss_history, tol, lookback_iterations, window_size=100):
	if len(loss_history) < window_size:
		return False

	diffs = np.abs(np.diff(moving_average(loss_history, window_size)))
	if np.all(diffs[-lookback_iterations:] < tol):
		print(f"Convergence reached")
		return True
	else:
		print(f"Not converged")
		return False	

def compute_observed_information_torch(model, use_cuda_if_available=True):
	print("Computing Hessian using torch...")
	# Check if CUDA is available.
	if use_cuda_if_available:
		print(f"CUDA available? {torch.cuda.is_available()}")
		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		model.to_device(device)

	log_post_grad = model.log_posterior_gradient(model.beta)
	gradient_matrix = torch.zeros(log_post_grad.size(0), model.beta.size(0), device=model.beta.device)
	# Compute the gradient for each component of log_post_grad w.r.t. beta
	for k in range(log_post_grad.size(0)):
		# Zero previous gradient
		if model.beta.grad is not None:
			model.beta.grad.zero_()

		# Backward on the k-th component of y
		log_post_grad[k].backward(retain_graph=True)

		# Store the gradient
		gradient_matrix[k,:] = model.beta.grad

	return -gradient_matrix

# TODO: MARKED FOR REMOVAL.
# Numba related.
# def compute_observed_information_numba(model):
# 	print("Computing Hessian using Numba...")
# 	H = model.log_likelihood_hessian(model.beta)
# 	return -H

def compute_observed_information(model, use_cuda_if_available=True):
	start = time.perf_counter()
	I = compute_observed_information_torch(model, use_cuda_if_available)
	end = time.perf_counter()
	print("Hessian computation time = {}s".format((end - start)))
	return I

def inference_beta(model, var, w1, w0, x_map, I):
	"""
	Compute inference statistics for contrasting two levels for a given variable of interest.
	Note: w0 corresponds to the denominator in the log ratio while w1 corresponds to the numerator.

	Parameters
	----------
	model : NegativeBinomialRegressionModel object with the following attributes:
		- `X_df`: a pandas DataFrame with the design matrix.
		- `Y_df`: a pandas DataFrame with the response variable.
	var : str
		The name of the variable of interest.
	w1 : str
		The name of the numerator level for the fold change.
	w0 : str
		The name of the denominator level for the fold change.
	x_map: dict
		The dictionary containing map from var_w1/var_w0 to column index for the design matrix.

	Returns
	-------
	res : pandas DataFrame
		A DataFrame with the following columns:
		- `features`: the names of the features in the model.
		- `beta`: the natural logarithm fold change between the two levels of the variable.
		- `stdErr`: the standard error of the log2 fold change.
		- `z-score`: the z-score.
		- `pValue`: the p-value.
		- `adjPValue`: the Benjamini-Hochberg adjusted p-value.
	"""
	# Compute observed information matrix.
	# Compute (pseudo) inverse of observed Fisher information matrix to get covariance matrix.
	# Compute standard errors.
	assert I is not None

	S = torch.linalg.pinv(I)

	std_err_reshaped = reshape(model, torch.sqrt(torch.diag(S))).data.numpy()
	beta_reshaped = get_beta(model).data.numpy()

	var_level0 = "{varname}_{levelname}".format(varname=var, levelname=w0)
	var_level1 = "{varname}_{levelname}".format(varname=var, levelname=w1)
	col_idx0 = x_map[var_level0] if var_level0 in x_map else None
	col_idx1 = x_map[var_level1] if var_level1 in x_map else None
	found = False
	if col_idx0 is not None:
		# Offset by 1 because the first column is the intercept.
		beta0 = beta_reshaped[1+col_idx0,:]
		std_err0 = std_err_reshaped[1+col_idx0,:]
		found = True
	else:
		beta0 = 0
		std_err0 = 0
	if col_idx1 is not None:
		beta1 = beta_reshaped[1+col_idx1,:]
		std_err1 = std_err_reshaped[1+col_idx1,:]
		found = True
	else:
		beta1 = 0
		std_err1 = 0
	if not found:
		raise ValueError("Error: {level0}, {level1} not found in {varname}.".format(level0=w0, level1=w1, varname=var))

	diff = (beta1 - beta0)
	std_err = (std_err0**2) + (std_err1**2)

	# Create a data frame with the results.
	res = pd.DataFrame()
	# First column is the variable name.
	#res["features"] = model.Y_df.index.to_list()
	# Second column is the log2 fold change.
	res["diff"] = diff
	# Third column is the standard error.
	res["stdErr"] = std_err
	# Fourth column is the z-score.
	res["stat"] = diff / std_err
	# Fifth column is the p-value.
	res["pvalue"] = 2 * scipy.stats.norm.cdf(-np.abs(res["stat"]))
	# Sixth column is the Benjamini-Hochberg adjusted p-value.
	#res["adjPValue"] = false_discovery_control(res["pvalue"], method="bh")
	return res

def inference_logRR(model, var, w1, w0, x_map, I):
	
	assert I is not None

	# we will do the computation on CPU.
	model.to_device("cpu")

	S = torch.linalg.pinv(I)
	var_level0 = "{varname}_{levelname}".format(varname=var, levelname=w0)
	var_level1 = "{varname}_{levelname}".format(varname=var, levelname=w1)
	col_idx0 = x_map[var_level0] if var_level0 in x_map else None
	col_idx1 = x_map[var_level1] if var_level1 in x_map else None
	#print(col_idx0, col_idx1)

	Z0 = model.X.clone()
	Z1 = model.X.clone()

	# Zero out columns corresponding to var 
	for i,colname in enumerate(x_map):
		if var in colname: # Checks if var is a substring of colname.
			Z0[:,i+1] = 0 # +1 to account for the intercept.
			Z1[:,i+1] = 0

	found = False
	if col_idx0 is not None:
		Z0[:,col_idx0+1] = 1
		found = True
	if col_idx1 is not None:
		Z1[:,col_idx1+1] = 1
		found = True
	assert found == True
	print(f"Found covariate {var} in the model.")
	pi0, _ = model.predict(model.beta, Z0)
	pi1, _ = model.predict(model.beta, Z1)
	logRRi = torch.log(pi1) - torch.log(pi0)
	log2RRi = torch.log2(pi1) - torch.log2(pi0)

	# The gradient of g_j wrt (k,d) is expressed by 
	# z_{1,d} (1[j = k] - \pi_{k|w_1}) - z_{0,d} (1[j = k] - \pi_{k|w_0}).
	# We will construct two tensors ipi1 and ipi0 of size (N, J, J) where N is the sample count, K is the number of features.
	# ipi1[n,j,k] = (1[j = k] - \pi_{k|z_{1,n}}) and ipi0[n,j,k] = (1[j = k] - \pi_{k|z_{0,n}}).
	# one exception is that if pivot is used, the entry ipi1[n,J,k] = 0 - pi_{k|z_{1,n}}  (likewise for ipi0).
	identity = torch.eye(model.dim) # (J-1, J-1) or (J, J)
	if model.pivot:
		identity = torch.cat([identity, torch.zeros(1, model.dim)], dim=0) #(J, J-1), the last row of zeros.
	identity_mat = torch.tile(identity, (model.sample_count, 1, 1))
	
	# Use broadcasting to compute the difference between the identity matrix and pi1, pi0.
	# Since we have a tensor of size (N, J, J), subtracting pi1 of size (N, J), we need to unsqueeze pi1 along the last dimension to get (N, J, 1).
	# Then, broadcasting will essentially make a copy of pi1 along the last dimension to get (N, J, J) where the entries along the last dimension are all the same.
	# That is ipi1[n,j,k] = (1[j = k] - \pi_{k|z_{1,n}}).
	ipi0 = (identity_mat - pi0.unsqueeze(2)).transpose(1,2)
	ipi1 = (identity_mat - pi1.unsqueeze(2)).transpose(1,2)

	# We will take the product of ipi1 and ipi0 with Z1 and Z0, respectively.
	# The result will be a tensor of size (N, J, J*P).
	# ipi0 has dimension (N, J, J) while Z0 has dimension (N, P).
	# We want the results to be ret0[n,j,k,d] = ipi0[n,j,k] * Z0[n,d].
	# Again, we will use broadcating. 
	# First, expand Z0 to have dimension (N, 1, 1, P).
	# Expand ipi0 to have dimension (N, J, J, 1).
	ret0 = ipi0.unsqueeze(3) * Z0.unsqueeze(1).unsqueeze(2)
	ret1 = ipi1.unsqueeze(3) * Z1.unsqueeze(1).unsqueeze(2)
	ret = ret1 - ret0
	ret = ret.transpose(2, 3).reshape(model.sample_count, model.rna_count, model.dim * model.covariate_count)

	S_batch = S.unsqueeze(0).expand(model.sample_count, -1, -1)
	cov_mat = torch.bmm(torch.bmm(ret, S_batch), ret.transpose(1, 2))

	logFC = logRRi.data.numpy()
	log2FC = log2RRi.data.numpy()
	return (logFC, log2FC, cov_mat)

def fit_posterior(model, optimizer, iterations):
	# Fit the model.
	loss_history = []
	# We will store the best solution.
	best_model_state = None
	best_loss = torch.inf

	for i in range(iterations):

		loss = -model.log_posterior(model.beta)
		optimizer.zero_grad()
		loss.backward(retain_graph=False)
		optimizer.step()

		loss_history.append(loss.data.numpy())
		if loss.data < best_loss:
			best_model_state = copy.deepcopy(model.state_dict())
			best_loss = loss.data

		if i % 100 == 0:
			print("Iter:", i)
			print(loss.data)

	return (loss_history, best_model_state, best_loss)

def construct_model(config):
	#click.echo(config)
	counts_pd = pd.read_csv(config.counts_path, index_col=0)
	if os.path.exists(config.coldata_path):
		coldata_pd = pd.read_csv(config.coldata_path, na_filter=False, skipinitialspace=True)
	else:
		coldata_pd = None
	Y = torch.tensor(counts_pd.transpose().to_numpy(), dtype=torch.float64)
	X, x_map = construct_tensor_from_coldata(coldata_pd, config.column_names, counts_pd.shape[1])
	# We are not using z variables for now.
	#Z, z_map = construct_tensor_from_coldata(coldata_pd, config["z_columns"], counts_pd.shape[1], False)

	# Allow fixed dispersion values to be passed in.
	dispersion = read_file_if_exists(config.dispersion_path)
	dispersion_model_path = Path(config.output_path) / config.dispersion_model_file if config.dispersion_model_file is not None else None
	trended = config.trended_dispersion

	print("Y: ", Y.shape)
	print("X: ", X.shape)
	# if Z is not None:
	# 	print("Z: ", Z.shape)

	lam = config.lam
	shape = config.shape
	scale = config.scale
	pivot = config.pivot
	disp_model = None
	if dispersion is not None:
		print("Run NBSR with pre-specified dispersion values.")
		model = NegativeBinomialRegressionModel(X, Y, lam=lam, shape=shape, scale=scale, dispersion_prior=disp_model, dispersion=dispersion, pivot=pivot)
	else:
		if dispersion_model_path is not None:
			print(f"Dispersion prior model is specified. Loading from {dispersion_model_path}")
			disp_model = torch.load(dispersion_model_path, weights_only=False)
		if trended:
			if disp_model is None:
				print(f"Dispersion trend will be estimated.")
				disp_model = DispersionModel(Y, estimate_sd=config.estimate_dispersion_sd)
			model = NBSRTrended(X, Y, disp_model, lam=lam, shape=shape, scale=scale, pivot=pivot)
		else:
			print("Run NBSR with shared dispersion per feature.")
			model = NegativeBinomialRegressionModel(X, Y, lam=lam, shape=shape, scale=scale, dispersion_prior=disp_model, dispersion=None, pivot=pivot)

	param_list = []
	print("Parameters being optimized:")
	for name, param in model.named_parameters():
		print(name)
		if "disp_model" in name and dispersion_model_path is not None: # don't optimize disp_model parameters.
			continue
		param_list.append(param)

	#model.specify_beta_prior(config.lam, config.shape, config.scale)
	#print(torch.get_default_dtype())
	#print(model.X.dtype, model.Y.dtype)
	return model, param_list, x_map

def load_model_from_state_dict(config, state_dict):
    model, params, x_map = construct_model(config)
    checkpoint = None
    if model_state_key in state_dict:
        print("Loading previously saved model...")
        checkpoint = state_dict[model_state_key]
        model.load_state_dict(checkpoint['model_state_dict'])
    return model, params, x_map

def run(config):
	state_dict = {}
	output_path = Path(config.output_path)
	create_directory(output_path)

	lr = config.lr
	iterations = config.iterations

	# Note: load_model_from_state_dict will set x_map/z_map in state_dict.
	model, params, x_map = load_model_from_state_dict(config, state_dict)
	state_dict["x_map"]= x_map

	# Initialize optimizers.
	if model_state_key in state_dict:
		checkpoint = state_dict[model_state_key]
		optimizer = torch.optim.Adam(params, lr = lr)
		optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
		curr_loss_history = checkpoint['loss']
		curr_best_loss = checkpoint['best_loss']
		curr_best_model_state = checkpoint['best_model_state_dict']
	else:
		optimizer = torch.optim.Adam(params, lr = lr)
		curr_loss_history = []
		curr_best_loss = torch.inf
		curr_best_model_state = None

	loss_history, best_model_state, best_loss = fit_posterior(model, optimizer, iterations)
	print("Training iterations completed.")

	curr_loss_history.extend(loss_history)
	if best_loss < curr_best_loss:
		curr_best_loss = best_loss
		curr_best_model_state = best_model_state

	checkpoint = {
        	'model_state_dict': model.state_dict(), # Current state of the model.
	        'best_model_state_dict': curr_best_model_state, # State of the model with best loss.
	        'optimizer_state_dict': optimizer.state_dict(),
	        'loss': curr_loss_history,
	        'best_loss': curr_best_loss
	}
	state_dict[model_state_key] = checkpoint

	# Generate output from the best state.
	model.load_state_dict(curr_best_model_state)
	pi, _ = model.predict(model.beta, model.X)
	if isinstance(model, NBSRTrended):
		phi = torch.exp(model.disp_model(pi))
		np.savetxt(output_path / "nbsr_dispersion_b0.csv", model.disp_model.b0.data.numpy().transpose(), delimiter=',')
		np.savetxt(output_path / "nbsr_dispersion_b1.csv", model.disp_model.b1.data.numpy().transpose(), delimiter=',')
		np.savetxt(output_path / "nbsr_dispersion_b2.csv", model.disp_model.b2.data.numpy().transpose(), delimiter=',')
		if config.estimate_dispersion_sd:
			np.savetxt(output_path / "nbsr_dispersion_sd.csv", model.disp_model.get_sd().data.numpy().transpose(), delimiter=',')
	else:
		phi = model.softplus(model.phi)

	np.savetxt(output_path / "nbsr_beta.csv", model.beta.data.numpy().transpose(), delimiter=',')
	np.savetxt(output_path / "nbsr_beta_sd.csv", model.softplus(model.psi.data).numpy().transpose(), delimiter=',')
	np.savetxt(output_path / "nbsr_pi.csv", pi.data.numpy().transpose(), delimiter=',')
	np.savetxt(output_path / "nbsr_dispersion.csv", phi.data.numpy().transpose(), delimiter=',')

	print("Compute observed Information matrix.")
	I = compute_observed_information(model, config.use_cuda_if_available)
	np.save(output_path / fisher_information_filename, I.detach().cpu().numpy())

	torch.save(state_dict, output_path / checkpoint_filename)
	config.dump_json(output_path / "config.json")

	return(curr_loss_history, model)

def generate_results(results_path, var, w1, w0, absolute_fc=True, recompute_hessian=False):
	results_path = Path(results_path)
	config = NBSRConfig.load_json(results_path / "config.json")
	state_dict = torch.load(results_path / checkpoint_filename, weights_only=False)
	x_map = state_dict["x_map"]

	model, _, x_maps2 = load_model_from_state_dict(config, state_dict)

	# Perform sanity check, the column name mapping should match.
	for k in x_map.keys():
		assert k in x_maps2
		assert x_map[k] == x_maps2[k]

	# Check if hessian matrix exists, compute it otherwise.
	I = None
	if not recompute_hessian and os.path.exists(results_path / fisher_information_filename):
		I = torch.from_numpy(np.load(results_path / fisher_information_filename)).double()
	else: # compute Hessian
		print("Compute observed Information matrix.")
		I = compute_observed_information(model, config.use_cuda_if_available).detach().cpu()
		np.save(results_path / fisher_information_filename, I)

	logRR, log2RR, cov_mat  = inference_logRR(model, var, w1, w0, x_map, I)
	logRR_std = torch.sqrt(torch.diagonal(cov_mat, dim1 = 1, dim2 = 2)).data.numpy()

	# Compute the test statistic and the p-values.
	log_bias = 0
	if absolute_fc:
		# Find the mode of the log2RR.
		# log2RR is N x P (N: number of samples, P: number of features).
		def kde_mode(x):
			kde = ss.gaussian_kde(x)
			neg_kde = lambda x: -kde(x)
			result = so.minimize_scalar(neg_kde, bounds=(x.min(), x.max()), method='bounded')
			return result.x
		log_bias = np.array(list(map(kde_mode, logRR)))

	counts_pd = pd.read_csv(config.counts_path, index_col=0)
	samples = counts_pd.columns
	features = counts_pd.index

	logRR = (logRR - log_bias)
	stat = logRR / logRR_std
	pvalue = 2 * ss.norm.cdf(-np.abs(stat))
	# Fifth column is the adjusted p-value.
	padj = np.array(list(map(lambda x: ss.false_discovery_control(x, method="bh"), pvalue)))

	# Output logRR, se, p-value, adjusted p-value.
	# Output using h5 file format.
	with pd.HDFStore(results_path / "nbsr_results.h5", mode='w') as store:
		store['logRR'] = pd.DataFrame(logRR.T, index=features, columns=samples)
		store['se'] = pd.DataFrame(logRR_std.T, index=features, columns=samples)
		store['stat'] = pd.DataFrame(stat.T, index=features, columns=samples)
		store['pvalue'] = pd.DataFrame(pvalue.T, index=features, columns=samples)
		store['padj'] = pd.DataFrame(padj.T, index=features, columns=samples)
		if absolute_fc:
			store['log_bias'] = pd.DataFrame(log_bias, index=samples)

	# If there is only one covariate (experimental factor) output it as csv results file.
	if np.allclose(log2RR, log2RR[0, :], atol=1e-8):
		#Output results table.
		res = pd.DataFrame({
			"feature": features,
			"log2FC": log2RR[0,:],
			"pvalue": pvalue[0,:],
			"padj": padj[0,:]})
		res.to_csv(results_path / "nbsr_results.csv", index=False)
		return res

	return None

@click.command()
@click.argument('data_path', type=click.Path(exists=True))
@click.argument('vars', nargs=-1)
@click.option('-f', '--mu_file', default="deseq2_mu.csv", type=str, help="File name containing the initial fit for mu.")
@click.option('-i', '--iterations', default=10000, type=int)
@click.option('-l', '--lr', default=0.05, type=float, help="NBSR model parameters learning rate.")
@click.option('--eb_iter', default=3000, type=int, help="NBSR dispersion model training iterations.")
@click.option('--eb_lr', default=0.05, type=float, help="NBSR dispersion model parameters learning rate.")
@click.option('--lam', default=1., type=float)
@click.option('--shape', default=3, type=float)
@click.option('--scale', default=2, type=float)
@click.option('--estimate_dispersion_sd', is_flag=True, show_default=False, default=False, type=bool)
@click.option('--update_dispersion', is_flag=True, show_default=False, default=False, type=bool)
@click.option('--pivot', is_flag=True, show_default=True, default=False, type=bool)
def eb(data_path, vars, mu_file, iterations, lr, eb_iter, eb_lr, lam, shape, scale, estimate_dispersion_sd, update_dispersion, pivot):

	data_path = Path(data_path)
	# Read in the mean expression.
	# Optimize NBSREmpiricalBayes to get MLE dispersions.
	# Fit GRBF with phi_mle ~ f(mu_bar).
	# Obtain the mean dispersion and use it for fitting NBSR and output it to file.
	print("Performing Empirical Bayes estimation of dispersion.")
	column_names = list(vars)
	mu_hat = pd.read_csv(data_path / mu_file)
	mu_hat = torch.tensor(mu_hat.transpose().to_numpy())
	counts_pd = pd.read_csv(data_path / "Y.csv", index_col=0) # first column is just the name of the miRNAs.
	if os.path.exists(data_path / "X.csv"):
		coldata_pd = pd.read_csv(data_path / "X.csv", na_filter=False, skipinitialspace=True)
	else:
		coldata_pd = None
	Y = torch.tensor(counts_pd.transpose().to_numpy(), dtype=torch.float64)
	X, x_map = construct_tensor_from_coldata(coldata_pd, column_names, counts_pd.shape[1])
	disp_model_path = "disp_model.pth"
	config = NBSRConfig(counts_path=data_path / "Y.csv",
						coldata_path=data_path / "X.csv",
						output_path=data_path, # output to where the data is.
						column_names=column_names,
						z_columns=None,
						lr=lr,
						lam=lam,
						shape=shape,
						scale=scale,
						estimate_dispersion_sd=estimate_dispersion_sd,
						trended_dispersion=True,
						dispersion_model_file=disp_model_path,
						pivot=pivot
						)
	
	pi_hat = mu_hat / mu_hat.sum(dim=1, keepdim=True)

	disp_model = DispersionModel(Y, estimate_sd=estimate_dispersion_sd)
	nbsr_model = NBSRTrended(X, Y, disp_model=disp_model, lam=config.lam, shape=config.shape, scale=config.scale, pivot=pivot)
	optimizer = torch.optim.Adam(nbsr_model.disp_model.parameters(),lr=eb_lr)
	print("Optimizing NBSR dispersion parameters given DESeq2 mean expression levels.")
	for i in range(eb_iter):
		phi = torch.exp(nbsr_model.disp_model.forward(pi_hat))
		log_prior = nbsr_model.disp_model.log_prior()
		loss = -(nbsr_model.log_likelihood(pi_hat, phi) + log_prior)
		if loss.isnan():
			print("nan")
			break
		optimizer.zero_grad()
		loss.backward(retain_graph=True)
		optimizer.step()
		if i % 100 == 0:
			print("Iter:", i)
			print(loss.data)

	phi = torch.exp(nbsr_model.disp_model.forward(pi_hat))
	np.savetxt(data_path / "eb_dispersion.csv", phi.data.numpy().transpose(), delimiter=',')
	np.savetxt(data_path / "nbsr_dispersion_b0.csv", nbsr_model.disp_model.b0.data.numpy().transpose(), delimiter=',')
	np.savetxt(data_path / "nbsr_dispersion_b1.csv", nbsr_model.disp_model.b1.data.numpy().transpose(), delimiter=',')
	np.savetxt(data_path / "nbsr_dispersion_b2.csv", nbsr_model.disp_model.b2.data.numpy().transpose(), delimiter=',')
	torch.save(disp_model, config.output_path / disp_model_path)

	# Fit NBSR parameters but do not update the dispersion parameters.
	if update_dispersion:
		param_list = nbsr_model.named_parameters()
	else:
		param_list = []
		for name, param in nbsr_model.named_parameters():
			if "disp_model" in name:
				continue
			param_list.append(param)

	#nbsr_model.specify_beta_prior(lam, shape, scale)
	optimizer = torch.optim.Adam(param_list,lr=lr)
	print("Optimizing NBSR parameters given DESeq2 mean expression levels.")
	for i in range(iterations):
		loss = -nbsr_model.log_posterior(nbsr_model.beta)
		if loss.isnan():
			print("nan")
			break
		optimizer.zero_grad()
		loss.backward(retain_graph=True)
		optimizer.step()
		if i % 100 == 0:
			print("Iter:", i)
			print(loss.data)

	pi, _ = nbsr_model.predict(nbsr_model.beta, nbsr_model.X)
	phi = torch.exp(nbsr_model.disp_model.forward(pi))
	np.savetxt(data_path / "nbsr_beta.csv", nbsr_model.beta.data.numpy().transpose(), delimiter=',')
	np.savetxt(data_path / "nbsr_beta_sd.csv", nbsr_model.softplus(nbsr_model.psi.data).numpy().transpose(), delimiter=',')
	np.savetxt(data_path / "nbsr_pi.csv", pi.data.numpy().transpose(), delimiter=',')
	np.savetxt(data_path / "nbsr_dispersion.csv", phi.data.numpy().transpose(), delimiter=',')

	I = compute_observed_information(nbsr_model)
	np.save(data_path / fisher_information_filename, I)

	model_state = {
		'model_state_dict': nbsr_model.state_dict(),
		'best_model_state_dict': nbsr_model.state_dict(),
		'optimizer_state_dict': optimizer.state_dict(),
		'loss': None,
		'best_loss': None,
		'converged': None
	}
	state_dict = {}
	state_dict[model_state_key] = model_state
	state_dict["x_map"] = x_map
	torch.save(state_dict, config.output_path / checkpoint_filename)
	config.dump_json(config.output_path / "config.json")

@click.command()
@click.argument('data_path', type=click.Path(exists=True))
@click.argument('vars', nargs=-1)
@click.option('-i', '--iterations', default=10000, type=int)
@click.option('-l', '--lr', default=0.05, type=float, help="NBSR model parameters learning rate.")
@click.option('-r', '--runs', default=1, type= int, help="Number of optimization runs (initialization).")
@click.option('--z_columns', multiple=True, help="Enter list of strings specifying the covariate names to use for the dispersion model.")
@click.option('--lam', default=1., type=float)
@click.option('--shape', default=3, type=float)
@click.option('--scale', default=2, type=float)
@click.option('--dispersion_model_file', default=None, type=str)
@click.option('--trended_dispersion', is_flag=True, show_default=True, default=False, type=bool)
@click.option('--estimate_dispersion_sd', is_flag=True, show_default=False, default=False, type=bool)
@click.option('--pivot', is_flag=True, show_default=True, default=False, type=bool)
def train(data_path, vars, iterations, lr, runs, z_columns, lam, shape, scale, dispersion_model_file, trended_dispersion, estimate_dispersion_sd, pivot):

	data_path = Path(data_path)
	losses = []
	for run_no in range(runs):
		outpath = data_path / ("run" + str(run_no))
		config = NBSRConfig(counts_path=data_path / "Y.csv",
					  		coldata_path=data_path / "X.csv",
							output_path=outpath,
					  		column_names=list(vars),
							z_columns=list(z_columns),
							lr=lr,
							iterations=iterations,
							lam=lam,
							shape=shape,
							scale=scale,
							estimate_dispersion_sd=estimate_dispersion_sd,
							trended_dispersion=trended_dispersion,
							dispersion_model_file=dispersion_model_file,
							pivot=pivot)
		loss_history, _ = run(config)
		losses.append(np.min(loss_history)) # store the best (minimal) loss.

	# Find the best run and copy all the output files to data_path.
	best_run = np.argmin(np.array(losses))
	best_run_path = data_path / ("run" + str(best_run))
	for filename in os.listdir(best_run_path):
		file_path = best_run_path / filename
		if os.path.isfile(file_path):
			# Copy each file to data_path
			shutil.copy2(file_path, data_path / filename)

@click.command()
@click.argument('checkpoint_path', type=click.Path(exists=True))
@click.argument('var', type=str)
@click.argument('w1', type=str) # "level to be used on the numerator"
@click.argument('w0', type=str) # "level to be used on the denominator"
@click.option('--absolute_fc', default=False, is_flag=True, type=bool)
@click.option('--recompute_hessian', is_flag=True, show_default=True, default=False, type=bool)
def results(checkpoint_path, var, w1, w0, absolute_fc, recompute_hessian):
	generate_results(checkpoint_path, var, w1, w0, absolute_fc, recompute_hessian)

	# res_beta = inference_beta(model, var, w1, w0, config["x_map"], I)
	# res_beta.to_csv(os.path.join(checkpoint_path, "coefficients.csv"), index=False)

	# Save Hessian for future use.
	# if save_hessian:
	# 	np.savetxt(os.path.join(checkpoint_path, "hessian.csv"), I, delimiter=',')
	# np.savetxt(os.path.join(output_path, "nbsr_logRR.csv"), logRR, delimiter=',')
	# np.savetxt(os.path.join(output_path, "nbsr_logRR_sd.csv"), logRR_std, delimiter=',')
	# cov_mat is NxKxK tensor. 
	# np.save(os.path.join(output_path, "nbsr_logRR_cov.npy"), cov_mat.data.numpy())


cli.add_command(eb)
cli.add_command(train)
cli.add_command(results)

if __name__ == '__main__':
    cli()



# TODO: MARKED FOR REMOVAL
# def get_config(data_path, output_path, cols, z_cols, lr, lam, shape, scale, estimate_dispersion_sd, dispersion_model_path, trended_dispersion, pivot):
# 	config = {
# 		"data_path": data_path,
# 		"output_path": output_path,
# 		"counts_path": os.path.join(data_path, "Y.csv"),
# 		"coldata_path": os.path.join(data_path, "X.csv"),
# 		"dispersion_path": os.path.join(data_path, "dispersion.csv"),
# 		"prior_sd_path": os.path.join(data_path, "prior_sd.csv"),
# 		"column_names": cols,
# 		"z_columns": z_cols,
# 		"lr": lr,
# 		"lam": lam,
# 		"shape": shape,
# 		"scale": scale,
# 		"estimate_dispersion_sd": estimate_dispersion_sd,
# 		"trended_dispersion": trended_dispersion,
# 		"dispersion_model_path": dispersion_model_path,
# 		"pivot": pivot
# 	}
# 	return config

# TODO: MARK FOR REMOVAL.
# @click.command()
# @click.argument('checkpoint_path', type=click.Path(exists=True))
# @click.option('-i', '--iterations', default=1000, type=int)
# @click.option('--tol', default=0.01, type=float)
# @click.option('--lookback_iterations', default=50, type=int)
# def resume(checkpoint_path, iterations, tol, lookback_iterations):
# 	state_dict = torch.load(os.path.join(checkpoint_path, checkpoint_filename), weights_only=False)
# 	run(state_dict, iterations, tol, lookback_iterations)
