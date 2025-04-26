import os
import copy
import shutil
import json

import click
import pandas as pd
import numpy as np
import scipy
import scipy.optimize as so
import scipy.stats as ss
import torch

from nbsr.negbinomial_model import NegativeBinomialRegressionModel
from nbsr.nbsr_dispersion import NBSRTrended
from nbsr.dispersion import DispersionModel
from nbsr.utils import *

torch.set_default_dtype(torch.float64)
torch.set_printoptions(precision=9)

checkpoint_filename = "checkpoint.pth"
model_state_key = "model_state"

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

def compute_observed_information_torch(model):
	print("Computing Hessian...")
	log_post_grad = model.log_posterior_gradient(model.beta)
	gradient_matrix = torch.zeros(log_post_grad.size(0), model.beta.size(0))
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

def compute_observed_information(model):
	print("Computing Hessian using Numba...")
	X = model.X.data.numpy()
	Y = model.Y.data.numpy()
	pi = model.predict(model.beta, model.X)[0].data.numpy()
	phi = model.softplus(model.phi).data.numpy()
	s = np.sum(Y, axis=1)
	mu = s[:,None] * pi
	_, H = log_lik_gradients(X, Y, pi, mu, phi, model.pivot)
	return -H

def inference_beta(model, var, w1, w0, x_map, I=None):
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
	if I is None:
		I = compute_observed_information(model)
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
	return (res, I)

def inference_logRR(model, var, w1, w0, x_map, I = None):
	if I is None:
		I = compute_observed_information(model)
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
	print(f"Found covariate {var} in the model.")
	pi0, _ = model.predict(model.beta, Z0)
	pi1, _ = model.predict(model.beta, Z1)
	logRRi = torch.log(pi1) - torch.log(pi0)
	log2RRi = torch.log2(pi1) - torch.log2(pi0)

	# The gradient of g_j wrt (k,d) is expressed by 
	# z_{1,d} (1[j = k] - \pi_{k|w_1}) - z_{0,d} (1[j = k] - \pi_{k|w_0}).
	# We will construct two tensors ipi1 and ipi0 of size (N, J, J) where N is the sample count, K is the number of features.
	# ipi1[n,j,k] = (1[j = k] - \pi_{k|z_{1,n}}) and ipi0[n,j,k] = (1[j = k] - \pi_{k|z_{0,n}}).
	identity_mat = torch.tile(torch.eye(model.dim), (model.sample_count, 1, 1))
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
	ret = ret.transpose(2, 3).reshape(model.sample_count, model.dim, model.dim * model.covariate_count)

	S_batch = S.unsqueeze(0).expand(model.sample_count, -1, -1)
	cov_mat = torch.bmm(torch.bmm(ret, S_batch), ret.transpose(1, 2))

	logFC = logRRi.data.numpy()
	log2FC = log2RRi.data.numpy()
	return (logFC, log2FC, cov_mat, I)

def fit_posterior(model, optimizer, iterations, tol, lookback_iterations):
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

	converged = assess_convergence(loss_history, tol, lookback_iterations)
	return (loss_history, best_model_state, best_loss, converged)

def construct_model(config):
	click.echo(config)
	counts_pd = pd.read_csv(config["counts_path"])
	if os.path.exists(config["coldata_path"]):
		coldata_pd = pd.read_csv(config["coldata_path"], na_filter=False, skipinitialspace=True)
	else:
		coldata_pd = None
	Y = torch.tensor(counts_pd.transpose().to_numpy(), dtype=torch.float64)
	X, x_map = construct_tensor_from_coldata(coldata_pd, config["column_names"], counts_pd.shape[1])
	#Z, z_map = construct_tensor_from_coldata(coldata_pd, config["z_columns"], counts_pd.shape[1], False)
	# Update the x_map and save it.
	config["x_map"] = x_map
	#config["z_map"] = z_map

	data_path = config["data_path"]
	dispersion = read_file_if_exists(config["dispersion_path"])
	prior_sd = read_file_if_exists(config["prior_sd_path"])
	dispersion_model_path = config["dispersion_model_path"]
	trended = config["trended_dispersion"]

	print("Y: ", Y.shape)
	print("X: ", X.shape)
	# if Z is not None:
	# 	print("Z: ", Z.shape)
	if prior_sd is not None:
		print("prior_sd: ", prior_sd.shape)
	else:
		print("prior_sd unspecified and will be estimated.")

	disp_model = None
	if dispersion is not None:
		print("Run NBSR with pre-specified dispersion values.")
		model = NegativeBinomialRegressionModel(X, Y, dispersion_prior=disp_model, dispersion=dispersion, prior_sd=prior_sd, pivot=config["pivot"])
	else:
		if dispersion_model_path is not None:
			print(f"Dispersion prior model is pre-specified. Loading from {dispersion_model_path}")
			disp_model = torch.load(os.path.join(data_path, dispersion_model_path), weights_only=False)
		if trended:
			if disp_model is None:
				print(f"Dispersion trend will be estimated.")
				disp_model = DispersionModel(Y, estimate_sd=config["estimate_dispersion_sd"])
			model = NBSRTrended(X, Y, disp_model, prior_sd=prior_sd, pivot=config["pivot"])
		else:
			model = NegativeBinomialRegressionModel(X, Y, dispersion_prior=disp_model, prior_sd=prior_sd, pivot=config["pivot"])

	param_list = []
	for name, param in model.named_parameters():
		print(name)
		if "disp_model" in name and dispersion_model_path is not None: # don't optimize disp_model parameters.
			continue
		param_list.append(param)

	model.specify_beta_prior(config["lam"], config["shape"], config["scale"])
	print(torch.get_default_dtype())
	print(model.X.dtype, model.Y.dtype)
	return model, param_list

def load_model_from_state_dict(state_dict, config):
    model, params = construct_model(config)
    checkpoint = None
    if model_state_key in state_dict:
        print("Loading previously saved model...")
        checkpoint = state_dict[model_state_key]
        model.load_state_dict(checkpoint['model_state_dict'])
    return model, params, checkpoint

def run(state_dict, iterations, tol, lookback_iterations):
	config = state_dict["config"]
	output_path = config["output_path"]
	create_directory(output_path)

	lr = config["lr"]

	model, params, checkpoint = load_model_from_state_dict(state_dict, config)

	# Initialize optimizers.
	if checkpoint:
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

	loss_history, best_model_state, best_loss, converged = fit_posterior(model, optimizer, iterations, tol, lookback_iterations)
	curr_loss_history.extend(loss_history)
	if best_loss < curr_best_loss:
		curr_best_loss = best_loss
		curr_best_model_state = best_model_state

	model_state = {
        	'model_state_dict': model.state_dict(),
	        'best_model_state_dict': curr_best_model_state,
	        'optimizer_state_dict': optimizer.state_dict(),
	        'loss': curr_loss_history,
	        'best_loss': curr_best_loss,
	        'converged': converged
	}
	torch.save({
		'model_state': model_state
        }, os.path.join(output_path, checkpoint_filename))

	model.load_state_dict(curr_best_model_state)
	pi, _ = model.predict(model.beta, model.X)
	if isinstance(model, NBSRTrended):
		phi = torch.exp(model.disp_model(pi))
		np.savetxt(os.path.join(output_path, "nbsr_dispersion_b0.csv"), model.disp_model.b0.data.numpy().transpose(), delimiter=',')
		np.savetxt(os.path.join(output_path, "nbsr_dispersion_b1.csv"), model.disp_model.b1.data.numpy().transpose(), delimiter=',')
		np.savetxt(os.path.join(output_path, "nbsr_dispersion_b2.csv"), model.disp_model.b2.data.numpy().transpose(), delimiter=',')
		if config["estimate_dispersion_sd"]:
			np.savetxt(os.path.join(output_path, "nbsr_dispersion_sd.csv"), model.disp_model.get_sd().data.numpy().transpose(), delimiter=',')
	else:
		phi = model.softplus(model.phi)

	np.savetxt(os.path.join(output_path, "nbsr_beta.csv"), model.beta.data.numpy().transpose(), delimiter=',')
	np.savetxt(os.path.join(output_path, "nbsr_beta_sd.csv"), model.softplus(model.psi.data).numpy().transpose(), delimiter=',')
	np.savetxt(os.path.join(output_path, "nbsr_pi.csv"), pi.data.numpy().transpose(), delimiter=',')
	np.savetxt(os.path.join(output_path, "nbsr_dispersion.csv"), phi.data.numpy().transpose(), delimiter=',')
	
	print("Training iterations completed.")
	print("Converged? " + str(converged))
	return(curr_best_loss)

def get_config(data_path, output_path, cols, z_cols, lr, lam, shape, scale, estimate_dispersion_sd, dispersion_model_path, trended_dispersion, pivot):
	config = {
		"data_path": data_path,
		"output_path": output_path,
		"counts_path": os.path.join(data_path, "Y.csv"),
		"coldata_path": os.path.join(data_path, "X.csv"),
		"dispersion_path": os.path.join(data_path, "dispersion.csv"),
		"prior_sd_path": os.path.join(data_path, "prior_sd.csv"),
		"column_names": cols,
		"z_columns": z_cols,
		"lr": lr,
		"lam": lam,
		"shape": shape,
		"scale": scale,
		"estimate_dispersion_sd": estimate_dispersion_sd,
		"trended_dispersion": trended_dispersion,
		"dispersion_model_path": dispersion_model_path,
		"pivot": pivot
	}
	return config

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

	# Read in the mean expression.
	# Optimize NBSREmpiricalBayes to get MLE dispersions.
	# Fit GRBF with phi_mle ~ f(mu_bar).
	# Obtain the mean dispersion and use it for fitting NBSR and output it to file.
	print("Performing Empirical Bayes estimation of dispersion.")
	column_names = list(vars)
	mu_hat = pd.read_csv(os.path.join(data_path, mu_file))
	mu_hat = torch.tensor(mu_hat.transpose().to_numpy())
	counts_pd = pd.read_csv(os.path.join(data_path, "Y.csv"))
	if os.path.exists(os.path.join(data_path, "X.csv")):
		coldata_pd = pd.read_csv(os.path.join(data_path, "X.csv"), na_filter=False, skipinitialspace=True)
	else:
		coldata_pd = None
	Y = torch.tensor(counts_pd.transpose().to_numpy(), dtype=torch.float64)
	X, x_map = construct_tensor_from_coldata(coldata_pd, column_names, counts_pd.shape[1])
	disp_model_path = "disp_model.pth"
	config = get_config(data_path, data_path, column_names, None, lr, lam, shape, scale, estimate_dispersion_sd, disp_model_path, True, pivot)
	config["x_map"] = x_map

	pi_hat = mu_hat / mu_hat.sum(dim=1, keepdim=True)

	disp_model = DispersionModel(Y, estimate_sd=estimate_dispersion_sd)
	nbsr_model = NBSRTrended(X, Y, disp_model=disp_model, pivot=pivot)
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
	sd = nbsr_model.disp_model.get_sd()
	np.savetxt(os.path.join(data_path, "eb_dispersion.csv"), phi.data.numpy().transpose(), delimiter=',')
	np.savetxt(os.path.join(data_path, "nbsr_dispersion_b0.csv"), nbsr_model.disp_model.b0.data.numpy().transpose(), delimiter=',')
	np.savetxt(os.path.join(data_path, "nbsr_dispersion_b1.csv"), nbsr_model.disp_model.b1.data.numpy().transpose(), delimiter=',')
	np.savetxt(os.path.join(data_path, "nbsr_dispersion_b2.csv"), nbsr_model.disp_model.b2.data.numpy().transpose(), delimiter=',')
	np.savetxt(os.path.join(data_path, "nbsr_dispersion_sd.csv"), np.array([sd.data.numpy()]), delimiter=',')
	torch.save(disp_model, os.path.join(data_path, disp_model_path))

	# Fit NBSR parameters but do not update the dispersion parameters.
	if update_dispersion:
		param_list = nbsr_model.named_parameters()
	else:
		param_list = []
		for name, param in nbsr_model.named_parameters():
			if "disp_model" in name:
				continue
			param_list.append(param)

	nbsr_model.specify_beta_prior(lam, shape, scale)
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
	np.savetxt(os.path.join(data_path, "nbsr_beta.csv"), nbsr_model.beta.data.numpy().transpose(), delimiter=',')
	np.savetxt(os.path.join(data_path, "nbsr_beta_sd.csv"), nbsr_model.softplus(nbsr_model.psi.data).numpy().transpose(), delimiter=',')
	np.savetxt(os.path.join(data_path, "nbsr_pi.csv"), pi.data.numpy().transpose(), delimiter=',')
	np.savetxt(os.path.join(data_path, "nbsr_dispersion.csv"), phi.data.numpy().transpose(), delimiter=',')

	I = compute_observed_information(nbsr_model)
	np.savetxt(os.path.join(data_path, "hessian.csv"), I, delimiter=',')

	model_state = {
		'model_state_dict': nbsr_model.state_dict(),
		'best_model_state_dict': nbsr_model.state_dict(),
		'optimizer_state_dict': optimizer.state_dict(),
		'loss': None,
		'best_loss': None,
		'converged': None
	}
	torch.save({
		'model_state': model_state,
        'config': config
        }, os.path.join(data_path, 'checkpoint.pth'))


@click.command()
@click.argument('data_path', type=click.Path(exists=True))
@click.argument('vars', nargs=-1)
@click.option('-d', '--dispersion_model_file', default=None, type=str)
@click.option('-i', '--iterations', default=10000, type=int)
@click.option('-l', '--lr', default=0.05, type=float, help="NBSR model parameters learning rate.")
@click.option('-r', '--runs', default=1, type= int, help="Number of optimization runs (initialization).")
@click.option('--z_columns', multiple=True, help="Enter list of strings specifying the covariate names to use for the dispersion model.")
@click.option('--lam', default=1., type=float)
@click.option('--shape', default=3, type=float)
@click.option('--scale', default=2, type=float)
@click.option('--trended_dispersion', is_flag=True, show_default=True, default=False, type=bool)
@click.option('--estimate_dispersion_sd', is_flag=True, show_default=False, default=False, type=bool)
@click.option('--pivot', is_flag=True, show_default=True, default=False, type=bool)
@click.option('--tol', default=0.01, type=float)
@click.option('--lookback_iterations', default=50, type=int)
def train(data_path, vars, iterations, lr, runs, z_columns, lam, shape, scale, dispersion_model_file, trended_dispersion, estimate_dispersion_sd, pivot, tol, lookback_iterations):

	losses = []
	for run_no in range(runs):
		outpath = os.path.join(data_path, "run" + str(run_no))
		config = get_config(data_path=data_path, output_path=outpath, cols=list(vars), z_cols=list(z_columns), 
					  lr=lr, lam=lam, shape=shape, scale=scale, 
					  estimate_dispersion_sd=estimate_dispersion_sd,
					  dispersion_model_path=dispersion_model_file, 
					  trended_dispersion=trended_dispersion, 
					  pivot=pivot)
		state = {"config": config}
		loss = run(state, iterations, tol, lookback_iterations)
		losses.append(loss)

		# Save config file.
		json_text = json.dumps(config, indent=4)  # `indent` makes the JSON pretty-printed
		with open(os.path.join(outpath, "config.json"), "w") as file:
			file.write(json_text)

	# Find the best run and copy the results and checkpoint file up to the data_path.
	best_run = np.argmin(np.array(losses))
	best_run_path = os.path.join(data_path, "run" + str(best_run))
	for filename in os.listdir(best_run_path):
		file_path = os.path.join(best_run_path, filename)
		if os.path.isfile(file_path):
			# Copy each file to data_path
			shutil.copy2(file_path, os.path.join(data_path, filename))

	# Obtain Hessian matrix.
	state_dict = torch.load(os.path.join(data_path, checkpoint_filename), weights_only=False)
	model, _,  _ = load_model_from_state_dict(state_dict, config)
	I = compute_observed_information(model)
	np.savetxt(os.path.join(data_path, "hessian.csv"), I, delimiter=',')

@click.command()
@click.argument('checkpoint_path', type=click.Path(exists=True))
@click.option('-i', '--iterations', default=1000, type=int)
@click.option('--tol', default=0.01, type=float)
@click.option('--lookback_iterations', default=50, type=int)
def resume(checkpoint_path, iterations, tol, lookback_iterations):
	state_dict = torch.load(os.path.join(checkpoint_path, checkpoint_filename), weights_only=False)
	run(state_dict, iterations, tol, lookback_iterations)

@click.command()
@click.argument('checkpoint_path', type=click.Path(exists=True))
@click.argument('var', type=str)
@click.argument('w1', type=str) # "level to be used on the numerator"
@click.argument('w0', type=str) # "level to be used on the denominator"
@click.option('--absolute_fc', default=False, is_flag=True, type=bool)
@click.option('--output_path', default=None, type=str)
@click.option('--recompute_hessian', is_flag=True, show_default=True, default=False, type=bool)
@click.option('--save_hessian', is_flag=True, show_default=True, default=False, type=bool)
def results(checkpoint_path, var, w1, w0, absolute_fc, output_path, recompute_hessian, save_hessian):
	state_dict = torch.load(os.path.join(checkpoint_path, checkpoint_filename), weights_only=False)
	with open(os.path.join(checkpoint_path, "config.json"), "r") as f:
		config = json.load(f)

	model, _, _ = load_model_from_state_dict(state_dict, config)

	# Check if hessian matrix exists.
	# Load it and set it as the observed information matrix on model.
	#import pdb; pdb.set_trace()
	I = None
	if not recompute_hessian and os.path.exists(os.path.join(checkpoint_path, "hessian.csv")):
		hessian = np.loadtxt(os.path.join(checkpoint_path, "hessian.csv"), delimiter=',')
		I = torch.from_numpy(hessian).double()

	res_beta, I = inference_beta(model, var, w1, w0, config["x_map"], I)

	if output_path is None:
		output_path = os.path.join(checkpoint_path, w1 + "_" + w0)
	create_directory(output_path)
	#print(config["x_map"])

	res_beta.to_csv(os.path.join(checkpoint_path, "coefficients.csv"), index=False)

	logRR, log2RR, cov_mat, _  = inference_logRR(model, var, w1, w0, config["x_map"], I)
	logRR_std = torch.sqrt(torch.diagonal(cov_mat, dim1 = 1, dim2 = 2)).data.numpy()

	# Save Hessian for future use.
	if save_hessian:
		np.savetxt(os.path.join(checkpoint_path, "hessian.csv"), I, delimiter=',')
	np.savetxt(os.path.join(output_path, "nbsr_logRR.csv"), logRR, delimiter=',')
	np.savetxt(os.path.join(output_path, "nbsr_logRR_sd.csv"), logRR_std, delimiter=',')
	# cov_mat is NxKxK tensor. 
	np.save(os.path.join(output_path, "nbsr_logRR_cov.npy"), cov_mat.data.numpy())

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

	counts_pd = pd.read_csv(config["counts_path"])
	samples = counts_pd.columns
	features = counts_pd.index

	logRR = (logRR - log_bias)
	stat = logRR / logRR_std
	pvalue = 2 * ss.norm.cdf(-np.abs(stat))
	# Fifth column is the adjusted p-value.
	padj = np.array(list(map(lambda x: ss.false_discovery_control(x, method="bh"), pvalue)))
	
	# Output logRR, se, p-value, adjusted p-value.
	# If there is sample-level variability, output it using h5 file format.
	# Otherwise, output it as csv results file.
	with pd.HDFStore(os.path.join(output_path, "nbsr_results.h5"), mode='w') as store:
		store['logRR'] = pd.DataFrame(logRR.T, index=features, columns=samples)
		store['se'] = pd.DataFrame(logRR_std.T, index=features, columns=samples)
		store['stat'] = pd.DataFrame(stat.T, index=features, columns=samples)
		store['pvalue'] = pd.DataFrame(pvalue.T, index=features, columns=samples)
		store['padj'] = pd.DataFrame(padj.T, index=features, columns=samples)
		if absolute_fc:
			store['log_bias'] = pd.DataFrame(log_bias, index=samples)

	# If there is only one variable, then we can output it as a table.
	if np.allclose(log2RR, log2RR[0, :], atol=1e-8):
		#Output results table.
		res = pd.DataFrame({
			"feature": features,
			"log2FC": log2RR[0,:],
			"pvalue": pvalue[0,:],
			"padj": padj[0,:]})
		res.to_csv(os.path.join(output_path, "nbsr_results.csv"), index=False)

cli.add_command(eb)
cli.add_command(train)
cli.add_command(resume)
cli.add_command(results)

if __name__ == '__main__':
    cli()
