import os
import copy

import click
import pandas as pd
import numpy as np
import scipy
import torch

from nbsr.negbinomial_model import NegativeBinomialRegressionModel
from nbsr.zinbsr import ZINBSR
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
	
def compute_observed_information(model):
	print("Computeing Hessian...")
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

def inference_beta(model, var, w1, w0, x_map):
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
		- `logFC`: the natural logarithm fold change between the two levels of the variable.
		- `stdErr`: the standard error of the log2 fold change.
		- `z-score`: the z-score.
		- `pValue`: the p-value.
		- `adjPValue`: the Benjamini-Hochberg adjusted p-value.
	"""
	# Compute observed information matrix.
	# Compute (pseudo) inverse of observed Fisher information matrix to get covariance matrix.
	# Compute standard errors.
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

	logFC = (beta1 - beta0)
	std_err = (std_err0**2) + (std_err1**2)

	# Create a data frame with the results.
	res = pd.DataFrame()
	# First column is the variable name.
	res["features"] = model.Y_df.index.to_list()
	# Second column is the log2 fold change.
	res["logFC"] = logFC/np.log(2)
	# Third column is the standard error.
	res["stdErr"] = std_err/np.log(2)
	# Fourth column is the z-score.
	res["stat"] = logFC / std_err
	# Fifth column is the p-value.
	res["pvalue"] = 2 * scipy.stats.norm.cdf(-np.abs(res["stat"]))
	# Sixth column is the Benjamini-Hochberg adjusted p-value.
	#res["adjPValue"] = false_discovery_control(res["pvalue"], method="bh")
	return (res, I)

def inference_logRR(model, var, w1, w0):
	I = model.compute_observed_information()
	S = torch.linalg.pinv(I)

	X_df = pd.get_dummies(model.X_df, drop_first=True, dtype=int)
	colnames = X_df.columns.values
	Z0 = model.X.clone()
	Z1 = model.X.clone()

	var_level0 = "{varname}_{levelname}".format(varname=var, levelname=w0)
	var_level1 = "{varname}_{levelname}".format(varname=var, levelname=w1)
	col_idx0 = np.where(colnames == var_level0)[0]
	col_idx1 = np.where(colnames == var_level1)[0]
	# Zero out all columns of Z0,Z1 corresponding to var.
	for i,colname in enumerate(colnames):
		if var in colname: # Checks if var is a substring of colname.
			Z0[:,i+1] = 0
			Z1[:,i+1] = 0

	if len(col_idx0) > 0:
		Z0[:,col_idx0[0]+1] = 1
	if len(col_idx1) > 0:
		Z1[:,col_idx1[0]+1] = 1

	pi0, _ = model.predict(model.beta, Z0)
	pi1, _ = model.predict(model.beta, Z1)
	logRRi = torch.log(pi1) - torch.log(pi0)

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
	std_err = torch.sqrt(torch.diagonal(cov_mat, dim1 = 1, dim2 = 2)).data.numpy()
	return (logFC, std_err, I)

def fit_posterior(model, optimizer, optimizer_grbf, iterations, tol, lookback_iterations):
	# Fit the model.
	loss_history = []
	# We will store the best solution.
	best_model_state = None
	best_loss = torch.inf
	for i in range(iterations):

		loss = -model.log_posterior(model.beta)
		optimizer.zero_grad()
		loss.backward(retain_graph=True)
		optimizer.step()

		if optimizer_grbf is not None:
			loss = -model.log_posterior(model.beta)
			optimizer_grbf.zero_grad()
			loss.backward()
			optimizer_grbf.step()

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
		coldata_pd = pd.read_csv(config["coldata_path"], na_filter=False)
	else:
		coldata_pd = None
	Y = torch.tensor(counts_pd.transpose().to_numpy(), dtype=torch.float64)
	X, x_map = construct_tensor_from_coldata(coldata_pd, config["column_names"], counts_pd.shape[1])
	Z, z_map = construct_tensor_from_coldata(coldata_pd, config["z_columns"], counts_pd.shape[1], False)
	dispersion = read_file_if_exists(config["dispersion_path"])
	prior_sd = read_file_if_exists(config["prior_sd_path"])
	config["x_map"] = x_map
	config["z_map"] = z_map

	# Construct Gaussian RBF prior on the dispersion parameter if grbf is specified.

	print("Y: ", Y.shape)
	print("X: ", X.shape)
	if Z is not None:
		print("Z: ", Z.shape)

	if dispersion is not None:
		print("dispersion: ", dispersion.shape)
	else:
		print("dispersion unspecified and will be estimated.")
	if prior_sd is not None:
		print("prior_sd: ", prior_sd.shape)
	else:
		print("prior_sd unspecified and will be estimated.")

	if Z is None:
		model = NegativeBinomialRegressionModel(X, Y, dispersion=dispersion, prior_sd=prior_sd, pivot=config["pivot"])
	else:
		model = ZINBSR(X, Y, Z, config["logistic_max"], dispersion, prior_sd=prior_sd, pivot=config["pivot"])
	model.specify_beta_prior(config["lam"], config["shape"], config["scale"])

	print(torch.get_default_dtype())
	print(model.X.dtype, model.Y.dtype)
	return model

def load_model_from_state_dict(state_dict, config):
    model = construct_model(config)
    checkpoint = None
    if model_state_key in state_dict:
        print("Loading previously saved model...")
        checkpoint = state_dict[model_state_key]
        model.load_state_dict(checkpoint['model_state_dict'])
    return model, checkpoint

def run(state_dict, iterations, tol, lookback_iterations):
	config = state_dict["config"]
	output_path = config["output_path"]
	create_directory(output_path)

	lr = config["lr"]
	grbf_lr = config["grbf_lr"]

	model, checkpoint = load_model_from_state_dict(state_dict, config)

	# Initialize optimizers.
	if checkpoint:
		optimizer = torch.optim.Adam([model.beta, model.psi, model.b], lr = lr)
		optimizer_grbf = torch.optim.Adam(model.grbf.parameters(), lr = grbf_lr)

		optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
		optimizer_grbf.load_state_dict(checkpoint['optimizer_grbf_state_dict'])
		curr_loss_history = checkpoint['loss']
		curr_best_loss = checkpoint['best_loss']
		curr_best_model_state = checkpoint['best_model_state_dict']
	else:
		# if isinstance(model, ZINBSR):
		# 	if model.grbf is None:
		# 		optimizer = torch.optim.Adam([model.beta, model.phi, model.psi, model.b], lr = lr)
		# 		optimizer_grbf = None
		# 	else:
		# 		optimizer = torch.optim.Adam([model.beta, model.psi, model.b], lr = lr)
		# 		optimizer_grbf = torch.optim.Adam(model.grbf.parameters(), lr = grbf_lr)
		# else:
		# 	optimizer = torch.optim.Adam(model.parameters(), lr = lr)
		# 	optimizer_grbf = None
		optimizer = torch.optim.Adam(model.parameters(), lr = lr)
		optimizer_grbf = None
		curr_loss_history = []
		curr_best_loss = torch.inf
		curr_best_model_state = None

	loss_history, best_model_state, best_loss, converged = fit_posterior(model, optimizer, optimizer_grbf, iterations, tol, lookback_iterations)
	curr_loss_history.extend(loss_history)
	if best_loss < curr_best_loss:
		curr_best_loss = best_loss
		curr_best_model_state = best_model_state

	model_state = {
        	'model_state_dict': model.state_dict(),
	        'best_model_state_dict': curr_best_model_state,
	        'optimizer_state_dict': optimizer.state_dict(),
			'optimizer_grbf_state_dict': optimizer_grbf.state_dict() if optimizer_grbf else None,
	        'loss': curr_loss_history,
	        'best_loss': curr_best_loss,
	        'converged': converged
	}
	torch.save({
		'model_state': model_state,
        'config': config
        }, os.path.join(output_path, 'checkpoint.pth'))

	model.load_state_dict(curr_best_model_state)
	pi, _ = model.predict(model.beta, model.X)
	#s = torch.sum(model.Y, 1)
	mu = model.s[:,None] * pi
	phi = model.softplus(model.phi)

	if isinstance(model, ZINBSR):
		np.savetxt(os.path.join(output_path, "nbsr_zinb_coef.csv"), model.b.data.numpy().transpose(), delimiter=',')

	np.savetxt(os.path.join(output_path, "nbsr_beta.csv"), model.beta.data.numpy().transpose(), delimiter=',')
	np.savetxt(os.path.join(output_path, "nbsr_beta_sd.csv"), model.softplus(model.psi.data).numpy().transpose(), delimiter=',')
	np.savetxt(os.path.join(output_path, "nbsr_pi.csv"), pi.data.numpy().transpose(), delimiter=',')
	np.savetxt(os.path.join(output_path, "nbsr_dispersion.csv"), phi.data.numpy().transpose(), delimiter=',')
	
	print("Training iterations completed.")
	print("Converged? " + str(converged))

def get_config(data_path, cols, z_cols, lr, grbf_lr, logistic_max, lam, shape, scale, grbf, pivot):
	config = {
		"output_path": data_path,
		"counts_path": os.path.join(data_path, "Y.csv"),
		"coldata_path": os.path.join(data_path, "X.csv"),
		"dispersion_path": os.path.join(data_path, "dispersion.csv"),
		"prior_sd_path": os.path.join(data_path, "prior_sd.csv"),
		"column_names": cols,
		"z_columns": z_cols,
		"lr": lr,
		"grbf_lr": grbf_lr,
		"logistic_max": logistic_max,
		"lam": lam,
		"shape": shape,
		"scale": scale,
		"grbf": grbf,
		"pivot": pivot
	}
	return config

@click.command()
@click.argument('data_path', type=click.Path(exists=True))
@click.argument('vars', nargs=-1)
@click.option('-i', '--iterations', default=1000, type=int)
@click.option('-l', '--lr', default=0.05, type=float, help="NBSR model parameters learning rate.")
@click.option('-r', '--grbf_lr', default=0.01, type=float, help="GRBF parameters learning rate.")
@click.option('-L', '--logistic_max', default=0.2, type=float, help="ZINBSR max value for logistic function.")
@click.option('--z_columns', multiple=True, help="Enter list of strings specifying the covariate names to use for zero inflation.")
@click.option('--lam', default=1., type=float)
@click.option('--shape', default=3, type=float)
@click.option('--scale', default=2, type=float)
@click.option('--grbf', default=False, type=bool)
@click.option('--pivot', default=False, type=bool)
@click.option('--tol', default=0.01, type=float)
@click.option('--lookback_iterations', default=50, type=int)
def train(data_path, vars, iterations, lr, grbf_lr, logistic_max, z_columns, lam, shape, scale, grbf, pivot, tol, lookback_iterations):
	config = get_config(data_path, list(vars), list(z_columns), lr, grbf_lr, logistic_max, lam, shape, scale, grbf, pivot)
	state = {"config": config}
	run(state, iterations, tol, lookback_iterations)

@click.command()
@click.argument('checkpoint_path', type=click.Path(exists=True))
@click.option('-i', '--iterations', default=1000, type=int)
@click.option('--tol', default=0.01, type=float)
@click.option('--lookback_iterations', default=50, type=int)
def resume(checkpoint_path, iterations, tol, lookback_iterations):
	state_dict = torch.load(os.path.join(checkpoint_path, checkpoint_filename))
	run(state_dict, iterations, tol, lookback_iterations)

@click.command()
@click.argument('checkpoint_path', type=click.Path(exists=True))
@click.argument('var', type=str)
@click.argument('w1', type=str)
@click.argument('w0', type=str)
@click.option('--recompute_hessian', default=False, type=bool)
def results(checkpoint_path, var, w1, w0, recompute_hessian):
	state_dict = torch.load(os.path.join(checkpoint_path, checkpoint_filename))
	config = state_dict["config"]
	output_path = config["output_path"]
	create_directory(output_path)

	model, _ = load_model_from_state_dict(state_dict, config)

	# Check if hessian matrix exists.
	# Load it and set it as the observed information matrix on model.
	if not recompute_hessian and os.path.exists(os.path.join(output_path, "hessian.csv")):
		hessian = np.loadtxt(os.path.join(output_path, "hessian.csv"), delimiter=',')
		model.hessian = torch.from_numpy(hessian).double()

	res_beta, hessian = inference_beta(model, var, w1, w0, config["x_map"])
	logRR, logRR_std, _  = inference_logRR(model, var, w1, w0, config["x_map"])

	res_beta.to_csv(os.path.join(output_path, "nbsr_results.csv"), index=False)
	if not os.path.exists(os.path.join(output_path, "hessian.csv")):
		np.savetxt(os.path.join(output_path, "hessian.csv"), hessian, delimiter=',')
	np.savetxt(os.path.join(output_path, "nbsr_logRR.csv"), logRR, delimiter=',')
	np.savetxt(os.path.join(output_path, "nbsr_logRR_sd.csv"), logRR_std, delimiter=',')

cli.add_command(train)
cli.add_command(resume)
cli.add_command(results)

if __name__ == '__main__':
    cli()
