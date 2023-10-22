import os
import copy

import click
import pandas as pd
import numpy as np
import scipy
import torch

from nbsr.negbinomial_model import NegativeBinomialRegressionModel
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

def inference(model, var, w1, w0):
	"""
	Compute inference statistics for contrasting two levels for a given variable of interest.
	Note: typically w0 should correspond to the baseline corresponding to control while w1 the treatment level.
	If neither w0 nor w1 corresponds to a baseline level, then we will return beta_{w1} - beta_{w0}.
	The variance will be given by var(beta_{w1}) + var(beta_{w0}), 
	i.e., the term -2 cov(beta_{w1}, beta_{w0}) is not included in the calculation of variance.

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

	Returns
	-------
	res : pandas DataFrame
		A DataFrame with the following columns:
		- `features`: the names of the features in the model.
		- `log2Fc`: the log2 fold change between the two levels of the variable.
		- `stdErr`: the standard error of the log2 fold change.
		- `z-score`: the z-score.
		- `pValue`: the p-value.
		- `adjPValue`: the Benjamini-Hochberg adjusted p-value.
	"""
	# Compute observed information matrix.
	# Compute (pseudo) inverse of observed Fisher information matrix to get covariance matrix.
	# Compute standard errors.
	I = model.compute_observed_information(recompute=False)
	S = torch.linalg.pinv(I)

	std_err_reshaped = reshape(model, torch.sqrt(torch.diag(S))).data.numpy()
	beta_reshaped = get_beta(model).data.numpy()

	var_level0 = "{varname}_{levelname}".format(varname=var, levelname=w0)
	var_level1 = "{varname}_{levelname}".format(varname=var, levelname=w1)
	# Get a copy of the design matrix.
	X_df = pd.get_dummies(model.X_df, drop_first=True, dtype=int)
	colnames = X_df.columns.values
	col_idx0 = np.where(colnames == var_level0)[0]
	col_idx1 = np.where(colnames == var_level1)[0]
	beta_ = model.beta.reshape(model.dim, model.covariate_count)
	contrast = torch.zeros_like(beta_)
	found = False
	if len(col_idx0) > 0:
		# Offset by 1 because the first column is the intercept.
		beta0 = beta_reshaped[1+col_idx0[0],:]
		std_err0 = std_err_reshaped[1+col_idx0[0],:]
		found = True
	else:
		beta0 = 0
		std_err0 = 0
	if len(col_idx1) > 0:
		beta1 = beta_reshaped[1+col_idx1[0],:]
		std_err1 = std_err_reshaped[1+col_idx1[0],:]
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
	res["log2FC"] = logFC/np.log(2)
	# Third column is the standard error.
	res["stdErr"] = std_err/np.log(2)
	# Fourth column is the z-score.
	res["stat"] = logFC / std_err
	# Fifth column is the p-value.
	res["pvalue"] = 2 * scipy.stats.norm.cdf(-np.abs(res["stat"]))
	# Sixth column is the Benjamini-Hochberg adjusted p-value.
	res["adjPValue"] = false_discovery_control(res["pvalue"], method="bh")
	return res

def inference2(model, var, w1, w0):
	I = model.compute_observed_information(recompute=False)
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
	return (logFC, std_err)

def fit_posterior(model, optimizer, iterations, tol, lookback_iterations):
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
	coldata_pd = pd.read_csv(config["coldata_path"], na_filter=False)
	dispersion = np.loadtxt(config["dispersion_path"])

	print("Y: ", counts_pd.shape)
	X = coldata_pd[config["column_names"]]
	print("X: ", X.shape)
	print("dispersion: ", dispersion.shape)

	model = NegativeBinomialRegressionModel(X, counts_pd, dispersion=dispersion, pivot=config["pivot"])
	model.specify_beta_prior(config["lam"], config["shape"], config["scale"])

	print(torch.get_default_dtype())
	print(model.X.dtype, model.Y.dtype)
	return model

def run(state_dict, iterations, tol, lookback_iterations):
	config = state_dict["config"]
	output_path = config["output_path"]
	if not os.path.exists(output_path):
		os.makedirs(output_path)

	model = construct_model(config)
	if model_state_key in state_dict:
		checkpoint = state_dict[model_state_key]
		model.load_state_dict(checkpoint['model_state_dict'])
		optimizer = torch.optim.Adam(model.parameters())
		optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
		curr_loss_history = checkpoint['loss']
		curr_best_loss = checkpoint['best_loss']
		curr_best_state = checkpoint['best_model_state_dict']
	else:
		optimizer = torch.optim.Adam(model.parameters(),lr=config["lr"])
		curr_loss_history = []
		curr_best_loss = torch.inf
		curr_best_state = None

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
		'model_state': model_state,
        'config': config
        }, os.path.join(output_path, 'checkpoint.pth'))

	# Compute pi for each 
	model.load_state_dict(curr_best_model_state)
	pi, _ = model.predict(model.beta, model.X)
	np.savetxt(os.path.join(output_path, "nbsr_beta.csv"), model.beta.data.numpy().transpose(), delimiter=',')
	np.savetxt(os.path.join(output_path, "nbsr_beta_sd.csv"), model.softplus(model.psi.data).numpy().transpose(), delimiter=',')
	np.savetxt(os.path.join(output_path, "nbsr_pi.csv"), pi.data.numpy().transpose(), delimiter=',')
	
	print("Training iterations completed.")
	print("Converged? " + str(converged))

def get_config(data_path, cols, learning_rate, lam, shape, scale, pivot):
	config = {
		"output_path": data_path,
		"counts_path": os.path.join(data_path, "Y.csv"),
		"coldata_path": os.path.join(data_path, "X.csv"),
		"dispersion_path": os.path.join(data_path, "dispersion.csv"),
		"column_names": cols,
		"lr": learning_rate,
		"lam": lam,
		"shape": shape,
		"scale": scale,
		"pivot": pivot
	}
	return config

@click.command()
@click.argument('data_path', type=click.Path(exists=True))
@click.argument('vars', nargs=-1)
@click.option('-i', '--iterations', default=1000, type=int)
@click.option('-l', '--learning_rate', default=0.1, type=float)
@click.option('--lam', default=1., type=float)
@click.option('--shape', default=3, type=float)
@click.option('--scale', default=2, type=float)
@click.option('--pivot', default=False, type=bool)
@click.option('--tol', default=0.01, type=float)
@click.option('--lookback_iterations', default=50, type=int)
def train(data_path, vars, iterations, learning_rate, lam, shape, scale, pivot, tol, lookback_iterations):
	assert(len(vars) > 0)
	cols = list(vars)
	config = get_config(data_path, cols, learning_rate, lam, shape, scale, pivot)
	state = {"config": config}
	run(state, iterations, tol, lookback_iterations)

@click.command()
@click.argument('checkpoint_path', type=click.Path(exists=True))
@click.option('-i', '--iterations', default=1000, type=int)
@click.option('--tol', default=0.01, type=float)
@click.option('--lookback_iterations', default=50, type=int)
def resume(checkpoint_path, iterations, tol, lookback_iterations):
	checkpoint = torch.load(os.path.join(checkpoint_path, checkpoint_filename))
	run(checkpoint, iterations, tol, lookback_iterations)

cli.add_command(train)
cli.add_command(resume)

if __name__ == '__main__':
    cli()
