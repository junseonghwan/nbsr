import copy
import os
import re
import shutil
import time
from pathlib import Path

import click
import numpy as np
import pandas as pd
import scipy.optimize as so
import scipy.stats as ss
import torch

from nbsr.nbsr_config import NBSRConfig
from nbsr.negbinomial_model import NegativeBinomialRegressionModel
from nbsr.nbsr_dispersion import NBSRTrended
from nbsr.dispersion import DispersionModel
from nbsr.utils import *

torch.set_printoptions(precision=9)

checkpoint_filename = "checkpoint.pth"
model_state_key = "model_state"
hessian_filename = "hessian.npy"
covariance_path = "covariance.pth"

@click.group()
def cli():
	pass

def compute_negative_hessian_log_posterior(model, use_cuda_if_available=True):
	"""Negative Hessian of the log posterior at the fitted beta, in closed form (see utils.kron_hessian).
	Computed on the GPU when requested and available; always returned on the CPU in float64."""
	if use_cuda_if_available:
		print(f"CUDA available? {torch.cuda.is_available()}")
		model.to_device(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
	start = time.perf_counter()
	with torch.no_grad():
		I = -model.log_posterior_hessian(model.beta.detach())
	print("Hessian computation time = {}s".format(time.perf_counter() - start))
	return I.detach().cpu().double()

def cholesky_with_jitter(I, max_jitter=1e-3):
	"""Cholesky factor of a symmetric matrix that should be positive definite (a negative Hessian at a
	mode). Roundoff or an unconverged fit can make it indefinite; a small ridge is added in decades up to
	max_jitter before giving up."""
	I = 0.5 * (I + I.T)
	eye = torch.eye(I.shape[0], dtype=I.dtype, device=I.device)
	eps = 0.0
	while True:
		L, info = torch.linalg.cholesky_ex(I + eps * eye)
		if info == 0:
			if eps > 0:
				print(f"Negative Hessian needed a ridge of {eps:g} to be positive definite.")
			return L
		if eps >= max_jitter:
			raise RuntimeError(f"Negative Hessian is not positive definite even with a ridge of {max_jitter:g}; "
							   "the fit has probably not converged.")
		eps = 1e-8 if eps == 0.0 else eps * 10.0

def inference_logRR(model, var, w1, w0, x_map, I, return_cov=True):
	"""Log relative-abundance ratio of level w1 over w0 of `var` for every sample and feature, with
	delta-method standard errors from the negative Hessian I (covariate-major, index d*dim + k).

	Each logRR_nj is a function g_nj(beta); its gradient a_nj (length dim*P) gives Var(logRR_nj) = a_nj' I^-1 a_nj.
	With I = L L' this is ||L^-1 a_nj||^2, so all N*J variances come from one triangular solve. return_cov
	also forms the (N, J, J) covariance of the logRR vector within each sample.
	"""
	assert I is not None
	var_level0 = "{varname}_{levelname}".format(varname=var, levelname=w0)
	var_level1 = "{varname}_{levelname}".format(varname=var, levelname=w1)
	col_idx0 = x_map[var_level0] if var_level0 in x_map else None
	col_idx1 = x_map[var_level1] if var_level1 in x_map else None

	# Design rows with the levels of `var` set to w0 / w1 (all other covariates as observed).
	Z0 = model.X.clone()
	Z1 = model.X.clone()
	for colname, col_idx in x_map.items():
		if var in colname:  # var is a substring of colname
			Z0[:, col_idx + 1] = 0  # +1 to account for the intercept.
			Z1[:, col_idx + 1] = 0
	found = False
	if col_idx0 is not None:
		Z0[:, col_idx0 + 1] = 1
		found = True
	if col_idx1 is not None:
		Z1[:, col_idx1 + 1] = 1
		found = True
	assert found, f"{var} not found among the covariates"
	print(f"Found covariate {var} in the model.")

	pi0, _ = model.predict(model.beta, Z0)
	pi1, _ = model.predict(model.beta, Z1)
	logRRi = torch.log(pi1) - torch.log(pi0)
	log2RRi = torch.log2(pi1) - torch.log2(pi0)

	N, J = pi0.shape
	P = model.covariate_count
	d = model.dim  # J-1 if model.pivot; J otherwise.

	# Gradient of logRR_nj w.r.t. beta_{k,d}: z_{1,d} (1[j=k] - pi1_k) - z_{0,d} (1[j=k] - pi0_k), k < dim.
	identity = torch.eye(J, d, device=pi0.device, dtype=pi0.dtype).unsqueeze(0).expand(N, J, d)
	ipi0 = identity - pi0[:, :d].unsqueeze(1)                                   # (N, J, d)
	ipi1 = identity - pi1[:, :d].unsqueeze(1)
	ret = ipi1.unsqueeze(3) * Z1.unsqueeze(1).unsqueeze(2) - ipi0.unsqueeze(3) * Z0.unsqueeze(1).unsqueeze(2)
	ret = ret.transpose(2, 3).reshape(N, J, d * P)                                # covariate-major, matches I

	I = I.to(device=ret.device, dtype=ret.dtype)
	L = cholesky_with_jitter(I)
	V = torch.linalg.solve_triangular(L, ret.reshape(N * J, d * P).T, upper=False)  # (dP, N*J) = L^-1 a
	se = V.pow(2).sum(0).sqrt().reshape(N, J)
	cov_mat = None
	if return_cov:
		Vn = V.T.reshape(N, J, d * P)
		cov_mat = torch.bmm(Vn, Vn.transpose(1, 2))

	return (logRRi.detach().cpu().numpy(), log2RRi.detach().cpu().numpy(), se.detach().cpu().numpy(),
			cov_mat.detach().cpu() if cov_mat is not None else None)

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

def build_dispersion_covariates(coldata_pd, z_columns, sample_count, z_log, Y=None, z_total_counts=False):
	"""External covariates W (N, Q) of the dispersion model: columns of X.csv (--z_columns, optionally
	log-transformed) and/or log total counts per sample (--z_total_counts); None if neither."""
	parts = []
	if z_columns:
		ret = construct_tensor_from_coldata(coldata_pd, list(z_columns), sample_count, include_intercept=False)
		if ret is not None:
			W, _ = ret
			if z_log:
				assert torch.all(W > 0), "--z_log requires strictly positive dispersion covariates"
				W = torch.log(W)
			parts.append(W)
	if z_total_counts:
		assert Y is not None
		parts.append(torch.log(Y.sum(1, keepdim=True)))
	return torch.cat(parts, dim=1) if parts else None

def build_dispersion_model(config, feature_count, W):
	return DispersionModel(feature_count, W=W, sigma_b=config.sigma_b,
						   b_pi_prior=(config.b_pi_prior_mean, config.b_pi_prior_sd),
						   sigma_bj_prior_sd=config.sigma_bj_prior_sd,
						   link=config.dispersion_link, feature_offsets=config.feature_offsets,
						   estimate_sd=config.estimate_dispersion_sd)

def construct_model(config):
	#click.echo(config)
	counts_pd = pd.read_csv(config.counts_path, index_col=0)
	if os.path.exists(config.coldata_path):
		coldata_pd = pd.read_csv(config.coldata_path, na_filter=False, skipinitialspace=True)
	else:
		coldata_pd = None
	Y = torch.tensor(counts_pd.transpose().to_numpy(), dtype=torch.float64)  # float32 cannot represent counts above 2^24 exactly
	X, x_map = construct_tensor_from_coldata(coldata_pd, config.column_names, counts_pd.shape[1])
	W = build_dispersion_covariates(coldata_pd, config.z_columns, counts_pd.shape[1], config.z_log, Y, config.z_total_counts)

	# Allow fixed dispersion values to be passed in.
	dispersion = read_file_if_exists(config.dispersion_path)
	dispersion_model_path = Path(config.output_path) / config.dispersion_model_file if config.dispersion_model_file is not None else None
	trended = config.trended_dispersion

	print("Y: ", Y.shape)
	print("X: ", X.shape)
	if W is not None:
		print("W: ", W.shape)

	pivot = config.pivot
	beta_prior_sd = config.beta_prior_sd if config.beta_prior_sd is not None else config.stage1_prior_sd
	disp_model = None
	if dispersion is not None:
		print("Run NBSR with pre-specified dispersion values.")
		model = NegativeBinomialRegressionModel(X, Y, beta_prior_sd=beta_prior_sd, dispersion_prior=disp_model, dispersion=dispersion, pivot=pivot)
	else:
		if dispersion_model_path is not None:
			print(f"Dispersion prior model is specified. Loading from {dispersion_model_path}")
			disp_model = torch.load(dispersion_model_path, weights_only=False)
		if trended:
			if disp_model is None:
				print(f"Dispersion trend will be estimated.")
				disp_model = build_dispersion_model(config, Y.shape[1], W)
			model = NBSRTrended(X, Y, disp_model, beta_prior_sd=beta_prior_sd, pivot=pivot)
		else:
			print("Run NBSR with shared dispersion per feature.")
			model = NegativeBinomialRegressionModel(X, Y, beta_prior_sd=beta_prior_sd, dispersion_prior=disp_model, dispersion=None, pivot=pivot)

	param_list = []
	print("Parameters being optimized:")
	for name, param in model.named_parameters():
		print(name)
		if "disp_model" in name and dispersion_model_path is not None and not config.update_dispersion:
			continue  # a pre-fitted dispersion model is held fixed unless update_dispersion is set.
		param_list.append(param)

	#model.specify_beta_prior(config.lam, config.shape, config.scale)
	#print(torch.get_default_dtype())
	#print(model.X.dtype, model.Y.dtype)
	return model, param_list, x_map

def load_model_from_state_dict(config, state_dict):
    model, params, x_map = construct_model(config)
    if model_state_key in state_dict:
        print("Loading previously saved model...")
        model.load_state_dict(_upgrade_state_dict(state_dict[model_state_key]['model_state_dict']))
    elif config.init_from is not None:
        # Warm start: beta and the dispersion-model parameters of another fit (e.g. the stage-1 fit); the beta
        # prior sd is left as constructed, since that is what differs between the stages.
        source = torch.load(Path(config.init_from), weights_only=False)[model_state_key]['model_state_dict']
        own = model.state_dict()
        transfer = {k: v for k, v in source.items() if k != "beta_prior_sd" and k in own and own[k].shape == v.shape}
        model.load_state_dict(transfer, strict=False)
        print(f"Initialised {sorted(transfer)} from {config.init_from}")
    return model, params, x_map

def _upgrade_state_dict(sd):
	"""Translate checkpoints written before the prior sd became a fixed buffer: psi (softplus-parameterized
	learned sd) becomes beta_prior_sd, and the retired lam / inverse-gamma hyperprior buffers are dropped."""
	sd = dict(sd)
	if "psi" in sd and "beta_prior_sd" not in sd:
		sd["beta_prior_sd"] = torch.nn.functional.softplus(sd.pop("psi"))
		print("checkpoint: converted learned psi to a fixed beta_prior_sd")
	for k in ["lam", "beta_var_shape", "beta_var_scale"]:
		sd.pop(k, None)
	return sd

def empirical_prior_sd(beta, hessian, covariate_count, quantile=0.95, max_abs=10.0):
	"""DESeq2-style prior width per covariate from a (near-)MLE fit: the sd of a zero-centred normal whose
	upper tail matches the precision-weighted `quantile` of |beta| over the features, after dropping
	|beta| > max_abs as non-converged. beta is the flat covariate-major vector, hessian the negative Hessian."""
	dim = beta.size // covariate_count
	b = beta.reshape(covariate_count, dim)
	se = np.sqrt(np.diag(np.linalg.inv(hessian))).reshape(covariate_count, dim)
	z = ss.norm.ppf(0.5 + quantile / 2)
	sds = []
	for d in range(covariate_count):
		ok = np.abs(b[d]) < max_abs
		w = 1.0 / np.maximum(se[d][ok], np.median(se[d][ok])) ** 2   # cap the weight of well-measured features
		w = w / w.sum()
		order = np.argsort(np.abs(b[d][ok]))
		q = np.abs(b[d][ok])[order][min(np.searchsorted(np.cumsum(w[order]), quantile), ok.sum() - 1)]
		sds.append(float(q / z))
	return sds

def run(config):
	"""Fit the model of `config`; with beta_prior == "empirical" run a wide-prior stage-1 fit first and set the
	prior sd of each covariate from it before the main fit."""
	if config.beta_prior_sd is None:   # empirical: stage-1 fit with a wide prior sets the sd per covariate
		if config.dispersion_model_file is not None:
			# Resolve against the main output directory now, since stage 1 writes to a subdirectory of it.
			config = copy.deepcopy(config)
			config.dispersion_model_file = str(Path(config.output_path) / config.dispersion_model_file)
		stage1 = copy.deepcopy(config)
		stage1.beta_prior = "fixed"
		stage1.beta_prior_sd = [config.stage1_prior_sd]
		stage1.output_path = Path(config.output_path) / "stage1"
		stage1.iterations = config.stage1_iterations or max(config.iterations // 2, 1)
		stage1.init_from = None
		print(f"Stage 1: wide prior (sd {config.stage1_prior_sd}) for {stage1.iterations} iterations.")
		_, model1 = _run_single(stage1)
		beta1 = model1.beta.detach().cpu().numpy()
		I1 = np.load(stage1.output_path / hessian_filename)
		sds = empirical_prior_sd(beta1, I1, model1.covariate_count, config.prior_quantile)
		print(f"Empirical prior sd per covariate (quantile {config.prior_quantile}): {np.round(sds, 4).tolist()}")
		main = copy.deepcopy(config)
		main.beta_prior = "fixed"
		main.beta_prior_sd = sds
		main.init_from = stage1.output_path / checkpoint_filename
		return _run_single(main)
	return _run_single(config)

def _run_single(config):
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
		phi = model.dispersion(pi)
		save_dispersion_model_outputs(model.disp_model, output_path)
	else:
		phi = model.softplus(model.phi)

	np.savetxt(output_path / "nbsr_beta.csv", model.beta.data.numpy().transpose(), delimiter=',')
	np.savetxt(output_path / "nbsr_beta_sd.csv", model.beta_prior_sd.cpu().numpy(), delimiter=',')
	np.savetxt(output_path / "nbsr_pi.csv", pi.data.numpy().transpose(), delimiter=',')
	np.savetxt(output_path / "nbsr_dispersion.csv", phi.data.numpy().transpose(), delimiter=',')

	torch.save(state_dict, output_path / checkpoint_filename)
	config.dump_json(output_path / "config.json")

	print("Compute negative Hessian matrix.")
	I = compute_negative_hessian_log_posterior(model, config.use_cuda_if_available)
	np.save(output_path / hessian_filename, I.detach().cpu().numpy())

	return(curr_loss_history, model)

def save_dispersion_model_outputs(disp_model, output_path):
	"""Write the dispersion model's parameters next to the other outputs:
	nbsr_dispersion_params.csv (b_0, b_pi, sigma_bj), nbsr_dispersion_bj.csv (per feature) and
	nbsr_dispersion_bw.csv (per external covariate, if any)."""
	output_path = Path(output_path)
	with torch.no_grad():
		pd.DataFrame({"b_0": disp_model.b_0.numpy(), "b_pi": disp_model.b_pi.numpy(),
					  "sigma_bj": disp_model.sigma_bj.numpy()}).to_csv(output_path / "nbsr_dispersion_params.csv", index=False)
		np.savetxt(output_path / "nbsr_dispersion_bj.csv", disp_model.b_j.numpy(), delimiter=',')
		if disp_model.covariate_count:
			np.savetxt(output_path / "nbsr_dispersion_bw.csv", disp_model.b_w.numpy(), delimiter=',')
		if disp_model.estimate_sd:
			np.savetxt(output_path / "nbsr_dispersion_sd.csv", disp_model.get_sd().numpy(), delimiter=',')

def _safe_name(x):
    """Convert arbitrary string to filesystem-safe name."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(x))

def generate_results(results_path, var, w1, w0, absolute_fc=True, recompute_hessian=False, save_cov=True):
	results_path = Path(results_path)
	
	comparison_name = f"{_safe_name(var)}__{_safe_name(w1)}_vs_{_safe_name(w0)}"
	comparison_path = results_path / comparison_name
	comparison_path.mkdir(parents=True, exist_ok=True)

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
	if not recompute_hessian and os.path.exists(results_path / hessian_filename):
		I = torch.from_numpy(np.load(results_path / hessian_filename)).double()
	else: # compute Hessian
		print("Compute negative hessian matrix.")
		I = compute_negative_hessian_log_posterior(model, config.use_cuda_if_available).detach().cpu()
		np.save(results_path / hessian_filename, I)

	logRR, log2RR, logRR_std, cov_mat = inference_logRR(model, var, w1, w0, x_map, I, return_cov=save_cov)
	#logRR_std = torch.sqrt(torch.diagonal(cov_mat, dim1 = 1, dim2 = 2)).data.numpy()

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

	if cov_mat is not None:
		torch.save(cov_mat, comparison_path / covariance_path)

	# Output logRR, se, p-value, adjusted p-value.
	# Output using h5 file format.
	with pd.HDFStore(comparison_path / "nbsr_results.h5", mode='w') as store:
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
		res.to_csv(comparison_path / "nbsr_results.csv", index=False)
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
@click.option('--estimate_dispersion_sd', is_flag=True, show_default=False, default=False, type=bool)
@click.option('--update_dispersion', is_flag=True, show_default=False, default=False, type=bool)
@click.option('--z_columns', multiple=True, help="Columns of X.csv used as external covariates of the dispersion model.")
@click.option('--z_log', is_flag=True, default=False, help="Log-transform the --z_columns (e.g. library sizes) before use.")
@click.option('--b_pi_prior', nargs=2, type=float, default=(1.0, 0.1), show_default=True, help="Mean and sd of the normal prior on b_pi (dispersion vs logit composition).")
@click.option('--sigma_bj_prior_sd', type=float, default=0.5, show_default=True, help="Scale of the half-normal prior on sigma_bj (per-feature dispersion offsets).")
@click.option('--sigma_b', type=float, default=1.0, show_default=True, help="Prior sd of the coefficients on --z_columns.")
@click.option('--dispersion_link', type=click.Choice(["logit", "log"]), default="logit", show_default=True, help="Transform of pi in the dispersion model: logit (NBSR-HMC) or log (previous NBSR model).")
@click.option('--no_feature_offsets', is_flag=True, default=False, help="Drop the per-feature offsets b_j from the dispersion model.")
@click.option('--z_total_counts', is_flag=True, default=False, help="Add log total counts per sample as an external dispersion covariate (the log R_i term of the previous model).")
@click.option('--beta_prior_sd', multiple=True, type=float, help="Fix the prior sd of beta (one value, or one per covariate incl. the intercept). Default: empirical, set per covariate from a wide-prior stage-1 fit by matching the upper quantile of |beta| (DESeq2-style).")
@click.option('--stage1_prior_sd', type=float, default=10.0, show_default=True, help="Prior sd of every covariate in the wide-prior stage-1 fit (empirical prior).")
@click.option('--stage1_iterations', type=int, default=None, help="Iterations of the stage-1 fit (default: half of -i).")
@click.option('--prior_quantile', type=float, default=0.95, show_default=True, help="Quantile of |beta| matched to the prior tail (empirical prior).")
@click.option('--pivot', is_flag=True, show_default=True, default=False, type=bool)
def eb(data_path, vars, mu_file, iterations, lr, eb_iter, eb_lr, estimate_dispersion_sd, update_dispersion, z_columns, z_log, b_pi_prior, sigma_bj_prior_sd, sigma_b, dispersion_link, no_feature_offsets, z_total_counts, beta_prior_sd, stage1_prior_sd, stage1_iterations, prior_quantile, pivot):
	"""Empirical-Bayes workflow: fit the dispersion model to DESeq2's fitted means, then run NBSR with
	that dispersion model (fixed unless --update_dispersion)."""
	data_path = Path(data_path)
	column_names = list(vars)
	config = NBSRConfig(counts_path=data_path / "Y.csv",
						coldata_path=data_path / "X.csv",
						output_path=data_path,  # output to where the data is.
						column_names=column_names,
						z_columns=list(z_columns),
						z_log=z_log,
						b_pi_prior_mean=b_pi_prior[0], b_pi_prior_sd=b_pi_prior[1],
						sigma_bj_prior_sd=sigma_bj_prior_sd, sigma_b=sigma_b,
							dispersion_link=dispersion_link, feature_offsets=not no_feature_offsets, z_total_counts=z_total_counts,
						lr=lr,
						iterations=iterations,
						estimate_dispersion_sd=estimate_dispersion_sd,
						trended_dispersion=True,
						dispersion_model_file="disp_model.pth",
						update_dispersion=update_dispersion,
						beta_prior_sd=list(beta_prior_sd) or None,
						beta_prior="fixed" if beta_prior_sd else "empirical",
						stage1_prior_sd=stage1_prior_sd, stage1_iterations=stage1_iterations, prior_quantile=prior_quantile,
						pivot=pivot)

	print("Performing Empirical Bayes estimation of dispersion.")
	mu_hat = torch.tensor(pd.read_csv(data_path / mu_file).transpose().to_numpy(), dtype=torch.float64)
	pi_hat = mu_hat / mu_hat.sum(dim=1, keepdim=True)
	counts_pd = pd.read_csv(config.counts_path, index_col=0)
	coldata_pd = pd.read_csv(config.coldata_path, na_filter=False, skipinitialspace=True) if os.path.exists(config.coldata_path) else None
	Y = torch.tensor(counts_pd.transpose().to_numpy(), dtype=torch.float64)
	X, _ = construct_tensor_from_coldata(coldata_pd, column_names, counts_pd.shape[1])
	W = build_dispersion_covariates(coldata_pd, z_columns, counts_pd.shape[1], z_log, Y, z_total_counts)
	disp_model = build_dispersion_model(config, Y.shape[1], W)
	nbsr_model = NBSRTrended(X, Y, disp_model=disp_model, beta_prior_sd=stage1_prior_sd, pivot=pivot)
	fit_dispersion_model(nbsr_model, pi_hat, eb_iter, eb_lr)

	phi = torch.exp(disp_model.log_dispersion(pi_hat))
	np.savetxt(data_path / "eb_dispersion.csv", phi.data.numpy().transpose(), delimiter=',')
	save_dispersion_model_outputs(disp_model, data_path)
	torch.save(disp_model, data_path / config.dispersion_model_file)

	print("Optimizing NBSR parameters with the dispersion model" + (" (updated jointly)." if update_dispersion else " held fixed."))
	run(config)

def fit_dispersion_model(nbsr_model, pi_hat, iterations, lr):
	"""Fit the dispersion model's parameters by maximizing the NB log-likelihood of the counts at the given
	composition pi_hat (e.g. from DESeq2 fitted means) plus the dispersion model's prior."""
	disp_model = nbsr_model.disp_model
	optimizer = torch.optim.Adam(disp_model.parameters(), lr=lr)
	for i in range(iterations):
		phi = torch.exp(disp_model.log_dispersion(pi_hat))
		loss = -(nbsr_model.log_likelihood(pi_hat, phi) + disp_model.log_prior())
		if loss.isnan():
			print("nan")
			break
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		if i % 100 == 0:
			print("Iter:", i)
			print(loss.data)

@click.command()
@click.argument('data_path', type=click.Path(exists=True))
@click.argument('vars', nargs=-1)
@click.option('-i', '--iterations', default=10000, type=int)
@click.option('-l', '--lr', default=0.05, type=float, help="NBSR model parameters learning rate.")
@click.option('-r', '--runs', default=1, type= int, help="Number of optimization runs (initialization).")
@click.option('--z_columns', multiple=True, help="Columns of X.csv used as external covariates of the dispersion model (with --trended_dispersion).")
@click.option('--z_log', is_flag=True, default=False, help="Log-transform the --z_columns (e.g. library sizes) before use.")
@click.option('--b_pi_prior', nargs=2, type=float, default=(1.0, 0.1), show_default=True, help="Mean and sd of the normal prior on b_pi (dispersion vs logit composition).")
@click.option('--sigma_bj_prior_sd', type=float, default=0.5, show_default=True, help="Scale of the half-normal prior on sigma_bj (per-feature dispersion offsets).")
@click.option('--sigma_b', type=float, default=1.0, show_default=True, help="Prior sd of the coefficients on --z_columns.")
@click.option('--dispersion_link', type=click.Choice(["logit", "log"]), default="logit", show_default=True, help="Transform of pi in the dispersion model: logit (NBSR-HMC) or log (previous NBSR model).")
@click.option('--no_feature_offsets', is_flag=True, default=False, help="Drop the per-feature offsets b_j from the dispersion model.")
@click.option('--z_total_counts', is_flag=True, default=False, help="Add log total counts per sample as an external dispersion covariate (the log R_i term of the previous model).")
@click.option('--dispersion_model_file', default=None, type=str)
@click.option('--trended_dispersion', is_flag=True, show_default=True, default=False, type=bool)
@click.option('--estimate_dispersion_sd', is_flag=True, show_default=False, default=False, type=bool)
@click.option('--beta_prior_sd', multiple=True, type=float, help="Fix the prior sd of beta (one value, or one per covariate incl. the intercept). Default: empirical, set per covariate from a wide-prior stage-1 fit by matching the upper quantile of |beta| (DESeq2-style).")
@click.option('--stage1_prior_sd', type=float, default=10.0, show_default=True, help="Prior sd of every covariate in the wide-prior stage-1 fit (empirical prior).")
@click.option('--stage1_iterations', type=int, default=None, help="Iterations of the stage-1 fit (default: half of -i).")
@click.option('--prior_quantile', type=float, default=0.95, show_default=True, help="Quantile of |beta| matched to the prior tail (empirical prior).")
@click.option('--pivot', is_flag=True, show_default=True, default=False, type=bool)
def train(data_path, vars, iterations, lr, runs, z_columns, z_log, b_pi_prior, sigma_bj_prior_sd, sigma_b, dispersion_link, no_feature_offsets, z_total_counts, dispersion_model_file, trended_dispersion, estimate_dispersion_sd, beta_prior_sd, stage1_prior_sd, stage1_iterations, prior_quantile, pivot):

	data_path = Path(data_path)
	losses = []
	for run_no in range(runs):
		outpath = data_path / ("run" + str(run_no))
		config = NBSRConfig(counts_path=data_path / "Y.csv",
					  		coldata_path=data_path / "X.csv",
							output_path=outpath,
					  		column_names=list(vars),
							z_columns=list(z_columns),
							z_log=z_log,
							b_pi_prior_mean=b_pi_prior[0], b_pi_prior_sd=b_pi_prior[1],
							sigma_bj_prior_sd=sigma_bj_prior_sd, sigma_b=sigma_b,
							dispersion_link=dispersion_link, feature_offsets=not no_feature_offsets, z_total_counts=z_total_counts,
							lr=lr,
							iterations=iterations,
							estimate_dispersion_sd=estimate_dispersion_sd,
							trended_dispersion=trended_dispersion,
							dispersion_model_file=dispersion_model_file,
							beta_prior_sd=list(beta_prior_sd) or None,
							beta_prior="fixed" if beta_prior_sd else "empirical",
							stage1_prior_sd=stage1_prior_sd, stage1_iterations=stage1_iterations, prior_quantile=prior_quantile,
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
@click.option('--skip_cov', is_flag=True, show_default=True, default=False, type=bool, help="Do not compute/save the per-sample covariance of logRR across features.")
def results(checkpoint_path, var, w1, w0, absolute_fc, recompute_hessian, skip_cov):
	generate_results(checkpoint_path, var, w1, w0, absolute_fc, recompute_hessian, save_cov=not skip_cov)


cli.add_command(eb)
cli.add_command(train)
cli.add_command(results)

if __name__ == '__main__':
    cli()

