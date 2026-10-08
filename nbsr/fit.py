"""Model construction and posterior-mode fitting for NBSR.

`fit_once(config)` fits one model (given prior sds and dispersion settings) and writes its outputs; the
two-stage empirical prior and the `prior.json` handling live in nbsr.prior, inference in nbsr.inference.
"""
import copy
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from nbsr.dispersion import DispersionModel
from nbsr.legacy import convert_legacy_dispersion_model, upgrade_state_dict
from nbsr.nbsr_config import NBSRConfig
from nbsr.nbsr_dispersion import NBSRTrended
from nbsr.negbinomial_model import NegativeBinomialRegressionModel
from nbsr.utils import construct_tensor_from_coldata, create_directory, read_file_if_exists

checkpoint_filename = "checkpoint.pth"
model_state_key = "model_state"
hessian_filename = "hessian.npy"


# ---------------------------------------------------------------------------- data and model

def load_data(config):
    """Counts (features x samples in the file, returned as Y (N, J)), metadata, design X (N, P) with its
    column map, and the dispersion covariates W (N, Q) or None."""
    counts_pd = pd.read_csv(config.counts_path, index_col=0)
    coldata_pd = (pd.read_csv(config.coldata_path, na_filter=False, skipinitialspace=True)
                  if os.path.exists(config.coldata_path) else None)
    Y = torch.tensor(counts_pd.transpose().to_numpy(), dtype=torch.float64)  # float32 cannot represent counts above 2^24
    X, x_map = construct_tensor_from_coldata(coldata_pd, config.column_names, counts_pd.shape[1])
    W = build_dispersion_covariates(coldata_pd, config.z_columns, counts_pd.shape[1], config.z_log, Y, config.z_total_counts)
    return counts_pd, coldata_pd, Y, X, x_map, W


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
    """Build the model of `config` and the list of parameters to optimise."""
    counts_pd, coldata_pd, Y, X, x_map, W = load_data(config)
    dispersion = read_file_if_exists(config.dispersion_path)  # fixed per-feature dispersions, if given
    dispersion_model_path = Path(config.dispersion_model_file) if config.dispersion_model_file is not None else None
    if dispersion_model_path is not None and not dispersion_model_path.is_absolute():
        dispersion_model_path = Path(config.output_path) / dispersion_model_path

    print("Y: ", Y.shape)
    print("X: ", X.shape)
    if W is not None:
        print("W: ", W.shape)

    pivot = config.pivot
    beta_prior_sd = config.beta_prior_sd if config.beta_prior_sd is not None else config.stage1_prior_sd
    disp_model = None
    if dispersion is not None:
        print("Run NBSR with pre-specified dispersion values.")
        model = NegativeBinomialRegressionModel(X, Y, beta_prior_sd=beta_prior_sd, dispersion=dispersion, pivot=pivot)
    else:
        if dispersion_model_path is not None:
            print(f"Loading the dispersion model from {dispersion_model_path}")
            disp_model = torch.load(dispersion_model_path, weights_only=False)
            if hasattr(disp_model, "b0"):   # pickled previous dispersion model
                disp_model = convert_legacy_dispersion_model(disp_model, Y)
        if config.trended_dispersion:
            if disp_model is None:
                print("Dispersion trend will be estimated.")
                disp_model = build_dispersion_model(config, Y.shape[1], W)
            model = NBSRTrended(X, Y, disp_model, beta_prior_sd=beta_prior_sd, pivot=pivot)
        else:
            print("Run NBSR with one free dispersion per feature.")
            model = NegativeBinomialRegressionModel(X, Y, beta_prior_sd=beta_prior_sd, dispersion_prior=disp_model,
                                                    dispersion=None, pivot=pivot)

    hold_dispersion = dispersion_model_path is not None and not config.update_dispersion
    params = [p for name, p in model.named_parameters() if not (hold_dispersion and "disp_model" in name)]
    print("Parameters being optimized:", [n for n, p in model.named_parameters() if not (hold_dispersion and "disp_model" in n)])
    return model, params, x_map


def load_model_from_state_dict(config, state_dict):
    """Model of `config`, with its state loaded from a checkpoint (if `state_dict` holds one) or warm-started
    from `config.init_from` (beta and dispersion parameters; the prior sd stays as configured)."""
    model, params, x_map = construct_model(config)
    if model_state_key in state_dict:
        print("Loading previously saved model...")
        load_state(model, state_dict[model_state_key]["model_state_dict"])
    elif config.init_from is not None:
        source = torch.load(Path(config.init_from), weights_only=False)[model_state_key]["model_state_dict"]
        own = model.state_dict()
        transfer = {k: v for k, v in source.items() if k != "beta_prior_sd" and k in own and own[k].shape == v.shape}
        model.load_state_dict(transfer, strict=False)
        print(f"Initialised {sorted(transfer)} from {config.init_from}")
    return model, params, x_map


def load_state(model, sd):
    """Load a (possibly upgraded) state dict; buffers that newer model versions add may be missing."""
    missing, unexpected = model.load_state_dict(upgrade_state_dict(sd), strict=False)
    allowed_missing = {"disp_model.kappa_bj", "disp_model.sigma_b", "disp_model.b_pi_prior_mean",
                       "disp_model.b_pi_prior_sd", "disp_model.sigma_bj_prior_sd", "disp_model.W"}
    bad = [k for k in missing if k not in allowed_missing]
    assert not bad and not unexpected, f"checkpoint does not match the model: missing {bad}, unexpected {list(unexpected)}"


# ---------------------------------------------------------------------------- fitting

def fit_posterior(model, optimizer, iterations, verbose_every=100):
    """Adam ascent on the log posterior; returns the loss history and the best state seen."""
    loss_history, best_state, best_loss = [], None, torch.inf
    for i in range(iterations):
        loss = -model.log_posterior(model.beta)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_history.append(loss.item())
        if loss.item() < best_loss:
            best_state, best_loss = copy.deepcopy(model.state_dict()), loss.item()
        if verbose_every and i % verbose_every == 0:
            print(f"Iter: {i}  loss: {loss.item():.4f}")
    return loss_history, best_state, best_loss


def fit_dispersion_model_to_composition(nbsr_model, pi_hat, iterations, lr, verbose_every=100):
    """Fit the dispersion model's parameters at a fixed composition pi_hat (N, J), e.g. DESeq2's fitted
    proportions, by maximising the NB log-likelihood of the counts plus the dispersion prior."""
    disp_model = nbsr_model.disp_model
    optimizer = torch.optim.Adam(disp_model.parameters(), lr=lr)
    for i in range(iterations):
        phi = torch.exp(disp_model.log_dispersion(pi_hat))
        loss = -(nbsr_model.log_likelihood(pi_hat, phi) + disp_model.log_prior())
        if loss.isnan():
            print("nan loss while fitting the dispersion model; stopping")
            break
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if verbose_every and i % verbose_every == 0:
            print(f"Iter: {i}  loss: {loss.item():.4f}")


def compute_negative_hessian_log_posterior(model, use_cuda_if_available=True):
    """Negative Hessian of the log posterior at the fitted beta, in closed form (utils.kron_hessian).
    Computed on the GPU when requested and available; always returned on the CPU in float64."""
    if use_cuda_if_available:
        model.to_device(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    start = time.perf_counter()
    with torch.no_grad():
        I = -model.log_posterior_hessian(model.beta.detach())
    print(f"Hessian computation time = {time.perf_counter() - start:.3f}s")
    model.to_device(torch.device("cpu"))
    return I.detach().cpu().double()


def fit_once(config):
    """Fit the model of `config` once and write beta, prior sd, composition, dispersion, dispersion-model
    parameters, checkpoint.pth, config.json and hessian.npy into config.output_path. Returns
    (loss_history, model)."""
    output_path = Path(config.output_path)
    create_directory(output_path)
    state_dict = {}
    model, params, x_map = load_model_from_state_dict(config, state_dict)
    state_dict["x_map"] = x_map

    optimizer = torch.optim.Adam(params, lr=config.lr)
    loss_history, best_state, best_loss = fit_posterior(model, optimizer, config.iterations)
    print("Training iterations completed.")
    model.load_state_dict(best_state)

    state_dict[model_state_key] = {"model_state_dict": model.state_dict(), "best_model_state_dict": best_state,
                                   "optimizer_state_dict": optimizer.state_dict(), "loss": loss_history, "best_loss": best_loss}

    pi, _ = model.predict(model.beta, model.X)
    if isinstance(model, NBSRTrended):
        phi = model.dispersion(pi)
        save_dispersion_model_outputs(model.disp_model, output_path)
    else:
        phi = model.softplus(model.phi)
    np.savetxt(output_path / "nbsr_beta.csv", model.beta.detach().numpy().transpose(), delimiter=",")
    np.savetxt(output_path / "nbsr_beta_sd.csv", model.beta_prior_sd.cpu().numpy(), delimiter=",")
    np.savetxt(output_path / "nbsr_pi.csv", pi.detach().numpy().transpose(), delimiter=",")
    np.savetxt(output_path / "nbsr_dispersion.csv", phi.detach().numpy().transpose(), delimiter=",")
    torch.save(state_dict, output_path / checkpoint_filename)
    config.dump_json(output_path / "config.json")

    print("Compute negative Hessian matrix.")
    I = compute_negative_hessian_log_posterior(model, config.use_cuda_if_available)
    np.save(output_path / hessian_filename, I.numpy())
    return loss_history, model


def save_dispersion_model_outputs(disp_model, output_path):
    """nbsr_dispersion_params.csv (b_0, b_pi, sigma_bj), nbsr_dispersion_bj.csv (per feature) and
    nbsr_dispersion_bw.csv (per external covariate, if any)."""
    output_path = Path(output_path)
    with torch.no_grad():
        pd.DataFrame({"b_0": disp_model.b_0.numpy(), "b_pi": disp_model.b_pi.numpy(),
                      "sigma_bj": disp_model.sigma_bj.numpy()}).to_csv(output_path / "nbsr_dispersion_params.csv", index=False)
        np.savetxt(output_path / "nbsr_dispersion_bj.csv", disp_model.b_j.numpy(), delimiter=",")
        if disp_model.covariate_count:
            np.savetxt(output_path / "nbsr_dispersion_bw.csv", disp_model.b_w.numpy(), delimiter=",")
        if disp_model.estimate_sd:
            np.savetxt(output_path / "nbsr_dispersion_sd.csv", disp_model.get_sd().numpy(), delimiter=",")
