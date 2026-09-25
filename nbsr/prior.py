"""Empirical prior elicitation for NBSR (stage 1 of the workflow).

`elicit_prior(config)` runs a diffuse-prior fit, optionally with the dispersion model first fitted to DESeq2's
composition (the small-sample route), sets the prior sd of each covariate by DESeq2-style quantile matching,
and writes prior.json. `nbsr fit` and the Stan pipeline both read that file.
"""
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as ss
import torch

from nbsr.fit import (build_dispersion_model, checkpoint_filename, fit_dispersion_model_to_composition, fit_once,
                      hessian_filename, load_data, save_dispersion_model_outputs)
from nbsr.inference import pooled_proportion, reference_set
from nbsr.nbsr_dispersion import NBSRTrended

prior_filename = "prior.json"


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


def deseq2_composition(counts_pd, coldata_pd, column_names, n_cpus=4):
    """Fitted proportions pi_hat (N, J) from PyDESeq2's fitted means under the design ~ column_names.
    Used to fit the dispersion model when the sample size is too small for the joint fit."""
    from pydeseq2.dds import DeseqDataSet
    from pydeseq2.default_inference import DefaultInference
    counts = counts_pd.transpose()                       # samples x features
    meta = coldata_pd.iloc[:, 1:] if coldata_pd.columns[0].lower().startswith("sample") else coldata_pd
    meta = meta.copy()
    meta.index = counts.index
    design = "~" + " + ".join(column_names) if column_names else "~1"
    dds = DeseqDataSet(counts=counts, metadata=meta, design=design, inference=DefaultInference(n_cpus=n_cpus), quiet=True)
    dds.deseq2()
    mu = np.asarray(dds.layers["_mu_hat"], dtype=np.float64)   # samples x features
    pi_hat = torch.tensor(mu / mu.sum(1, keepdims=True), dtype=torch.float64)
    return pi_hat


def elicit_prior(config):
    """Stage 1. Writes prior.json into config.output_path and returns its contents.

    With config.dispersion_from == "deseq2", the dispersion model is first fitted at DESeq2's composition and
    held fixed in the diffuse fit (and, by default, in the main fit that uses this prior).
    """
    output_path = Path(config.output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    stage = copy.deepcopy(config)
    stage.beta_prior = "fixed"
    stage.beta_prior_sd = [config.stage1_prior_sd]
    stage.output_path = output_path
    stage.iterations = config.stage1_iterations or max(config.iterations // 2, 1)
    stage.init_from = None
    stage.prior_file = None

    dispersion_fixed = False
    if config.dispersion_from == "deseq2":
        assert config.trended_dispersion, "--dispersion_from deseq2 needs the trended dispersion model"
        counts_pd, coldata_pd, Y, X, x_map, W = load_data(config)
        print("Fitting the dispersion model at DESeq2's composition.")
        pi_hat = deseq2_composition(counts_pd, coldata_pd, config.column_names)
        disp_model = build_dispersion_model(config, Y.shape[1], W)
        holder = NBSRTrended(X, Y, disp_model=disp_model, beta_prior_sd=config.stage1_prior_sd, pivot=config.pivot)
        fit_dispersion_model_to_composition(holder, pi_hat, config.eb_iterations, config.eb_lr)
        torch.save(disp_model, output_path / "disp_model.pth")
        save_dispersion_model_outputs(disp_model, output_path)
        stage.dispersion_model_file = str(output_path / "disp_model.pth")
        stage.update_dispersion = False
        dispersion_fixed = True

    print(f"Diffuse fit: prior sd {config.stage1_prior_sd} on every covariate, {stage.iterations} iterations.")
    _, model = fit_once(stage)
    beta = model.beta.detach().cpu().numpy()
    I = np.load(output_path / hessian_filename)
    sds = empirical_prior_sd(beta, I, model.covariate_count, config.prior_quantile)
    names = ["Intercept"] + list(model_x_map_names(config, model))
    print(f"Empirical prior sd per covariate (quantile {config.prior_quantile}): "
          + ", ".join(f"{n}={s:.4f}" for n, s in zip(names, sds)))

    prior = {
        "covariates": names,
        "beta_prior_sd": sds,
        "sigma_beta2": [s ** 2 for s in sds],           # Stan data block
        "prior_quantile": config.prior_quantile,
        "stage1_prior_sd": config.stage1_prior_sd,
        "stage1_iterations": stage.iterations,
        "dispersion_from": config.dispersion_from,
        "dispersion_fixed": dispersion_fixed,
        "dispersion_model_file": stage.dispersion_model_file if dispersion_fixed else None,
        "init_from": str(output_path / checkpoint_filename),
        "pivot": config.pivot,
    }
    # Laplace mode of beta as a covariate x feature CSV (readable by R for Stan inits), and the abundance
    # ranking of the features so that a reference set for the compositional shift can be formed downstream.
    beta_mat = beta.reshape(model.covariate_count, model.dim)
    pd.DataFrame(beta_mat, index=names).to_csv(output_path / "beta_mode.csv")
    pooled = pooled_proportion(model)
    order = torch.argsort(pooled, descending=True).tolist()
    prior["beta_mode_file"] = str(output_path / "beta_mode.csv")
    prior["features_by_abundance"] = order            # feature indices (0-based, column order of Y.csv), most abundant first
    prior["reference_top10"] = reference_set(pooled, 0.1, 30).tolist()
    if isinstance(model, NBSRTrended):
        dm = model.disp_model
        with torch.no_grad():
            prior["dispersion"] = {"b_0": float(dm.b_0), "b_pi": float(dm.b_pi), "sigma_bj": float(dm.sigma_bj),
                                   "b_w": dm.b_w.tolist(), "link": dm.link if isinstance(dm.link, str) else "callable",
                                   "feature_offsets": dm.feature_offsets}
    with open(output_path / prior_filename, "w") as f:
        json.dump(prior, f, indent=2)
    print(f"Wrote {output_path / prior_filename}")
    return prior


def model_x_map_names(config, model):
    """Names of the non-intercept design columns, in column order."""
    _, _, _, _, x_map, _ = load_data(config)
    return [name for name, _ in sorted(x_map.items(), key=lambda kv: kv[1])]


def load_prior(path):
    with open(path) as f:
        return json.load(f)


def apply_prior(config, prior):
    """Configure a main fit from a prior.json: fixed beta sds, warm start, and the fixed dispersion model when
    the prior was elicited from DESeq2."""
    config = copy.deepcopy(config)
    config.beta_prior = "fixed"
    config.beta_prior_sd = list(prior["beta_prior_sd"])
    config.init_from = prior.get("init_from")
    if prior.get("dispersion_fixed") and config.dispersion_model_file is None:
        config.dispersion_model_file = prior["dispersion_model_file"]
        config.update_dispersion = False
    return config
