"""Inference from a fitted NBSR model: Hessian-based standard errors, per-sample log relative-abundance ratios,
the reference-free CLR contrast, the compositional shift over a reference set, and posterior sign
probabilities by the delta method or by draws from the Laplace approximation.
"""
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as ss
import torch

from nbsr.fit import checkpoint_filename, compute_negative_hessian_log_posterior, hessian_filename, load_model_from_state_dict
from nbsr.nbsr_config import NBSRConfig

covariance_filename = "covariance.pth"


# ---------------------------------------------------------------------------- linear algebra

def cholesky_with_jitter(I, max_jitter=1e-3):
    """Cholesky factor of a symmetric matrix that should be positive definite (a negative Hessian at a mode).
    Roundoff or an unconverged fit can make it indefinite; a small ridge is added in decades up to max_jitter
    before giving up."""
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


def contrast_se(L, R):
    """Standard errors of linear functionals R beta (R: (m, p)) under Cov(beta) = (L L')^-1: ||L^-1 r||."""
    V = torch.linalg.solve_triangular(L, R.T, upper=False)   # (p, m)
    return V.pow(2).sum(0).sqrt(), V


def laplace_draws_of_contrasts(L, R, estimate, n_draws, seed=0):
    """Draws of R beta with beta ~ N(beta_hat, (L L')^-1): estimate + R L^-T z. Returns (n_draws, m)."""
    gen = torch.Generator().manual_seed(seed)
    Z = torch.randn(L.shape[0], n_draws, generator=gen, dtype=L.dtype)
    B = torch.linalg.solve_triangular(L.T, Z, upper=True)     # (p, S) = L^-T z
    return estimate.unsqueeze(0) + (R @ B).T


# ---------------------------------------------------------------------------- design contrasts

def level_columns(x_map, var, w1, w0):
    """Design column indices (including the intercept offset) of levels w1 and w0 of factor `var`; None for
    the reference level. Raises if neither level is a column."""
    c1 = x_map.get(f"{var}_{w1}")
    c0 = x_map.get(f"{var}_{w0}")
    assert c1 is not None or c0 is not None, f"neither {var}_{w1} nor {var}_{w0} among the covariates {list(x_map)}"
    return (c1 + 1 if c1 is not None else None), (c0 + 1 if c0 is not None else None)


def clr_contrast(model, x_map, var, w1, w0):
    """Reference-free CLR contrast of level w1 against w0 for every feature, Delta_j = d_j - mean_k d_k with
    d_j the coefficient difference of feature j (0 for the pivot feature), and the (J, P*dim) matrix R of its
    gradient w.r.t. the flat covariate-major beta, so that Delta = R beta."""
    c1, c0 = level_columns(x_map, var, w1, w0)
    J, dim, P = model.rna_count, model.dim, model.covariate_count
    R = torch.zeros(J, P * dim, dtype=torch.float64)
    for col, sign in [(c1, 1.0), (c0, -1.0)]:
        if col is None:
            continue
        block = sign * (torch.eye(J, dim, dtype=torch.float64) - 1.0 / J)   # (J, dim): delta_jk - 1/K
        R[:, col * dim:(col + 1) * dim] = block
    delta = R @ model.beta.detach().cpu().double()
    return delta, R


def pooled_proportion(model):
    """Mean fitted proportion of each feature over samples: ranks features by abundance without reference
    to condition."""
    with torch.no_grad():
        pi, _ = model.predict(model.beta, model.X)
    return pi.mean(0).cpu().double()


def reference_set(pooled, top_frac=None, min_features=30):
    """Indices of the features over which the compositional shift is estimated: all features (top_frac None)
    or the top fraction by pooled proportion, with a floor of min_features."""
    J = pooled.numel()
    if top_frac is None:
        return torch.arange(J)
    n = int(max(min(J, min_features), round(top_frac * J)))
    return torch.argsort(pooled, descending=True)[:n]


def kde_mode(values, grid_points=512):
    """Mode of a Gaussian kernel density estimate (Scott's bandwidth) for each row of `values` (S, R), refined
    off the grid by a parabola through the three highest grid points. Returns (S,)."""
    values = torch.as_tensor(values, dtype=torch.float64)
    if values.ndim == 1:
        values = values.unsqueeze(0)
    S, R = values.shape
    sd = values.std(1, unbiased=True).clamp(min=1e-12)
    h = sd * R ** (-1.0 / 5.0)                                     # Scott's rule
    lo, hi = values.min(1).values - 3 * h, values.max(1).values + 3 * h
    grid = lo.unsqueeze(1) + (hi - lo).unsqueeze(1) * torch.linspace(0, 1, grid_points, dtype=torch.float64)
    dens = torch.exp(-0.5 * ((grid.unsqueeze(2) - values.unsqueeze(1)) / h.reshape(-1, 1, 1)) ** 2).sum(2)   # (S, G)
    k = dens.argmax(1).clamp(1, grid_points - 2)
    idx = torch.arange(S)
    y0, y1, y2 = dens[idx, k - 1], dens[idx, k], dens[idx, k + 1]
    denom = (y0 - 2 * y1 + y2)
    offset = torch.where(denom.abs() > 0, 0.5 * (y0 - y2) / denom, torch.zeros_like(denom))
    step = (hi - lo) / (grid_points - 1)
    return grid[idx, k] + offset.clamp(-1, 1) * step


# ---------------------------------------------------------------------------- per-sample log relative-abundance ratio

def inference_logRR(model, var, w1, w0, x_map, I, return_cov=True):
    """Log relative-abundance ratio of level w1 over w0 of `var` for every sample and feature, with
    delta-method standard errors from the negative Hessian I (covariate-major, index d*dim + k).

    Each logRR_nj is a function g_nj(beta); its gradient a_nj (length dim*P) gives Var(logRR_nj) = a_nj' I^-1 a_nj.
    With I = L L' this is ||L^-1 a_nj||^2, so all N*J variances come from one triangular solve. return_cov
    also forms the (N, J, J) covariance of the logRR vector within each sample.
    """
    c1, c0 = level_columns(x_map, var, w1, w0)
    Z0, Z1 = model.X.clone(), model.X.clone()
    for colname, col_idx in x_map.items():
        if var in colname:
            Z0[:, col_idx + 1] = 0
            Z1[:, col_idx + 1] = 0
    if c0 is not None:
        Z0[:, c0] = 1
    if c1 is not None:
        Z1[:, c1] = 1

    pi0, _ = model.predict(model.beta, Z0)
    pi1, _ = model.predict(model.beta, Z1)
    logRRi = torch.log(pi1) - torch.log(pi0)
    N, J = pi0.shape
    P, d = model.covariate_count, model.dim
    identity = torch.eye(J, d, device=pi0.device, dtype=pi0.dtype).unsqueeze(0).expand(N, J, d)
    ipi0 = identity - pi0[:, :d].unsqueeze(1)
    ipi1 = identity - pi1[:, :d].unsqueeze(1)
    ret = ipi1.unsqueeze(3) * Z1.unsqueeze(1).unsqueeze(2) - ipi0.unsqueeze(3) * Z0.unsqueeze(1).unsqueeze(2)
    ret = ret.transpose(2, 3).reshape(N, J, d * P)
    L = cholesky_with_jitter(I.to(device=ret.device, dtype=ret.dtype))
    se, V = contrast_se(L, ret.reshape(N * J, d * P))
    cov_mat = None
    if return_cov:
        Vn = V.T.reshape(N, J, d * P)
        cov_mat = torch.bmm(Vn, Vn.transpose(1, 2))
    return (logRRi.detach().cpu().numpy(), (logRRi / np.log(2)).detach().cpu().numpy(), se.reshape(N, J).detach().cpu().numpy(),
            cov_mat.detach().cpu() if cov_mat is not None else None)


# ---------------------------------------------------------------------------- results

def _safe_name(x):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(x))


def load_fit(results_path, recompute_hessian=False):
    """Model, x_map and negative Hessian of a finished fit directory."""
    results_path = Path(results_path)
    config = NBSRConfig.load_json(results_path / "config.json")
    state_dict = torch.load(results_path / checkpoint_filename, weights_only=False)
    x_map = state_dict["x_map"]
    model, _, x_map2 = load_model_from_state_dict(config, state_dict)
    assert all(k in x_map2 and x_map2[k] == v for k, v in x_map.items()), "design columns of the checkpoint and the data differ"
    if not recompute_hessian and os.path.exists(results_path / hessian_filename):
        I = torch.from_numpy(np.load(results_path / hessian_filename)).double()
    else:
        I = compute_negative_hessian_log_posterior(model, config.use_cuda_if_available)
        np.save(results_path / hessian_filename, I.numpy())
    return config, model, x_map, I


def feature_contrasts(model, x_map, I, var, w1, w0, ref_top_frac=None, ref_min=30, laplace_draws=0, seed=0):
    """Feature-level table for the contrast w1 vs w0 of `var`:

    delta_clr, se_clr : reference-free CLR contrast and its delta-method standard error;
    shift             : mode of delta_clr over the reference set (compositional shift, sparsity assumption);
    delta_adj         : delta_clr - shift, the absolute-abundance effect, with the same se (delta method, shift
                        treated as fixed), Wald z, p-value, BH-adjusted p-value and ppos = Phi(z);
    with laplace_draws > 0: posterior summaries of delta_adj from draws of beta ~ N(beta_hat, -H^-1) in which
    the shift is recomputed per draw (post_mean, post_sd, q025, q975, ppos_draws, lfsr_draws).
    Returns (DataFrame indexed by feature position, dict of scalars).
    """
    delta, R = clr_contrast(model, x_map, var, w1, w0)
    L = cholesky_with_jitter(I.double())
    se, _ = contrast_se(L, R)
    pooled = pooled_proportion(model)
    ref = reference_set(pooled, ref_top_frac, ref_min)
    shift = kde_mode(delta[ref].unsqueeze(0))[0]
    adj = delta - shift
    z = adj / se
    pval = 2 * ss.norm.cdf(-np.abs(z.numpy()))
    table = pd.DataFrame({
        "delta_clr": delta.numpy(), "se_clr": se.numpy(),
        "delta_adj": adj.numpy(), "z": z.numpy(), "pvalue": pval,
        "padj": ss.false_discovery_control(pval, method="bh"),
        "ppos": ss.norm.cdf(z.numpy()),
        "pooled_proportion": pooled.numpy(),
        "reference": np.isin(np.arange(delta.numel()), ref.numpy()),
    })
    summary = {"shift": float(shift), "reference_size": int(ref.numel()), "ref_top_frac": ref_top_frac, "ref_min": ref_min}
    if laplace_draws > 0:
        draws = laplace_draws_of_contrasts(L, R, delta, laplace_draws, seed)      # (S, J) draws of delta_clr
        shifts = kde_mode(draws[:, ref])                                             # (S,) shift per draw
        adj_draws = draws - shifts.unsqueeze(1)
        table["post_mean"] = adj_draws.mean(0).numpy()
        table["post_sd"] = adj_draws.std(0).numpy()
        q = torch.quantile(adj_draws, torch.tensor([0.025, 0.975], dtype=torch.float64), dim=0)
        table["q025"], table["q975"] = q[0].numpy(), q[1].numpy()
        ppos = (adj_draws > 0).double().mean(0).numpy()
        table["ppos_draws"] = ppos
        table["lfsr_draws"] = np.minimum(ppos, 1 - ppos)
        summary.update({"laplace_draws": laplace_draws, "shift_draw_mean": float(shifts.mean()), "shift_draw_sd": float(shifts.std())})
    return table, summary


def generate_results(results_path, var, w1, w0, absolute_fc=False, recompute_hessian=False, save_cov=True,
                     ref_top_frac=None, ref_min=30, laplace_draws=0, seed=0):
    """Write the results of contrast w1 vs w0 of `var` into <results_path>/<var>__<w1>_vs_<w0>/:
    nbsr_results.h5 and nbsr_results.csv (per-sample log relative-abundance ratios and Wald tests, as before),
    contrast.csv and contrast_summary.json (feature-level CLR contrast, shift, absolute effect, ppos).
    absolute_fc subtracts the shift from the per-sample logRR as well."""
    results_path = Path(results_path)
    comparison_path = results_path / f"{_safe_name(var)}__{_safe_name(w1)}_vs_{_safe_name(w0)}"
    comparison_path.mkdir(parents=True, exist_ok=True)
    config, model, x_map, I = load_fit(results_path, recompute_hessian)
    counts_pd = pd.read_csv(config.counts_path, index_col=0)
    samples, features = counts_pd.columns, counts_pd.index

    # Feature-level contrast.
    table, summary = feature_contrasts(model, x_map, I, var, w1, w0, ref_top_frac, ref_min, laplace_draws, seed)
    table.insert(0, "feature", features)
    table.to_csv(comparison_path / "contrast.csv", index=False)
    with open(comparison_path / "contrast_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Compositional shift {summary['shift']:+.4f} over {summary['reference_size']} reference features"
          + (f" (draw sd {summary['shift_draw_sd']:.4f})" if laplace_draws else ""))

    # Per-sample log relative-abundance ratios (kept for compatibility with earlier outputs).
    logRR, log2RR, logRR_std, cov_mat = inference_logRR(model, var, w1, w0, x_map, I, return_cov=save_cov)
    log_bias = summary["shift"] if absolute_fc else 0.0
    logRR = logRR - log_bias
    stat = logRR / logRR_std
    pvalue = 2 * ss.norm.cdf(-np.abs(stat))
    padj = np.array([ss.false_discovery_control(p, method="bh") for p in pvalue])
    if cov_mat is not None:
        torch.save(cov_mat, comparison_path / covariance_filename)
    with pd.HDFStore(comparison_path / "nbsr_results.h5", mode="w") as store:
        store["logRR"] = pd.DataFrame(logRR.T, index=features, columns=samples)
        store["se"] = pd.DataFrame(logRR_std.T, index=features, columns=samples)
        store["stat"] = pd.DataFrame(stat.T, index=features, columns=samples)
        store["pvalue"] = pd.DataFrame(pvalue.T, index=features, columns=samples)
        store["padj"] = pd.DataFrame(padj.T, index=features, columns=samples)
    res = None
    if np.allclose(log2RR, log2RR[0, :], atol=1e-8):   # the contrast does not vary by sample
        res = pd.DataFrame({"feature": features, "log2FC": logRR[0, :] / np.log(2), "pvalue": pvalue[0, :], "padj": padj[0, :]})
        res.to_csv(comparison_path / "nbsr_results.csv", index=False)
    return table, res
