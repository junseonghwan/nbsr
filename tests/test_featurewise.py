"""Tests for the feature-wise NB path: Dataset -> FeaturewiseNBStats.fit -> results.

Uses the first 40 features of data/test (20 samples, one two-level factor `trt`).
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from torch.func import grad, hessian, vmap

from nbsr.dataset import Dataset
from nbsr.featurewise_dispersion import LogDispersionTrendPrior, MeanPowerCovariateDispersion
from nbsr.distributions import log_negbinomial
from nbsr.featurewise_nb_model import FeaturewiseNegBinom
from nbsr.fnb_stats import FeaturewiseNBStats

DATA = Path(__file__).resolve().parents[1] / "data" / "test"
N_FEATURES = 40
IN_DIMS = (0, 0, 0, None, None, None)  # vmap over (theta, y, mu_bar); X, W, sf shared


def _load(disp_formula=None):
    # Y.csv is features x samples; Dataset wants samples x features.
    counts = pd.read_csv(DATA / "Y.csv", index_col=0).T.iloc[:, :N_FEATURES]
    # na_filter=False: pandas would otherwise read the level "null" as NaN.
    metadata = pd.read_csv(DATA / "X.csv", index_col="sample", na_filter=False).loc[counts.index]
    metadata["trt"] = pd.Categorical(metadata["trt"], categories=["null", "alt"])
    return Dataset(counts, metadata, mean_formula="~trt", disp_formula=disp_formula, n_cpus=1)


@pytest.fixture(scope="module")
def dataset():
    return _load()


@pytest.fixture(scope="module")
def dataset_with_disp_covariate():
    return _load(disp_formula="~trt")


def _reference_terms(stats, theta):
    """Value, gradient and Hessian of the log posterior via autograd on the reference implementation."""
    Y, X, W, sf, mu_bar = stats._tensors()
    YT = Y.T.contiguous()
    log_post = stats._make_log_posterior_fn(stats._build_model(mu_bar[0]))
    return (vmap(log_post, in_dims=IN_DIMS)(theta, YT, mu_bar, X, W, sf),
            vmap(grad(log_post), in_dims=IN_DIMS)(theta, YT, mu_bar, X, W, sf),
            vmap(hessian(log_post), in_dims=IN_DIMS)(theta, YT, mu_bar, X, W, sf))


# ---------------------------------------------------------------- dataset and priors

def test_dataset_shapes(dataset):
    assert dataset.gene_count == N_FEATURES
    assert dataset.sample_count == 20
    assert dataset.covariate_names == ["Intercept", "trt[T.alt]"]
    assert dataset.dispersion_covariate_count == 0
    assert torch.all(dataset.size_factors > 0)


def test_beta_prior_is_flat_on_intercept(dataset):
    stats = FeaturewiseNBStats(dataset, b_prior_sd=0.5, beta_prior_sd=5.0)
    model = stats._build_model(dataset.mu_bar[0])
    sd = model.beta_prior_sd
    assert sd.shape == (dataset.mean_covariate_count,)
    assert sd[dataset.covariate_names.index("Intercept")] == FeaturewiseNBStats.INTERCEPT_PRIOR_SD
    assert sd[dataset.covariate_names.index("trt[T.alt]")] == 5.0
    assert torch.isfinite(model.log_prior())


def test_scalar_beta_prior_sd_is_broadcast():
    prior = LogDispersionTrendPrior(0.1, 1.0, 0.5)
    disp = MeanPowerCovariateDispersion(0, prior, torch.tensor(10.0))
    model = FeaturewiseNegBinom(3, disp, beta_prior_sd=2.0)
    assert torch.equal(model.beta_prior_sd, torch.full((3,), 2.0))
    with pytest.raises(AssertionError):
        FeaturewiseNegBinom(3, disp, beta_prior_sd=torch.ones(2))


def test_gamma_prior_sd_is_passed_through(dataset):
    stats = FeaturewiseNBStats(dataset, gamma_prior_sd=0.25)
    model = stats._build_model(dataset.mu_bar[0])
    assert model.dispersion_model.gamma_prior_sd.item() == 0.25


# ---------------------------------------------------------------- derivatives

@pytest.mark.parametrize("fixture", ["dataset", "dataset_with_disp_covariate"])
def test_closed_form_derivatives_match_autograd(fixture, request):
    ds = request.getfixturevalue(fixture)
    stats = FeaturewiseNBStats(ds, b_prior_sd=0.5, gamma_prior_sd=0.7)
    Y, X, W, sf, mu_bar = stats._tensors()
    torch.manual_seed(0)
    # Perturb the warm start so the check is not at a stationary point.
    theta = stats._initial_params(Y, X, sf, mu_bar) + 0.3 * torch.randn(ds.gene_count, stats.n_parameters, dtype=torch.float64)
    f, g, H = stats._log_posterior_terms(theta, Y.T.contiguous(), mu_bar, X, W, sf, stats._build_model(mu_bar[0]))
    f_ref, g_ref, H_ref = _reference_terms(stats, theta)
    torch.testing.assert_close(f, f_ref, rtol=1e-10, atol=1e-8)
    torch.testing.assert_close(g, g_ref, rtol=1e-8, atol=1e-6)
    torch.testing.assert_close(H, H_ref, rtol=1e-8, atol=1e-6)


def test_score_outer_products_match_autograd(dataset_with_disp_covariate):
    ds = dataset_with_disp_covariate
    stats = FeaturewiseNBStats(ds, b_prior_sd=0.5)
    Y, X, W, sf, mu_bar = stats._tensors()
    YT = Y.T.contiguous()
    template = stats._build_model(mu_bar[0])
    theta = stats._initial_params(Y, X, sf, mu_bar)
    S = stats._score_outer_products(theta, YT, mu_bar, X, W, sf)

    def per_sample_loglik(th, y_i, x_i, w_i, sf_i):
        params = template.unpack(th)
        eta = x_i @ params["beta"] + torch.log(sf_i)
        log_phi = params["dispersion_model.a"] + params["dispersion_model.b"] * eta + w_i @ params["dispersion_model.gamma"]
        return log_negbinomial(y_i, torch.exp(eta), torch.exp(log_phi)).squeeze()

    scores = vmap(vmap(grad(per_sample_loglik), in_dims=(None, 0, 0, 0, 0)), in_dims=(0, 0, None, None, None))(theta, YT, X, W, sf)
    torch.testing.assert_close(S, torch.einsum("gnp,gnq->gpq", scores, scores), rtol=1e-8, atol=1e-6)


# ---------------------------------------------------------------- fitting and inference

@pytest.mark.parametrize("method", ["newton", "lbfgs"])
def test_fit_and_results(dataset, method):
    stats = FeaturewiseNBStats(dataset, b_prior_sd=0.5)
    stats.fit(method=method, **({"n_cpus": 2} if method == "lbfgs" else {}))

    n_params = dataset.mean_covariate_count + 2  # beta, a, b
    assert stats.results_["params"].shape == (N_FEATURES, n_params)
    assert stats.results_["hess_L"].shape == (N_FEATURES, n_params, n_params)
    converged = stats.results_["converged"].numpy()
    assert converged.all(), f"only {converged.sum()}/{N_FEATURES} genes converged"

    res = stats.results("trt", reference_level="null", test_level="alt")
    assert list(res.index) == list(dataset.var_names)
    ok = res["converged"].to_numpy()
    for col in ["estimate", "se", "z", "pval", "padj"]:
        assert np.all(np.isfinite(res.loc[ok, col])), f"non-finite {col} for converged genes"
    assert np.all(res.loc[ok, "se"] > 0)
    assert np.all((res.loc[ok, "pval"] >= 0) & (res.loc[ok, "pval"] <= 1))
    assert np.all((res.loc[ok, "padj"] >= 0) & (res.loc[ok, "padj"] <= 1))
    # The contrast alt vs null is the trt coefficient itself.
    np.testing.assert_allclose(res.loc[ok, "estimate"], stats.results_["params"][:, 1].numpy()[ok], rtol=1e-6)


def test_newton_matches_lbfgs(dataset):
    newton = FeaturewiseNBStats(dataset, b_prior_sd=0.5)
    newton.fit(method="newton")
    lbfgs = FeaturewiseNBStats(dataset, b_prior_sd=0.5)
    lbfgs.fit(method="lbfgs", n_cpus=2)
    ok = (newton.results_["converged"] & lbfgs.results_["converged"]).numpy()
    assert ok.sum() >= N_FEATURES - 1
    # L-BFGS stops on relative loss change and leaves gradient norms ~1e-3-1e-2; Newton reaches ~1e-7,
    # so agreement is limited by the L-BFGS solution.
    np.testing.assert_allclose(newton.results_["params"][ok].numpy(), lbfgs.results_["params"][ok].numpy(), rtol=1e-3, atol=5e-3)
    assert (newton.results_["grad_norm"][ok] < 1e-4).all()
    r_n = newton.results("trt", "null", "alt")
    r_l = lbfgs.results("trt", "null", "alt")
    np.testing.assert_allclose(r_n.loc[ok, "se"], r_l.loc[ok, "se"], rtol=1e-2)
    np.testing.assert_allclose(r_n.loc[ok, "z"], r_l.loc[ok, "z"], rtol=1e-2, atol=5e-3)


def test_analytic_and_autograd_newton_agree(dataset_with_disp_covariate):
    a = FeaturewiseNBStats(dataset_with_disp_covariate, b_prior_sd=0.5)
    a.fit_newton(derivatives="analytic")
    b = FeaturewiseNBStats(dataset_with_disp_covariate, b_prior_sd=0.5)
    b.fit_newton(derivatives="autograd")
    ok = (a.results_["converged"] & b.results_["converged"]).numpy()
    assert ok.sum() >= N_FEATURES - 1
    np.testing.assert_allclose(a.results_["params"][ok].numpy(), b.results_["params"][ok].numpy(), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(a.results_["hess_L"][ok].numpy(), b.results_["hess_L"][ok].numpy(), rtol=1e-6, atol=1e-6)


def test_robust_results(dataset):
    stats = FeaturewiseNBStats(dataset, b_prior_sd=0.5)
    stats.fit()
    model_based = stats.results("trt", "null", "alt")
    robust = stats.results("trt", "null", "alt", robust=True)
    ok = robust["converged"].to_numpy()
    assert np.all(np.isfinite(robust.loc[ok, "se"])) and np.all(robust.loc[ok, "se"] > 0)
    # Same point estimates, different standard errors.
    np.testing.assert_allclose(robust.loc[ok, "estimate"], model_based.loc[ok, "estimate"])
    assert not np.allclose(robust.loc[ok, "se"], model_based.loc[ok, "se"])


def test_batched_reference_hessian_matches_per_gene(dataset_with_disp_covariate):
    """vmap(hessian(...)) is the reference the closed forms are tested against, so it must itself agree with
    unbatched per-gene hessians. (torch.func's batched second derivatives are wrong for some ops, e.g. logdet,
    so this guards the reference implementation.)"""
    ds = dataset_with_disp_covariate
    stats = FeaturewiseNBStats(ds, b_prior_sd=0.5)
    Y, X, W, sf, mu_bar = stats._tensors()
    YT = Y.T.contiguous()
    log_post = stats._make_log_posterior_fn(stats._build_model(mu_bar[0]))
    theta = stats._initial_params(Y, X, sf, mu_bar)
    H_batched = vmap(hessian(log_post), in_dims=IN_DIMS)(theta, YT, mu_bar, X, W, sf)
    for g in [0, 1, N_FEATURES - 1]:
        H_single = hessian(log_post)(theta[g], YT[g], mu_bar[g], X, W, sf)
        torch.testing.assert_close(H_batched[g], H_single, rtol=1e-10, atol=1e-8)
