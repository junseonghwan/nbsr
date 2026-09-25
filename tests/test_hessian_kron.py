"""Closed-form NBSR gradients/Hessians (Kronecker structure) and the Cholesky-based logRR standard errors."""
import numpy as np
import pytest
import torch

import nbsr.dispersion as dm
import nbsr.main as main
import nbsr.negbinomial_model as nbm
import nbsr.nbsr_dispersion as nbsrd
from tests import reference_hessians


def _data(d=3, N=12, J=6, seed=0):
    rng = np.random.default_rng(seed)
    phi = np.log1p(np.exp(rng.standard_normal(J)))
    beta = rng.standard_normal((d, J))
    X = rng.standard_normal((N, d))
    X[:, 0] = 1.0
    s = rng.poisson(5000, N)
    pi = np.exp(X @ beta)
    pi /= pi.sum(1, keepdims=True)
    Y = np.stack([rng.multinomial(s[i], pi[i]) for i in range(N)]).astype(float)
    W = np.column_stack([np.log(s), rng.standard_normal(N)])
    return torch.tensor(X), torch.tensor(Y), phi, torch.tensor(W)


def _base_model(pivot, phi_fixed=True):
    X, Y, phi, _ = _data()
    return nbm.NegativeBinomialRegressionModel(X, Y, beta_prior_sd=1.0, dispersion=phi if phi_fixed else None, pivot=pivot)


def _trended_model(pivot, with_W=True, link="logit", feature_offsets=True):
    X, Y, _, W = _data()
    disp = dm.DispersionModel(Y.shape[1], W=W if with_W else None, link=link, feature_offsets=feature_offsets)
    with torch.no_grad():  # move every dispersion parameter off its initial value so all terms are exercised
        torch.manual_seed(1)
        disp.b_0.fill_(-0.5)
        disp.b_pi.fill_(1.3)
        disp.z_bj.copy_(torch.randn(Y.shape[1], dtype=torch.float64))
        disp.kappa_bj.fill_(0.2)
        if with_W:
            disp.b_w.copy_(torch.tensor([0.3, -0.2], dtype=torch.float64))
    return nbsrd.NBSRTrended(X, Y, disp_model=disp, beta_prior_sd=1.0, pivot=pivot)


MODELS = {"base": lambda pivot: _base_model(pivot),
          "trended": lambda pivot: _trended_model(pivot, with_W=True),
          "trended_noW": lambda pivot: _trended_model(pivot, with_W=False),
          "trended_loglink": lambda pivot: _trended_model(pivot, link="log"),
          "trended_no_offsets": lambda pivot: _trended_model(pivot, feature_offsets=False),
          # a user-supplied f(pi): derivatives come from autograd inside the dispersion model
          "trended_custom_f": lambda pivot: _trended_model(pivot, link=lambda pi: torch.sqrt(pi) - torch.log1p(-pi)),
          "trended_callable_logit": lambda pivot: _trended_model(pivot, link=lambda pi: torch.log(pi) - torch.log1p(-pi))}


@pytest.mark.parametrize("pivot", [False, True])
def test_base_hessian_matches_loop_reference(pivot):
    model = _base_model(pivot)
    beta = model.beta.detach()
    pi = model.predict(beta, model.X)[0].detach().numpy()
    mu = model.Y.sum(1, keepdim=True).numpy() * pi
    phi = model.softplus(model.phi).detach().numpy()
    expected = reference_hessians.hessian_nbsr(model.X.numpy(), model.Y.numpy(), pi, mu, phi, pivot)
    actual = model.log_likelihood_hessian(beta).numpy()
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-8)


@pytest.mark.parametrize("name", list(MODELS))
@pytest.mark.parametrize("pivot", [False, True])
def test_likelihood_gradient_matches_autograd(name, pivot):
    model = MODELS[name](pivot)
    beta = model.beta.detach().clone().requires_grad_(True)
    expected = torch.autograd.grad(model.log_likelihood_beta(beta), beta)[0]
    actual = model.log_lik_gradient(beta.detach())
    # Tolerance relative to the gradient's scale: with pivot the reference feature can sit near pi ~ 1 and the
    # logit factor 1/(1-pi) amplifies roundoff in both computations.
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-9 * expected.abs().max())
    per_sample = model.log_lik_gradient_persample(beta.detach())
    assert per_sample.shape == (model.sample_count, model.covariate_count * model.dim)


@pytest.mark.parametrize("name", list(MODELS))
@pytest.mark.parametrize("pivot", [False, True])
def test_posterior_hessian_matches_autograd(name, pivot):
    model = MODELS[name](pivot)
    with torch.no_grad():  # distinct prior sd per covariate so the layout matters
        model.beta_prior_sd.copy_(torch.linspace(0.5, 2.0, model.covariate_count, dtype=torch.float64))
    beta = model.beta.detach().clone()
    expected = torch.autograd.functional.hessian(model.log_posterior, beta)
    actual = model.log_posterior_hessian(beta)
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-8 * expected.abs().max())


def test_dispersion_model_forward_and_prior():
    X, Y, _, W = _data()
    disp = dm.DispersionModel(Y.shape[1], W=W, sigma_b=[1.0, 0.5])
    pi = torch.full((X.shape[0], Y.shape[1]), 1.0 / Y.shape[1], dtype=torch.float64)
    with torch.no_grad():
        disp.b_w.copy_(torch.tensor([0.5, 0.0], dtype=torch.float64))
    log_phi = disp.forward(pi)
    # b_0 + b_j (= 0 at init) + b_pi logit(1/J) + 0.5 log s_i
    expected = disp.b_0 + disp.b_pi * (np.log(1 / Y.shape[1]) - np.log(1 - 1 / Y.shape[1])) + 0.5 * W[:, :1]
    torch.testing.assert_close(log_phi, expected.expand_as(log_phi))
    assert torch.isfinite(disp.log_prior())
    assert disp.b_j.shape == (Y.shape[1],) and disp.sigma_bj > 0
    with pytest.raises(AssertionError):
        dm.DispersionModel(Y.shape[1], W=W, sigma_b=[1.0, 1.0, 1.0])


@pytest.mark.parametrize("pivot", [False, True])
def test_logRR_standard_errors_match_inverse_formula(pivot):
    model = _base_model(pivot)
    I = -model.log_posterior_hessian(model.beta.detach())
    x_map = {"grp_b": 1, "z_c": 0}  # column index excluding the intercept; only "grp" is contrasted
    logRR, log2RR, se, cov = main.inference_logRR(model, "grp", "b", "a", x_map, I, return_cov=True)
    N, J = model.Y.shape
    assert logRR.shape == se.shape == (N, J) and cov.shape == (N, J, J)
    np.testing.assert_allclose(log2RR, logRR / np.log(2))
    Z0, Z1 = model.X.clone(), model.X.clone()
    Z0[:, 2] = 0; Z1[:, 2] = 1
    pi0 = model.predict(model.beta, Z0)[0].detach()
    pi1 = model.predict(model.beta, Z1)[0].detach()
    d, P = model.dim, model.covariate_count
    eye = torch.eye(J, d, dtype=torch.float64).unsqueeze(0).expand(N, J, d)
    grad = ((eye - pi1[:, :d].unsqueeze(1)).unsqueeze(3) * Z1.unsqueeze(1).unsqueeze(2)
            - (eye - pi0[:, :d].unsqueeze(1)).unsqueeze(3) * Z0.unsqueeze(1).unsqueeze(2))
    grad = grad.transpose(2, 3).reshape(N, J, d * P)
    S = torch.linalg.inv(0.5 * (I + I.T))
    cov_ref = grad @ S @ grad.transpose(1, 2)
    torch.testing.assert_close(cov, cov_ref, rtol=1e-8, atol=1e-8)
    np.testing.assert_allclose(se, np.sqrt(np.diagonal(cov_ref.numpy(), axis1=1, axis2=2)), rtol=1e-8)
    np.testing.assert_allclose(logRR, (torch.log(pi1) - torch.log(pi0)).numpy())


def test_cholesky_with_jitter_adds_ridge_only_when_needed():
    A = torch.tensor([[4.0, 1.0], [1.0, 3.0]], dtype=torch.float64)
    torch.testing.assert_close(main.cholesky_with_jitter(A), torch.linalg.cholesky(A))
    singular = torch.tensor([[1.0, 1.0], [1.0, 1.0]], dtype=torch.float64)
    assert torch.isfinite(main.cholesky_with_jitter(singular)).all()
    with pytest.raises(RuntimeError):
        main.cholesky_with_jitter(-A)


def test_callable_link_matches_builtin_logit():
    """The autograd path for a callable f must reproduce the closed-form logit derivatives."""
    X, Y, _, W = _data()
    builtin = dm.DispersionModel(Y.shape[1], link="logit")
    custom = dm.DispersionModel(Y.shape[1], link=lambda pi: torch.log(pi) - torch.log1p(-pi))
    pi = torch.rand(4, Y.shape[1], dtype=torch.float64) * 0.9 + 0.01
    for a, b in zip(builtin.link_derivatives(pi), custom.link_derivatives(pi)):
        torch.testing.assert_close(a, b)


def test_empirical_prior_sd_matches_quantile():
    """With equal precision the matched sd is the 95% quantile of |beta| over 1.96; |beta| > 10 is dropped."""
    rng = np.random.default_rng(0)
    P, dim = 2, 400
    b = np.stack([rng.normal(0, 2.0, dim), rng.normal(0, 0.3, dim)])
    b[1, :3] = 50.0                                   # non-converged coefficients must not widen the prior
    H = np.eye(P * dim) * 4.0                         # equal precision -> equal weights
    sds = main.empirical_prior_sd(b.reshape(-1), H, P, quantile=0.95)
    for d, expected in enumerate([np.quantile(np.abs(b[0]), 0.95), np.quantile(np.abs(b[1, 3:]), 0.95)]):
        assert abs(sds[d] - expected / 1.959964) < 0.05 * expected / 1.959964
