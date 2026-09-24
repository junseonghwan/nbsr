"""Vectorized (Kronecker) NBSR Hessians and the Cholesky-based logRR standard errors."""
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
    return torch.tensor(X), torch.tensor(Y), phi


def _base_model(pivot, phi_fixed=True):
    X, Y, phi = _data()
    return nbm.NegativeBinomialRegressionModel(X, Y, lam=2.0, shape=3.0, scale=2.0,
                                               dispersion=phi if phi_fixed else None, pivot=pivot)


def _trended_model(pivot):
    X, Y, _ = _data()
    return nbsrd.NBSRTrended(X, Y, disp_model=dm.DispersionModel(Y), lam=2.0, shape=3.0, scale=2.0, pivot=pivot)


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


@pytest.mark.parametrize("pivot", [False, True])
def test_trended_hessian_matches_loop_reference(pivot):
    from scipy.special import digamma, polygamma
    model = _trended_model(pivot)
    beta = model.beta.detach()
    pi = model.predict(beta, model.X)[0].detach()
    phi = torch.exp(model.disp_model.forward(pi)).detach()
    mu = model.Y.sum(1, keepdim=True) * pi
    r = (1.0 / phi).numpy()
    p = (mu / (mu + phi * mu ** 2)).numpy()
    Y = model.Y.numpy()
    aa = digamma(Y + r) - digamma(r) + np.log(p)
    cc = polygamma(1, Y + r) - polygamma(1, r)
    expected = reference_hessians.hessian_trended_nbsr(model.X.numpy(), Y, pi.numpy(), p, r, aa, cc,
                                                       float(model.disp_model.b1.detach()), pivot)
    actual = model.log_likelihood_hessian(beta).numpy()
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-8)


@pytest.mark.parametrize("make_model", [_base_model, _trended_model])
@pytest.mark.parametrize("pivot", [False, True])
def test_posterior_hessian_matches_autograd(make_model, pivot):
    model = make_model(pivot)
    with torch.no_grad():  # distinct prior sd per covariate so the layout matters
        model.psi.copy_(torch.linspace(-1.0, 2.0, model.covariate_count, dtype=torch.float64))
    beta = model.beta.detach().clone()
    expected = torch.autograd.functional.hessian(model.log_posterior, beta)
    actual = model.log_posterior_hessian(beta)
    torch.testing.assert_close(actual, expected, rtol=1e-8, atol=1e-6)


@pytest.mark.parametrize("pivot", [False, True])
def test_logRR_standard_errors_match_inverse_formula(pivot):
    model = _base_model(pivot)
    I = -model.log_posterior_hessian(model.beta.detach())
    x_map = {"grp_b": 1, "z_c": 0}  # column index excluding the intercept; only "grp" is contrasted
    logRR, log2RR, se, cov = main.inference_logRR(model, "grp", "b", "a", x_map, I, return_cov=True)
    N, J = model.Y.shape
    assert logRR.shape == se.shape == (N, J) and cov.shape == (N, J, J)
    np.testing.assert_allclose(log2RR, logRR / np.log(2))
    # Reference: Var = a' I^-1 a with the same contrast gradient, via an explicit inverse.
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
    L = main.cholesky_with_jitter(singular)
    assert torch.isfinite(L).all()
    with pytest.raises(RuntimeError):
        main.cholesky_with_jitter(-A)
