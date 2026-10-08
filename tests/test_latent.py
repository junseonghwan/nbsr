"""NBSR with latent factors: eta_ij = x_i' beta_j + z_i' gamma_j, z_i ~ N(0, I_K), gamma_j ~ N(0, sd)."""
import numpy as np
import pandas as pd
import pytest
import torch
from torch.func import functional_call

import nbsr.dispersion as dm
import nbsr.main as main
import nbsr.negbinomial_model as nbm
import nbsr.nbsr_dispersion as nbsrd
from nbsr.distributions import log_normal
from nbsr.nbsr_config import NBSRConfig


def simulate(N=24, J=8, K=2, seed=0, latent_sd=1.0, effect=None, s=5000):
    """Two-group design (intercept + treatment), K latent factors; returns tensors and the truth."""
    rng = np.random.default_rng(seed)
    X = np.column_stack([np.ones(N), np.repeat([0.0, 1.0], N // 2)])
    beta = np.zeros((2, J))
    beta[0] = rng.normal(0, 1.0, J)
    beta[1] = rng.normal(0, 0.5, J) if effect is None else effect
    Z = rng.standard_normal((N, K))
    Gamma = rng.normal(0, latent_sd, (K, J))
    eta = X @ beta + Z @ Gamma
    pi = np.exp(eta - eta.max(1, keepdims=True))
    pi /= pi.sum(1, keepdims=True)
    sizes = rng.poisson(s, N)
    Y = np.stack([rng.multinomial(sizes[i], pi[i]) for i in range(N)]).astype(float)
    return (torch.tensor(X), torch.tensor(Y), dict(beta=beta, Z=Z, Gamma=Gamma, eta=eta, pi=pi))


def make_model(pivot=False, trended=False, K=2, seed=1):
    torch.manual_seed(seed)
    X, Y, truth = simulate(K=K)
    if trended:
        disp = dm.DispersionModel(Y.shape[1], W=torch.log(Y.sum(1, keepdim=True)))
        with torch.no_grad():
            disp.b_0.fill_(2.0); disp.b_pi.fill_(1.2)
            disp.z_bj.copy_(0.3 * torch.randn(Y.shape[1], dtype=torch.float64))
        model = nbsrd.NBSRTrended(X, Y, disp, beta_prior_sd=[1.0, 0.5], pivot=pivot, latent_dim=K, latent_prior_sd=0.7)
    else:
        phi = np.full(Y.shape[1], 0.05)
        model = nbm.NegativeBinomialRegressionModel(X, Y, beta_prior_sd=[1.0, 0.5], dispersion=phi, pivot=pivot,
                                                    latent_dim=K, latent_prior_sd=0.7)
    with torch.no_grad():  # move off the small initial values so every term is exercised
        model.Z.copy_(torch.randn_like(model.Z))
        model.Gamma.copy_(0.5 * torch.randn_like(model.Gamma))
    return model, truth


def theta_posterior(model):
    """log posterior as a function of the flat (beta, Gamma) vector, Z fixed (for autograd references)."""
    P, K, d = model.covariate_count, model.latent_dim, model.dim

    def f(theta):
        beta, Gamma = theta[:P * d], theta[P * d:].reshape(K, d)
        return functional_call(model, {"Gamma": Gamma}, (beta,))
    return f


# ----------------------------------------------------------------------------- construction
def test_latent_dim_zero_is_the_plain_model():
    X, Y, _ = simulate()
    model = nbm.NegativeBinomialRegressionModel(X, Y, beta_prior_sd=1.0, dispersion=np.ones(Y.shape[1]), latent_dim=0)
    assert model.Z is None and model.Gamma is None and model.loadings() is None
    assert set(model.state_dict()) == {"X", "Y", "s", "beta_prior_sd", "beta"}
    assert model.log_latent_prior() == 0.0
    assert model.theta_count == model.beta.numel() and model.theta() is model.beta
    torch.testing.assert_close(model.augmented_design(model.X), model.X)
    assert model.log_posterior_hessian(model.beta.detach()).shape == (model.beta.numel(),) * 2


@pytest.mark.parametrize("pivot", [False, True])
def test_predict_adds_latent_term(pivot):
    model, _ = make_model(pivot)
    beta = model.beta.detach()
    pi, eta = model.predict(beta, model.X)
    expected = model.X @ beta.reshape(model.covariate_count, model.dim) + model.Z @ model.Gamma
    if pivot:
        expected = torch.column_stack([expected, torch.zeros(model.sample_count, dtype=torch.float64)])
    torch.testing.assert_close(eta, expected)
    torch.testing.assert_close(pi, torch.softmax(expected, 1))
    # Z = 0 drops the latent term; an explicit Z is used as given.
    _, eta0 = model.predict(beta, model.X, model.latent_scores(zero=True))
    torch.testing.assert_close(eta0[:, :model.dim], model.X @ beta.reshape(model.covariate_count, model.dim))
    Zc = torch.randn_like(model.Z)
    _, etac = model.predict(beta, model.X, Zc)
    torch.testing.assert_close(etac[:, :model.dim] - eta0[:, :model.dim], Zc @ model.Gamma)
    with pytest.raises(AssertionError):
        model.predict(beta, model.X[:3])  # fitted Z has N rows


@pytest.mark.parametrize("pivot", [False, True])
def test_loadings_and_prior(pivot):
    model, _ = make_model(pivot)
    L = model.loadings()
    assert L.shape == (model.rna_count, model.latent_dim)
    torch.testing.assert_close(L[:model.dim], model.Gamma.T)
    if pivot:
        assert torch.all(L[-1] == 0)
    one = torch.ones((), dtype=torch.float64)
    expected = (log_normal(model.Z, 0 * one, one).sum()
                + log_normal(model.Gamma, 0 * one, torch.tensor(0.7, dtype=torch.float64)).sum())
    torch.testing.assert_close(model.log_latent_prior(), expected)


# ----------------------------------------------------------------------------- derivatives
@pytest.mark.parametrize("trended", [False, True])
@pytest.mark.parametrize("pivot", [False, True])
def test_gradient_matches_autograd(pivot, trended):
    model, _ = make_model(pivot, trended)
    beta = model.beta.detach().clone().requires_grad_(True)
    g_beta, g_Gamma = torch.autograd.grad(model.log_likelihood_beta(beta), [beta, model.Gamma])
    expected = torch.cat([g_beta, g_Gamma.reshape(-1)])
    actual = model.log_lik_gradient(beta.detach())
    assert actual.shape == (model.theta_count,)
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-9 * expected.abs().max())
    per_sample = model.log_lik_gradient_persample(beta.detach())
    assert per_sample.shape == (model.sample_count, model.theta_count)
    # Posterior gradient: the prior part covers beta and Gamma.
    theta = model.theta().detach().clone().requires_grad_(True)
    expected_post = torch.autograd.grad(theta_posterior(model)(theta), theta)[0]
    torch.testing.assert_close(model.log_posterior_gradient(beta.detach()), expected_post, rtol=1e-6, atol=1e-9 * expected_post.abs().max())


@pytest.mark.parametrize("trended", [False, True])
@pytest.mark.parametrize("pivot", [False, True])
def test_posterior_hessian_matches_autograd(pivot, trended):
    model, _ = make_model(pivot, trended)
    theta = model.theta().detach().clone()
    expected = torch.autograd.functional.hessian(theta_posterior(model), theta)
    actual = model.log_posterior_hessian(model.beta.detach())
    assert actual.shape == (model.theta_count,) * 2
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-8 * expected.abs().max())


# ----------------------------------------------------------------------------- inference
@pytest.mark.parametrize("pivot", [False, True])
@pytest.mark.parametrize("latent_at_zero", [True, False])
def test_logRR_and_standard_errors(pivot, latent_at_zero):
    model, _ = make_model(pivot)
    I = -model.log_posterior_hessian(model.beta.detach())
    x_map = {"trt_b": 0}
    logRR, log2RR, se, cov = main.inference_logRR(model, "trt", "b", "a", x_map, I, return_cov=True,
                                                  latent_at_zero=latent_at_zero)
    N, J = model.Y.shape
    assert logRR.shape == se.shape == (N, J) and cov.shape == (N, J, J)
    assert np.isfinite(se).all() and (se > 0).all()
    np.testing.assert_allclose(log2RR, logRR / np.log(2))
    if latent_at_zero:
        # With a single covariate the contrast at Z = 0 is the same for every sample.
        np.testing.assert_allclose(logRR, np.broadcast_to(logRR[:1], logRR.shape), atol=1e-10)
        np.testing.assert_allclose(se, np.broadcast_to(se[:1], se.shape), atol=1e-10)
    else:
        assert not np.allclose(logRR, np.broadcast_to(logRR[:1], logRR.shape), atol=1e-6)

    # Reference: delta method with the autograd Jacobian of logRR_n with respect to (beta, Gamma).
    P, K, d = model.covariate_count, model.latent_dim, model.dim
    Z0, Z1 = model.X.clone(), model.X.clone()
    Z0[:, 1] = 0; Z1[:, 1] = 1
    latent = model.latent_scores(zero=latent_at_zero)

    theta = model.theta().detach().clone()
    Jac = torch.autograd.functional.jacobian(lambda t: _logRR_with_theta(model, t, Z0, Z1, latent), theta)  # (N*J, theta)
    S = torch.linalg.inv(0.5 * (I + I.T))
    cov_ref = (Jac @ S @ Jac.T).reshape(N, J, N, J)
    cov_ref = torch.stack([cov_ref[n, :, n, :] for n in range(N)])
    torch.testing.assert_close(cov, cov_ref, rtol=1e-7, atol=1e-9)
    np.testing.assert_allclose(se, np.sqrt(np.diagonal(cov_ref.numpy(), axis1=1, axis2=2)), rtol=1e-7)


def _logRR_with_theta(model, theta, Z0, Z1, latent):
    """logRR (flattened N*J) as a function of the flat (beta, Gamma); Gamma is swapped in with functional_call."""
    P, K, d = model.covariate_count, model.latent_dim, model.dim
    beta, Gamma = theta[:P * d], theta[P * d:].reshape(K, d)
    wrapper = _Predict(model)
    pi0 = functional_call(wrapper, {"model.Gamma": Gamma}, (beta, Z0, latent))
    pi1 = functional_call(wrapper, {"model.Gamma": Gamma}, (beta, Z1, latent))
    return (torch.log(pi1) - torch.log(pi0)).reshape(-1)


class _Predict(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, beta, X, Z):
        return self.model.predict(beta, X, Z)[0]


def test_empirical_prior_sd_uses_the_beta_block():
    rng = np.random.default_rng(0)
    P, dim, K = 2, 50, 3
    b = np.stack([rng.normal(0, 2.0, dim), rng.normal(0, 0.3, dim)])
    H = np.eye((P + K) * dim) * 4.0
    H[P * dim:, P * dim:] *= 1000.0  # the Gamma block must not matter
    sds = main.empirical_prior_sd(b.reshape(-1), H, P, quantile=0.95)
    assert len(sds) == P
    for d_, expected in enumerate([np.quantile(np.abs(b[0]), 0.95), np.quantile(np.abs(b[1]), 0.95)]):
        assert abs(sds[d_] - expected / 1.959964) < 0.1 * expected / 1.959964


# ----------------------------------------------------------------------------- fitting
def double_center(M):
    return M - M.mean(0, keepdims=True) - M.mean(1, keepdims=True) + M.mean()


def test_fit_recovers_latent_structure_and_beats_no_factor_model():
    """Adam on the log posterior recovers Z Gamma up to the softmax/intercept invariances, and improves the
    likelihood over the same model without factors."""
    torch.manual_seed(3)
    X, Y, truth = simulate(N=30, J=10, K=2, seed=5, latent_sd=1.0)
    phi = np.full(Y.shape[1], 0.02)
    fits = {}
    for K in (0, 2):
        model = nbm.NegativeBinomialRegressionModel(X, Y, beta_prior_sd=[3.0, 3.0], dispersion=phi, latent_dim=K)
        opt = torch.optim.Adam(model.parameters(), lr=0.05)
        for _ in range(1500):
            loss = -model.log_posterior(model.beta)
            opt.zero_grad(); loss.backward(); opt.step()
        fits[K] = model
    ll0 = fits[0].log_likelihood(fits[0].beta).item()
    ll2 = fits[2].log_likelihood(fits[2].beta).item()
    assert ll2 > ll0 + 50, (ll0, ll2)
    # The fitted latent term, after removing per-sample and per-feature offsets (absorbed by the softmax and
    # the intercept), should match the truth; the scale is set by the N(0, 1) prior on Z, so compare by correlation.
    fitted = double_center((fits[2].Z @ fits[2].Gamma).detach().numpy())
    true = double_center(truth["Z"] @ truth["Gamma"])
    corr = np.corrcoef(fitted.ravel(), true.ravel())[0, 1]
    assert corr > 0.9, corr
    # Not asserted: that beta is closer to the truth with the factors. At this size it depends on how the
    # component of the true Z that is correlated with the treatment by chance is split between beta and Z.


def test_train_and_results_end_to_end(tmp_path):
    """run() and generate_results() with latent factors: outputs written, Hessian shape, csv table present."""
    X, Y, _ = simulate(N=16, J=6, K=1, seed=2)
    pd.DataFrame(Y.numpy().T, index=[f"f{j}" for j in range(Y.shape[1])],
                 columns=[f"s{i}" for i in range(Y.shape[0])]).to_csv(tmp_path / "Y.csv")
    pd.DataFrame({"trt": np.where(X[:, 1].numpy() > 0, "b", "a")}).to_csv(tmp_path / "X.csv", index=False)
    config = NBSRConfig(counts_path=tmp_path / "Y.csv", coldata_path=tmp_path / "X.csv", output_path=tmp_path,
                        column_names=["trt"], iterations=40, stage1_iterations=20, trended_dispersion=True,
                        latent_dim=2, use_cuda_if_available=False)
    _, model = main.run(config)
    assert model.latent_dim == 2
    scores = np.loadtxt(tmp_path / "nbsr_latent_scores.csv", delimiter=",")
    loadings = np.loadtxt(tmp_path / "nbsr_latent_loadings.csv", delimiter=",")
    assert scores.shape == (16, 2) and loadings.shape == (6, 2)
    H = np.load(tmp_path / "hessian.npy")
    assert H.shape == ((2 + 2) * 6,) * 2
    assert NBSRConfig.load_json(tmp_path / "config.json").latent_dim == 2
    res = main.generate_results(tmp_path, "trt", "b", "a", absolute_fc=False)
    assert res is not None and list(res.columns) == ["feature", "log2FC", "pvalue", "padj"] and len(res) == 6
    assert (tmp_path / "trt__b_vs_a" / "nbsr_results.h5").exists()
    # Fitted scores: sample-specific logRR, so no csv table.
    assert main.generate_results(tmp_path, "trt", "b", "a", absolute_fc=False, latent_at_zero=False) is None
