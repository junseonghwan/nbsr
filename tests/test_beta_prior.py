import numpy as np
import torch

import nbsr.negbinomial_model as nbm
from nbsr.distributions import log_normal, softplus_inv


def test_each_covariate_uses_its_own_sd():
    # 2 covariates (intercept, trt), 3 features.
    P, J, N = 2, 3, 4
    X = torch.ones(N, P, dtype=torch.float64)
    Y = torch.ones(N, J, dtype=torch.float64)
    model = nbm.NegativeBinomialRegressionModel(X, Y, lam=1., shape=3., scale=2., dispersion=np.ones(J))

    # The likelihood computes X @ beta.reshape(P, J): row 0 = intercept, row 1 = trt.
    beta_matrix = torch.tensor([[1., 2., 3.],      # intercept coefficients for features 0, 1, 2
                                [10., 20., 30.]],  # trt coefficients for features 0, 1, 2
                               dtype=torch.float64)
    beta = beta_matrix.flatten()  # = [1, 2, 3, 10, 20, 30], the layout model.beta uses

    # Different prior sd per covariate: intercept sd = 1, trt sd = 100.
    sd = torch.tensor([1., 100.], dtype=torch.float64)
    with torch.no_grad():
        model.psi.copy_(softplus_inv(sd))

    # Intended prior: every intercept coefficient uses sd 1, every trt coefficient uses sd 100.
    expected = log_normal(beta_matrix, torch.zeros_like(beta_matrix), sd[:, None]).sum()

    actual = model.log_beta_prior(beta)

    assert torch.allclose(actual, expected), (
        f"\nlog_beta_prior = {actual.item():.3f}, expected {expected.item():.3f}\n"
        f"beta.reshape(J, P) mixes covariates within a column:\n{beta.reshape(J, P).numpy()}\n"
        f"beta.reshape(P, J).T keeps each covariate in its own column:\n{beta.reshape(P, J).T.numpy()}"
    )
