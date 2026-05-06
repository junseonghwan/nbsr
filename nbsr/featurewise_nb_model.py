import torch
import torch.nn as nn

from nbsr.regression_context import RegressionContext
from nbsr.distributions import log_negbinomial, log_normal

class FeaturewiseNegBinom(nn.Module):
    """
    Feature-wise negative binomial regression.

    Mean model:
        log mu_ij = log s_i + X_i beta_j

    Dispersion model:
        log phi_ij = dispersion_model(W, pi, size_factors)
    """

    def __init__(self, n_covariates, dispersion_model, beta_init_sd=1.0):
        super().__init__()

        self.n_covariates = n_covariates

        self.beta = nn.Parameter(
            torch.zeros(self.n_covariates)
        )

        self.dispersion_model = dispersion_model

        self.register_buffer("beta_prior_sd", torch.as_tensor(beta_init_sd))

    def mean_model(self, X, sf):
        """
        x: N x 1 of covariates.
        sf: N
        """

        eta = X @ self.beta
        eta = eta + torch.log(sf)
        mu = torch.exp(eta)
        return mu

    def forward(self, X, W, sf):
        """
        Returns
        -------
        mu: vector of length N
        log_phi: vector of length N
        """

        mu = self.mean_model(X, sf)
        context = RegressionContext(X=X, W=W, size_factors=sf, mu=mu, pi=None)
        log_phi = self.dispersion_model(context)
        return mu, log_phi

    def log_prior(self):
        return log_normal(
            self.beta,
            torch.zeros_like(self.beta),
            self.beta_prior_sd
        ).sum()

    def log_posterior(self, y, X, W, sf):
        mu, log_phi = self.forward(X, W, sf)
        phi = torch.exp(log_phi)
    
        ll = log_negbinomial(y, mu, phi).sum()
        lp = self.dispersion_model.log_prior()
        return ll + lp

    def loss(self, y, X, W, sf):
        return -self.log_posterior(y, X, W, sf)
