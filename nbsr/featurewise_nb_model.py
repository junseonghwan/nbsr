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

    def __init__(self, n_covariates, dispersion_model, beta_prior_sd=5.0, dtype=torch.float32):
        """
        beta_prior_sd: scalar or length-n_covariates vector of prior sds for beta ~ N(0, sd^2).
            A vector lets each covariate carry its own sd (e.g. an effectively flat prior on the intercept).
        """
        super().__init__()

        self.n_covariates = n_covariates

        self.beta = nn.Parameter(
            torch.zeros(self.n_covariates, dtype=dtype)
        )

        self.dispersion_model = dispersion_model

        beta_prior_sd = torch.as_tensor(beta_prior_sd, dtype=dtype)
        if beta_prior_sd.ndim == 0:
            beta_prior_sd = beta_prior_sd.expand(self.n_covariates)
        assert beta_prior_sd.shape == (self.n_covariates,), \
            f"beta_prior_sd must be a scalar or have shape ({self.n_covariates},), got {tuple(beta_prior_sd.shape)}"
        self.register_buffer("beta_prior_sd", beta_prior_sd.clone())

    def mean_model(self, beta, X, sf):
        """
        x: N x 1 of covariates.
        sf: N
        """

        eta = X @ beta
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
        mu = self.mean_model(self.beta, X, sf)
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
    
        log_lik = log_negbinomial(y, mu, phi).sum()
        beta_log_prior = self.log_prior()
        disp_log_prior = self.dispersion_model.log_prior()
        return log_lik + beta_log_prior + disp_log_prior

    def loss(self, y, X, W, sf):
        return -self.log_posterior(y, X, W, sf)

    def pack(self):
        return torch.cat([p.detach().flatten() for _, p in self.named_parameters()])

    def unpack(self, theta):
        params, idx = {}, 0
        for name, shape in [(n, p.shape) for n, p in self.named_parameters()]:
            n_elements = shape.numel()
            params[name] = theta[idx : idx + n_elements].reshape(shape)
            idx += n_elements
        return params
