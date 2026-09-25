"""Dispersion models of the feature-wise NB regression (nbsr.fnb_stats).

    log phi_ij = a_j + b_j log mu_ij + w_i' gamma_j

with the DESeq2 dispersion-trend prior on a_j. Not to be confused with nbsr.dispersion.DispersionModel,
the dispersion model of the softmax (NBSR) family.
"""
import torch

from nbsr.regression_context import RegressionContext
from nbsr.distributions import log_normal

class LogDispersionTrendPrior(torch.nn.Module):
    def __init__(self, intercept, slope, disp_prior_var, dtype=torch.float32):
        super().__init__()
        self.register_buffer("a0", torch.as_tensor(intercept, dtype=dtype))
        self.register_buffer("a1", torch.as_tensor(slope, dtype=dtype))
        self.register_buffer("disp_prior_var", torch.as_tensor(disp_prior_var, dtype=dtype))

    @property
    def sd(self):
        return torch.sqrt(self.disp_prior_var)

    def mean(self, mu_bar):
        return torch.log(self.a1 / mu_bar + self.a0)

    def log_density(self, a_j, mu_bar):
        return log_normal(a_j, self.mean(mu_bar), self.sd)
    
class BaseDispersionModel(torch.nn.Module):
    
    def forward(self, context : RegressionContext) -> torch.Tensor:
        raise NotImplementedError

class MeanPowerCovariateDispersion(BaseDispersionModel):
    """
    log phi_ij = a_j + b_j log mu_ij + W_i^T gamma_j
    """

    def __init__(
        self,
        n_covariates,
        disp_trend_prior : LogDispersionTrendPrior,
        mu_bar : torch.tensor,
        b_prior_sd : float = 0.1,
        gamma_prior_sd : float =1.0,
        dtype=torch.float32,
    ):
        super().__init__()

        self.n_covariates = n_covariates

        # Start the dispersion intercept at the DESeq2 trend prior mean for this gene.
        with torch.no_grad():
            a_init = disp_trend_prior.mean(torch.as_tensor(mu_bar, dtype=dtype)).reshape(1)

        self.a = torch.nn.Parameter(a_init.clone().to(dtype))
        self.b = torch.nn.Parameter(
            torch.zeros(1, dtype=dtype)
            )

        if self.n_covariates > 0:
            self.gamma = torch.nn.Parameter(
                torch.zeros(self.n_covariates, dtype=dtype)
            )
        else:
            self.gamma = None

        self.disp_trend_prior = disp_trend_prior
        self.register_buffer("mu_bar", torch.as_tensor(mu_bar, dtype=dtype))
        self.register_buffer("b_prior_sd", torch.as_tensor(b_prior_sd, dtype=dtype))
        self.register_buffer("gamma_prior_sd", torch.as_tensor(gamma_prior_sd, dtype=dtype))

    def forward(self, context: RegressionContext) -> torch.Tensor:
        log_mu = torch.log(context.mu)
        log_phi = self.a + self.b * log_mu

        if self.gamma is not None:
            if context.W is None:
                raise ValueError("W is required when n_disp_covariates > 0.")
            assert self.n_covariates == context.W.shape[1]

            log_phi = log_phi + context.W @ self.gamma

        return log_phi

    def compute_log_prior(self, a, b, mu_bar, gamma=None):
        lp = self.disp_trend_prior.log_density(a, mu_bar).sum()
        lp += log_normal(b, torch.zeros_like(b), self.b_prior_sd).sum()
        if gamma is not None:
            lp += log_normal(gamma, torch.zeros_like(gamma), self.gamma_prior_sd).sum()
        return lp

    def log_prior(self):
        return self.compute_log_prior(self.a, self.b, self.mu_bar, self.gamma)

