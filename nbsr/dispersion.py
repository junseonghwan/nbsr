import torch
import torch.nn.functional as F

from nbsr.regression_context import RegressionContext
from nbsr.distributions import log_negbinomial, log_normal, log_lognormal

class LogDispersionTrendPrior(torch.nn.Module):
    def __init__(self, intercept, slope, dtype=torch.float64):
        super().__init__()
        self.register_buffer("a0", torch.as_tensor(intercept, dtype=dtype))
        self.register_buffer("a1", torch.as_tensor(slope, dtype=dtype))
        # We will estimate the sd of the log dispersion trend from data. 
        # We will represent it as kappa defined on the real line and map it to positive line using softplus.
        self.kappa = torch.nn.Parameter(
            torch.zeros(1, dtype=dtype)
        )

    @property
    def sd(self):
        return F.softplus(self.kappa)

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
        mu_bar : float,
        b_prior_sd=0.1,
        gamma_prior_sd=1.0,
        dtype=torch.float32,
    ):
        super().__init__()

        self.n_covariates = n_covariates

        with torch.no_grad():
            a_init = disp_trend_prior.mean(
                torch.as_tensor(mu_bar, dtype=dtype)
            )   

        self.a = torch.nn.Parameter(torch.zeros(1, dtype=dtype))
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

    def log_prior(self):
        lp = self.disp_trend_prior.log_density(
            self.a,
            self.mu_bar
        ).sum()

        lp += log_normal(
            self.b,
            torch.zeros_like(self.b),
            self.b_prior_sd,
        ).sum()

        if self.gamma is not None:
            lp += log_normal(
                self.gamma,
                torch.zeros_like(self.gamma),
                self.gamma_prior_sd,
            ).sum()

        return lp

# Model for dispersion parameters of the NBSR.
class DispersionModel(torch.nn.Module):
    def __init__(self, Y, Z = None, estimate_sd=False):
        super().__init__()
        self.softplus = torch.nn.Softplus()
        
        self.register_buffer("Y", Y)
        if Z is not None:
            self.register_buffer("Z", Z)
        else:
            self.Z = None
        R = Y.sum(1)
        log_R = torch.log(R)
        self.register_buffer("R", R)
        self.register_buffer("log_R", log_R)

        # Formulate design matrix: dimension is N x K x (P+3).
        # For each sample i=1,...,N, the KxP matrix contains 
        # a column of 1's (intercept_j), 
        # a column over j log \pi_{i,j}, 
        # a column of log R_i
        # and remaining P variables as containined in Z.
        # In total we need P+3 parameters.
        self.sample_count = Y.shape[0]
        self.feature_count = Y.shape[1]
        # Initialize coefficients to small values.
        self.b0 = torch.nn.Parameter(0.05*torch.randn(1, dtype=torch.float64), requires_grad=True)
        self.b1 = torch.nn.Parameter(0.05*torch.randn(1, dtype=torch.float64), requires_grad=True)
        self.b2 = torch.nn.Parameter(0.05*torch.randn(1, dtype=torch.float64), requires_grad=True)
        if self.Z is None:
            self.beta = None
            self.covariate_count = 0
        else:
            self.covariate_count = Z.shape[1]
            self.beta = torch.nn.Parameter(torch.randn(self.covariate_count, dtype=torch.float64), requires_grad=True)
        
        self.estimate_sd = estimate_sd
        if estimate_sd:
            # Optimize kappa over real line -- get_sd() will map it to positive line.
            self.kappa = torch.nn.Parameter(torch.randn(self.feature_count, dtype=torch.float64), requires_grad=True)
        else:
            self.kappa = None

    def forward(self, pi):
        # log_pi has shape (self.sample_count, self.feature_count)
        #assert(log_pi.shape[0] == self.sample_count)
        #assert(log_pi.shape[1] == self.feature_count)
        # log \phi_{ij} = b_{0,j} + b_1 log \pi_{i,j} + b_2 log R_i + \beta' z_i + \epsilon_j.
        # if self.feature_specific_intercept:
        #     val0 = self.b0.unsqueeze(-1).transpose(0,1).expand(self.sample_count, self.feature_count)
        # else:
        #     val0 = self.b0
        val0 = self.b0
        val1 = self.b1 * torch.log(pi)
        val2 = self.b2 * self.log_R.unsqueeze(-1).expand(-1, self.feature_count)
        if self.Z is None:
            val3 = 0
        else:
            val3 = torch.mm(self.Z, self.beta.unsqueeze(-1)).expand(-1, self.feature_count)
        log_phi_mean = val0 + val1 + val2 + val3
        if self.estimate_sd:
            log_phi_mean += (self.get_sd() ** 2) * 0.5
        return(log_phi_mean)

    def log_prior(self):
        log_prior0 = log_normal(self.b0, torch.zeros_like(self.b0), torch.tensor(1.)).sum()
        log_prior1 = log_normal(self.b1, torch.zeros_like(self.b1), torch.tensor(0.1))
        log_prior2 = log_normal(self.b2, torch.zeros_like(self.b2), torch.tensor(0.1))
        log_prior = log_prior0 + log_prior1 + log_prior2
        return log_prior.sum()

    def log_density(self, phi, pi):
        # Log Normal distribution density.
        log_phi_mean = self.forward(pi)
        log_lik_vals = log_lognormal(phi, log_phi_mean, self.get_sd().unsqueeze(0))
        return(log_lik_vals)

    def get_sd(self):
        return self.softplus(self.kappa)

    # log P(Y | \mu, dispersion) + log P(dispersion | \theta)
    def log_posterior(self, pi):
        mu = pi * self.R[:,None]

        log_phi = self.forward(pi)
        log_lik_vals = log_negbinomial(self.Y, mu, torch.exp(log_phi))
        log_posterior = log_lik_vals.sum() + self.log_prior()
        return(log_posterior)
