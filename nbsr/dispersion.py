import torch

from nbsr.distributions import log_negbinomial, log_normal, softplus_inv, log_lognormal

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

        # self.Y = Y
        # self.Z = Z # NxP design matrix for covariates to use in modeling the dispersion.

        # self.R = Y.sum(1)
        # self.log_R = torch.log(self.R)

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
