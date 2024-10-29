import torch

from nbsr.distributions import log_negbinomial, log_normal, log_half_normal, log_lognormal

# LogNormal model for the dispersion parameters of the NBSRDispersion.
class DispersionModel(torch.nn.Module):
    def __init__(self, Y, Z = None, feature_specific_intercept = False, b0j=None):
        super().__init__()
        #self.softplus = torch.nn.Softplus()
        self.Z = Z # NxP design matrix for covariates to use in modeling the dispersion.

        self.feature_specific_intercept = feature_specific_intercept

        self.R = Y.sum(1)
        self.log_R = torch.log(self.R)

        # Formulate design matrix: dimension is N x K x (P+3).
        # For each sample i=1,...,N, the KxP matrix contains 
        # a column of 1's (intercept_j), 
        # a column over j log \pi_{i,j}, 
        # a column of log R_i
        # and remaining P variables as containined in Z.
        # In total we need P+3 parameters.
        self.sample_count = Y.shape[0]
        self.feature_count = Y.shape[1]
        if b0j is not None:
            self.b0 = torch.nn.Parameter(b0j, requires_grad=True)
            self.feature_specific_intercept = True
        elif feature_specific_intercept:
            self.b0 = torch.nn.Parameter(torch.randn(self.feature_count, dtype=torch.float64), requires_grad=True)
        else:
            self.b0 = torch.nn.Parameter(torch.randn(1, dtype=torch.float64), requires_grad=True)
        self.b1 = torch.nn.Parameter(-torch.abs(torch.randn(1, dtype=torch.float64)), requires_grad=True)
        self.b2 = torch.nn.Parameter(-torch.abs(torch.randn(1, dtype=torch.float64)), requires_grad=True)
        if self.Z is None:
            self.beta = None
            self.covariate_count = 0
        else:
            self.covariate_count = Z.shape[1]
            self.beta = torch.nn.Parameter(torch.randn(self.covariate_count, dtype=torch.float64), requires_grad=True)

        self.kappa = torch.nn.Parameter(torch.randn(1, dtype=torch.float64), requires_grad=True)

        # self.estimate_sd = True
        # if estimate_sd:
        #     # sd = softplus(self.tau)
        #     self.kappa = torch.nn.Parameter(torch.randn(1, dtype=torch.float64), requires_grad=True)
        #     self.std_normal = torch.distributions.Normal(loc=0., scale=1.)

    def get_sd(self):
        return torch.nn.functional.softplus(self.kappa.detach())

    # Compute the mean of the distribution.
    def forward(self, log_pi):
        # log_pi has shape (self.sample_count, self.feature_count)
        #assert(log_pi.shape[0] == self.sample_count)
        #assert(log_pi.shape[1] == self.feature_count)
        # log \phi_{ij} = b_{0,j} + b_1 log \pi_{i,j} + b_2 log R_i + \beta' z_i + \epsilon_j.
        if self.feature_specific_intercept:
            val0 = self.b0.unsqueeze(-1).transpose(0,1).expand(self.sample_count, self.feature_count)
        else:
            val0 = self.b0
        val1 = self.b1 * log_pi 
        val2 = self.b2 * self.log_R.unsqueeze(-1).expand(-1, self.feature_count)
        if self.Z is None:
            val3 = 0
        else:
            val3 = torch.mm(self.Z, self.beta.unsqueeze(-1)).expand(-1, self.feature_count)
        log_phi_mean = val0 + val1 + val2 + val3
        return(log_phi_mean)

    def log_prior(self):
        log_prior0 = log_normal(self.b0, torch.zeros_like(self.b0), torch.tensor(1.)).sum()
        # log_prior1 = log_half_normal(-self.b1, torch.tensor(0.1))
        # log_prior2 = log_half_normal(-self.b2, torch.tensor(0.1))
        log_prior1 = log_normal(self.b1, torch.tensor([0.]), torch.tensor(0.1))
        log_prior2 = log_normal(self.b2, torch.tensor([0.]), torch.tensor(0.1))
        log_prior = log_prior0 + log_prior1 + log_prior2
        return log_prior

    # LogNormal evaluated at phi | pi and current model parameter.
    def log_density(self, phi, pi):
        log_pi = torch.log(pi)

        log_phi_mean = self.forward(log_pi)
        log_lik_vals = log_lognormal(phi, log_phi_mean, self.get_sd())
        return(log_lik_vals)
