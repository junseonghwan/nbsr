import torch

from nbsr.distributions import log_negbinomial, log_normal

# Model for dispersion parameters of the NBSR.
class DispersionModel(torch.nn.Module):
    def __init__(self, Y, Z = None):
        super().__init__()
        self.softplus = torch.nn.Softplus()
        self.Y = Y
        self.Z = Z # NxP design matrix for covariates to use in modeling the dispersion.

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
        self.b0 = torch.nn.Parameter(torch.randn(self.feature_count, dtype=torch.float64), requires_grad=True)
        self.b1 = torch.nn.Parameter(torch.randn(1, dtype=torch.float64), requires_grad=True)
        self.b2 = torch.nn.Parameter(torch.randn(1, dtype=torch.float64), requires_grad=True)
        if self.Z is None:
            self.beta = None
            self.covariate_count = 0
        else:
            self.covariate_count = Z.shape[1]
            self.beta = torch.nn.Parameter(torch.randn(self.covariate_count, dtype=torch.float64), requires_grad=True)
        # softplus(self.psi) to get tau -- we will put this back when we implement sampling feature.
        #self.psi = torch.nn.Parameter(torch.randn(self.feature_count, dtype=torch.float64), requires_grad=True)

    def output(self, log_pi):
        # log \phi_{ij} = b_{0,j} + b_1 log \pi_{i,j} + b_2 log R_i + \beta' z_i + \epsilon_j.
        val0 = self.b0.unsqueeze(-1).transpose(0,1).expand(self.sample_count, self.feature_count)
        val1 = self.b1 * log_pi
        val2 = self.b2 * self.log_R.unsqueeze(-1).expand(-1, self.feature_count)
        if self.Z is None:
            val3 = 0
        else:
            val3 = torch.mm(self.Z, self.beta.unsqueeze(-1)).expand(-1, self.feature_count)
        log_phi = val0 + val1 + val2 + val3
        return(log_phi)

    # log P(Y | \mu, dispersion) + log P(dispersion | \theta)
    def log_likelihood(self, pi):
        log_pi = torch.log(pi)
        mu = pi * self.R[:,None]

        log_phi = self.output(log_pi)
        log_lik_vals = log_negbinomial(self.Y, mu, torch.exp(log_phi))
        #tau = self.softplus(self.psi)
        #log_prior_vals = log_normal(log_phi, tau)
        #log_posterior = log_lik_vals.sum() + log_prior_vals.sum()  # Sum all values
        return(log_lik_vals.sum())
