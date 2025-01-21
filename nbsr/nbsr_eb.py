import torch

from nbsr.distributions import log_negbinomial, softplus

# Estimates feature specific dispersion parameters.
class NBSREmpiricalBayes(torch.nn.Module):
    def __init__(self, Y, mu):
        super().__init__()
        self.Y = Y
        self.mu = mu
        self.feature_count = self.Y.shape[1]
        self.phi = torch.nn.Parameter(torch.randn(self.feature_count, dtype=torch.float64), requires_grad=True)

    def log_likelihood(self):
        log_lik_vals = log_negbinomial(self.Y, self.mu, softplus(self.phi))
        log_lik = torch.sum(log_lik_vals)  # Sum all values
        return log_lik
