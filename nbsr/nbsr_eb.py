import torch

from nbsr.distributions import log_negbinomial, softplus

# Estimates feature specific dispersion parameters.
class NBSREmpiricalBayes(torch.nn.Module):
    def __init__(self, Y, mu, observation_wise=False):
        super().__init__()
        self.Y = Y
        self.mu = mu
        self.feature_count = self.Y.shape[1]
        if observation_wise:
            self.phi = torch.nn.Parameter(torch.randn_like(self.mu, dtype=torch.float64), requires_grad=True)
        else:
            self.phi = torch.nn.Parameter(torch.randn(1, self.feature_count, dtype=torch.float64), requires_grad=True)

    def log_likelihood(self):
        log_lik_vals = log_negbinomial(self.Y, self.mu, softplus(self.phi))
        return log_lik_vals
