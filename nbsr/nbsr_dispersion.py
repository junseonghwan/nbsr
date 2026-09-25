"""NBSR with the trended dispersion model (nbsr.dispersion.DispersionModel, NB2 convention):

    log dispersion_ij = -(b_0 + b_j + b_pi logit(pi_ij) + w_i' b_w)

The dispersion depends on beta through pi, so the derivatives carry the chain-rule terms
c_ij = d log dispersion_ij / d log pi_ij = -b_pi / (1 - pi_ij) and c2_ij = -b_pi pi_ij / (1 - pi_ij)^2.
"""
import torch

from nbsr.distributions import log_negbinomial, log_invgamma
from nbsr.negbinomial_model import NegativeBinomialRegressionModel


class NBSRTrended(NegativeBinomialRegressionModel):

    def __init__(self, X, Y, disp_model, lam, shape, scale, pivot=False):
        super().__init__(X, Y, lam=lam, shape=shape, scale=scale, dispersion_prior=disp_model, dispersion=None, pivot=pivot)
        assert disp_model.feature_count == self.rna_count, "dispersion model built for a different number of features"
        self.phi = None

    def dispersion(self, pi):
        return torch.exp(self.disp_model.log_dispersion(pi))

    def log_likelihood(self, pi, phi):
        """Log-likelihood at an explicit composition and dispersion (used when fitting the dispersion model
        to externally supplied means)."""
        return log_negbinomial(self.Y, self.s[:, None] * pi, phi).sum()

    def log_likelihood_beta(self, beta):
        pi, _ = self.predict(beta, self.X)
        return self.log_likelihood(pi, self.dispersion(pi))

    def log_posterior(self, beta):
        log_lik = self.log_likelihood_beta(beta)
        sd = self.softplus(self.psi)
        log_beta_prior = self.log_beta_prior(beta)
        log_var_prior = torch.sum(log_invgamma(sd ** 2, self.beta_var_shape, self.beta_var_scale))
        return log_lik + log_beta_prior + log_var_prior + self.disp_model.log_prior()

    def forward(self, beta):
        return self.log_posterior(beta)

    def _dispersion_terms(self, pi):
        phi = self.dispersion(pi)
        b_pi = self.disp_model.b_pi.detach().reshape(())
        d1, d2 = self.disp_model.link_derivatives(pi)
        return phi, -b_pi * d1, -b_pi * d2
