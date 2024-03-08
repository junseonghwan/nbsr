import torch

import os
import math
import numpy as np
import pandas as pd
from scipy.stats import invgamma
import time

from nbsr.distributions import log_negbinomial, log_normal, log_invgamma, softplus_inv, softplus
from nbsr.negbinomial_model import NegativeBinomialRegressionModel
from nbsr.dispersion import DispersionModel

# This model extends the basic NBSR model with dispersion
class NBSR_dispersion(NegativeBinomialRegressionModel):

    def __init__(self, X, Y, Z=None, dispersion_prior=None, dispersion=None, prior_sd=None, pivot=False):
        super().__init__(X, Y, dispersion_prior, dispersion, prior_sd, pivot)
        self.Z = Z
        self.disp_model = DispersionModel(self.Y, self.Z)

    def log_likelihood(self, mu, phi):
        # Define log_liklihood that uses the new architecture.
        log_lik_vals = log_negbinomial(self.Y, mu, phi)
        return log_lik_vals.sum()

    def log_posterior(self, beta):
        pi,_ = self.predict(beta, self.X)
        mu = self.s[:, None] * pi
        log_pi = torch.log(pi)
        dispersion = torch.exp(self.disp_model.forward(log_pi))

        # Compute the log likelihood of Y
        log_lik = self.log_likelihood(mu, dispersion)
        # Compute the log of prior.
        sd = self.softplus(self.psi)
        # normal prior on beta -- 0 mean and sd = softplus(psi).
        log_beta_prior = self.log_beta_prior(beta)
        # inv gamma prior on var = sd^2 -- hyper parameters specified to the model.
        log_var_prior = torch.sum(log_invgamma(sd**2, self.beta_var_shape, self.beta_var_scale))
        norm_expr, _ = self.predict(beta, self.X)
        #phi_sd = softplus(self.dispersion_prior.psi)
        log_dispersion_prior = 0
        if self.dispersion_prior is not None:
            log_dispersion_prior = torch.sum(self.dispersion_prior.log_density(torch.log(dispersion), torch.mean(mu, 0)))
        log_posterior = log_lik + log_beta_prior + log_var_prior + log_dispersion_prior
        return log_posterior
