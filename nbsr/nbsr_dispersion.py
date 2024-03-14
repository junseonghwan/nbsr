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
class NBSRDispersion(NegativeBinomialRegressionModel):

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
        phi = torch.exp(self.disp_model.forward(log_pi))

        # Compute the log likelihood of Y
        log_lik = self.log_likelihood(mu, phi)
        # Compute the log of prior.
        sd = self.softplus(self.psi)
        # normal prior on beta -- 0 mean and sd = softplus(psi).
        log_beta_prior = self.log_beta_prior(beta)
        # inv gamma prior on var = sd^2 -- hyper parameters specified to the model.
        log_var_prior = torch.sum(log_invgamma(sd**2, self.beta_var_shape, self.beta_var_scale))
        log_dispersion_prior = 0
        if self.dispersion_prior is not None:
            log_dispersion_prior = torch.sum(self.dispersion_prior.log_density(torch.log(phi), torch.mean(mu, 0)))
        log_posterior = log_lik + log_beta_prior + log_var_prior + log_dispersion_prior
        return log_posterior
    
    def forward(self, beta):
        return self.log_posterior(beta)

    def log_likelihood2(self, beta):
        # Define log_liklihood that uses the new architecture.
        pi,_ = self.predict(beta, self.X)
        mu = self.s[:, None] * pi
        log_pi = torch.log(pi)
        phi = torch.exp(self.disp_model.forward(log_pi))
        log_lik_vals = log_negbinomial(self.Y, mu, phi)
        return log_lik_vals.sum()

    ### Gradient of the model
    def log_lik_gradient_persample(self, beta):
        """
        Computes the gradient of the log-likelihood function with respect to the model parameters for each sample.

        Args:
            beta (torch.Tensor): A tensor of shape (covariate_count * dim,) containing the model parameters.

        Returns:
            torch.Tensor: A tensor of shape (sample_count, covariate_count * dim) containing the gradient of the log-likelihood function with respect to the model parameters for each sample.
        """        
        #beta_ = torch.reshape(beta, (self.covariate_count, self.dim))
        pi,_ = self.predict(beta, self.X)
        log_pi = torch.log(pi)
        phi = torch.exp(self.disp_model.forward(log_pi))
        J = self.rna_count

        grad = torch.zeros(self.sample_count, self.dim * self.covariate_count)
        for idx, (pi_i, phi_i, x, y) in enumerate(zip(pi, phi, self.X, self.Y)):
            s = torch.sum(y)
            mean = s * pi_i
            sigma2 = mean + phi_i * (mean ** 2)
            pp = mean/sigma2
            r = 1 / phi_i
            A = torch.eye(J) - pi_i.repeat((J, 1))
            A = A.transpose(0, 1)
            xx = x.repeat((J,1)).transpose(0,1)

            temp0 = -r * (1 - pp) * (1 + self.disp_model.b1)
            temp1 = y * pp * (1 + self.disp_model.b1)
            temp2 = xx * (temp0 + temp1)
            temp3 = temp2.unsqueeze(1).repeat(1, J, 1)
            val0 = temp3 * A

            temp1 = -self.disp_model.b1 * (torch.digamma(y + r) - torch.digamma(r) + torch.log(pp)) / phi_i
            temp2 = xx * temp1
            temp3 = temp2.unsqueeze(1).repeat(1, J, 1)
            val1 = temp3 * A

            grad[idx,:] = torch.sum(val0 + val1, 2).flatten()
        return grad
