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

# This model extends the basic NBSR model but with observation specific dispersion.
class NBSRDispersion(NegativeBinomialRegressionModel):

    def __init__(self, X, Y, prior_sd=None, pivot=False):
        super().__init__(X, Y, None, None, prior_sd, pivot)
        # Observation specific dispersion parameters.
        self.phi = torch.nn.Parameter(torch.randn(self.sample_count, self.rna_count, dtype=torch.float64), requires_grad=True)

    def log_obs_likelihood(self, pi, phi):
        mu = self.s[:,None] * pi
        log_lik_vals = log_negbinomial(self.Y, mu, phi)
        return log_lik_vals

    def log_obs_likelihood2(self, beta):
        pi, _ = self.predict(beta, self.X)
        mu = self.s[:,None] * pi
        phi = self.softplus(self.phi)
        log_lik_vals = log_negbinomial(self.Y, mu, phi)
        return log_lik_vals

    # def log_posterior(self, beta):
    #     pi,_ = self.predict(beta, self.X)
    #     phi = self.softplus(self.phi)

    #     # Compute the log likelihood of Y.
    #     log_obs_lik = self.log_obs_likelihood(pi, phi).sum()
    #     # Compute the log priors.
    #     log_nbsr_prior = self.log_prior(self.beta)
    #     log_dispersion_prior = self.dispersion_prior.log_density(phi, pi).sum()
    #     log_posterior = log_obs_lik + log_dispersion_prior + log_nbsr_prior
    #     return log_posterior
    
    # compute log likelihood using NBSR parameters beta.
    # def log_obs_likelihood2(self, beta, phi=None):
    #     pi,_ = self.predict(beta, self.X)
    #     dispersion = torch.exp(self.disp_model.forward(pi)) if phi is None else phi
    #     log_lik_vals = self.log_obs_likelihood(pi, dispersion)
    #     return log_lik_vals

    # Evaluates the log of the joint likelihood by drawing Monte Carlo samples.
    # Returns MxNxK log likelihood values and phi samples.
    def log_likelihood_samples(self, mc_samples):
        pi,_ = self.predict(self.beta, self.X)

        # We approximate the Q function by sampling the dispersion.
        N, K = pi.shape
        epsilon = torch.randn(mc_samples, N, K)

        phi = torch.exp(self.disp_model.forward(pi).unsqueeze(0) + epsilon * self.disp_model.get_sd())

        # Compute the log likelihood of Y.
        log_obs_lik = torch.stack([self.log_obs_likelihood(pi, phi_m) for phi_m in phi])
        log_dispersion_lik = torch.stack([self.disp_model.log_density(phi_m, pi) for phi_m in phi])
        return (log_obs_lik + log_dispersion_lik, phi)
        
    ### Gradients of the model
    def log_lik_gradient_persample(self, beta):
        """
        Computes the gradient of the log-likelihood function with respect to the model parameters for each sample.

        Args:
            beta (torch.Tensor): A tensor of shape (covariate_count * dim,) containing the model parameters.

        Returns:
            torch.Tensor: A tensor of shape (sample_count, covariate_count * dim) containing the gradient of the log-likelihood function with respect to the model parameters for each sample.
        """        
        #beta_ = torch.reshape(beta, (self.covariate_count, self.dim))
        pi,_ = self.predict(beta, self.X) # N x K
        phi = torch.exp(self.disp_model.forward(pi)) # N x K
        #J = self.rna_count-1 if self.pivot else self.rna_count
        I_K = torch.eye(self.rna_count) # K x K
        b1_term = (1 + self.disp_model.b1) # scalar

        # grad[i,k] = \sum_j \nabla_k \log P(Y_{ij}).
        grad = torch.zeros(self.sample_count, self.dim * self.covariate_count)
        for idx, (pi_i, phi_i, x, y) in enumerate(zip(pi, phi, self.X, self.Y)):
            #import pdb; pdb.set_trace()
            s = torch.sum(y)
            mean = s * pi_i
            sigma2 = mean + phi_i * (mean ** 2)
            pp = mean/sigma2
            rr = 1 / phi_i
            I_pi = I_K - pi_i.unsqueeze(1)
            xx = x.view(1, 1, self.covariate_count)
            I_pi_x = I_pi.unsqueeze(-1) * xx

            temp0 = -rr * (1 - pp) * b1_term
            temp1 = y * pp * b1_term
            temp2 = -self.disp_model.b1 * rr * (torch.digamma(y + rr) - torch.digamma(rr) + torch.log(pp))
            temp = (temp0 + temp1 + temp2)
            temp_reshaped = temp.view(self.rna_count, 1, 1)
            result = I_pi_x.transpose(0,1) * temp_reshaped
            result_sum = result.sum(dim=0)
            grad_idx = result_sum[:-1,:] if self.pivot else result_sum
            grad[idx,:] = grad_idx.transpose(0,1).flatten()
        return grad

    def log_posterior_gradient(self, beta):
        """
        Computes the gradient of the log posterior distribution with respect to the model parameters.

        Args:
            beta (torch.Tensor): A tensor of shape (dim * covariate_count,) representing the model parameters.
            tensorized (bool): Whether to use the tensorized version of the gradient computation.

        Returns:
            torch.Tensor: A tensor of the same shape as `beta` representing the gradient of the log posterior distribution.
        """
        log_prior_grad = self.log_beta_prior_gradient(beta)
        
        log_lik_grad = self.log_lik_gradient_persample(beta).sum(0)
        return log_lik_grad + log_prior_grad
