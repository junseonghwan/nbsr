import torch

import os
import math
import numpy as np
import pandas as pd
from scipy.stats import invgamma
import time

from nbsr.distributions import log_negbinomial, log_gamma, log_normal, log_invgamma, softplus_inv
from nbsr.negbinomial_model import NegativeBinomialRegressionModel
from nbsr.grbf import GaussianRBF

class ZINBSR(NegativeBinomialRegressionModel):
    def __init__(self, X, Y, Z, dispersion=None, prior_sd=None, pivot=False):
        super().__init__(X, Y, dispersion, prior_sd, pivot)
        # Z is a tensor storing the covariates to be used in predicting zero inflation.
        assert(isinstance(Z, torch.Tensor))
        self.Z = Z
        self.b = torch.nn.Parameter(torch.randn(self.Z.shape[1], dtype=torch.float64), requires_grad=True)
        # if knot_count > 0:
        #     self.grbf = GaussianRBF(0, torch.max(torch.log(Y)), knot_count)

    def log_likelihood(self, beta):
        beta_ = torch.reshape(beta, (self.covariate_count, self.dim))
        log_unnorm_exp = torch.matmul(self.X, beta_)
        if self.pivot:
            log_unnorm_exp = torch.column_stack((log_unnorm_exp, torch.zeros(self.sample_count)))
        norm = torch.logsumexp(log_unnorm_exp, 1)
        norm_expr = torch.exp(log_unnorm_exp - norm[:,None])
        mu = self.s[:, None] * norm_expr
        log_lik_vals = log_negbinomial(self.Y, mu, self.softplus(self.phi))

        # if self.grbf is not None:
        #     dispersion = self.grbf.evaluate(mu)
        #     log_lik_vals = log_negbinomial(self.Y, mu, dispersion)
        # else:
        #     log_lik_vals = log_negbinomial(self.Y, mu, self.softplus(self.phi))

        # compute epsilon.
        Zb = torch.matmul(self.Z, self.b)
        expZ = torch.exp(Zb)
        #epsilon = expZ / (1 + expZ)
        epsilon = 0.3 / (1 + expZ)
        epsilon = epsilon[:,None]

        log_lik_nb = torch.log(1 - epsilon) + log_lik_vals
        log_lik_zero = torch.log(epsilon) + torch.log(self.Y == 0).type(self.Y.dtype)
        log_lik_mixture = torch.logaddexp(log_lik_nb, log_lik_zero)

        log_lik = torch.sum(log_lik_mixture)  # Sum all values
        return(log_lik)
    
    def log_posterior(self, beta):
        log_lik = self.log_likelihood(beta)
        sd = self.softplus(self.psi)
        # normal prior on beta -- 0 mean and sd = softplus(psi).
        log_prior1 = self.log_beta_prior(self.beta)
        # inv gamma prior on var = sd^2 -- hyper parameters specified to the model.
        log_prior2 = torch.sum(log_invgamma(sd**2, self.beta_var_shape, self.beta_var_scale))
        log_prior3 = torch.sum(log_normal(self.b, torch.zeros_like(self.b), torch.tensor(5.0)))
        log_posterior = log_lik + log_prior1 + log_prior2 + log_prior3
        return(log_posterior)
