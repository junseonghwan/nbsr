import torch

import os
import math
import numpy as np
import pandas as pd
from scipy.stats import invgamma
import time

from nbsr.distributions import log_negbinomial, log_gamma, log_normal, log_invgamma, softplus_inv

class NegativeBinomialRegressionModel(torch.nn.Module):
    def __init__(self, X, Y, dispersion=None, pivot=True):
        super(NegativeBinomialRegressionModel, self).__init__()
        # Assume X is a pandas dataframe.
        assert(isinstance(X, pd.DataFrame))
        self.X_df = X
        self.X = torch.tensor(pd.get_dummies(X, drop_first=True, dtype=int).to_numpy(), dtype=torch.float64)
        self.Y = torch.tensor(Y, dtype=torch.float64)
        self.pivot = pivot
        self.softplus = torch.nn.Softplus()
        self.sample_count = Y.shape[0]
        self.covariate_count = self.X.shape[1]
        self.rna_count = Y.shape[1]
        # self.m0 = torch.tensor(m0, requires_grad=False)
        # self.s0 = torch.tensor(s0, requires_grad=False)
        # self.shape = torch.tensor(shape, requires_grad=False)
        # self.scale = torch.tensor(scale, requires_grad=False)
        self.converged = False
        print("RNA count:", self.rna_count)
        print("Sample count:", self.sample_count)
        print("Covariate count:", self.covariate_count)

        # The parameters we adjust during training.
        self.dim = self.rna_count - 1 if pivot else self.rna_count
        self.mu = torch.nn.Parameter(torch.randn(self.dim, dtype=torch.float64), requires_grad=True)
        self.beta = torch.nn.Parameter(torch.randn(self.covariate_count * self.dim, dtype=torch.float64), requires_grad=True)
        if dispersion is None:
            self.phi = torch.nn.Parameter(torch.randn(self.rna_count, dtype=torch.float64), requires_grad=True)
        else:
            self.phi = softplus_inv(torch.tensor(dispersion + 1e-9, requires_grad=False))
        self.psi = torch.nn.Parameter(torch.randn(self.covariate_count+1, dtype=torch.float64), requires_grad=True)

    def specify_beta_prior(self, lam, beta_var_shape, beta_var_scale):
        self.lam = torch.tensor(lam, requires_grad=False)
        self.beta_var_shape = torch.tensor(beta_var_shape, requires_grad=False)
        self.beta_var_scale = torch.tensor(beta_var_scale, requires_grad=False)
        sd = np.sqrt(invgamma.rvs(a=self.beta_var_shape, scale=self.beta_var_scale, size=self.covariate_count+1))
        print("Initial sd:", sd)
        self.psi = torch.nn.Parameter(softplus_inv(torch.tensor(sd)), requires_grad=True)
        print("Initial psi:", self.psi)

        #self.psi = softplus_inv(torch.tensor(invgamma.rvs(self.beta_var_shape, self.beta_var_scale, size=self.covariate_count+1)))

    # Useful for getting initial estimates of beta and dispersion parameters.
    def log_likelihood(self, mu, beta):
        # reshape beta:
        beta_ = torch.reshape(beta, (self.covariate_count, self.dim))
        log_unnorm_exp = mu + torch.matmul(self.X, beta_)
        if self.pivot:
            log_unnorm_exp = torch.column_stack((log_unnorm_exp, torch.zeros(self.sample_count)))
        norm = torch.logsumexp(log_unnorm_exp, 1)
        norm_expr = torch.exp(log_unnorm_exp - norm[:,None])
        log_lik = torch.zeros(1)
        for (pi, y) in zip(norm_expr, self.Y):
            s = torch.sum(y)
            log_lik += torch.sum(log_negbinomial(y, s * pi, self.softplus(self.phi)))
        return(log_lik)

    def log_posterior(self, mu, beta):
        log_lik = self.log_likelihood(mu, beta)
        # normal prior on beta -- 0 mean and var given as input from glmGamPoi.
        sd = self.softplus(self.psi)
        log_prior1 = torch.sum(log_normal(self.mu, 0, sd[0]/self.lam)) + torch.sum(log_normal(self.beta, 0, sd[1:]/self.lam))
        log_prior2 = torch.sum(log_invgamma(sd**2, self.beta_var_shape, self.beta_var_scale))
        log_posterior = log_lik + log_prior1 + log_prior2
        return(log_posterior)

    def log_lik_gradient(self, mu, beta):
        beta_ = torch.reshape(beta, (self.covariate_count, self.dim))
        dispersion = self.softplus(self.phi)
        J = self.rna_count
        log_unnorm_exp = mu + torch.matmul(self.X, beta_)
        if self.pivot:
            log_unnorm_exp = torch.column_stack((log_unnorm_exp, torch.zeros(self.sample_count)))
        norm = torch.logsumexp(log_unnorm_exp, 1)
        norm_expr = torch.exp(log_unnorm_exp - norm[:,None])

        grad = torch.zeros(self.sample_count, self.dim * self.covariate_count)
        for idx, (pi, x, y) in enumerate(zip(norm_expr, self.X, self.Y)):
            s = torch.sum(y)
            mean = s * pi
            sigma2 = mean + dispersion * (mean ** 2)
            p = mean / sigma2 # equivalent to r / (mu + r).
            r = 1 / dispersion
            A = torch.eye(J) - pi.repeat((J, 1))
            A = A.transpose(0, 1)
            temp1 = 1/mean - (1 + 2 * dispersion * mean)/sigma2
            temp2 = 2/mean - (1 + 2 * dispersion * mean)/sigma2
            temp = mean * (r * temp1 + y * temp2)
            ret = x.repeat((J, 1)).transpose(0,1) * temp
            ret2 = ret.unsqueeze(1).repeat(1, J, 1)
            ret3 = ret2 * A
            grad[idx,:] = torch.sum(ret3, 2).flatten()
        return torch.sum(grad, 0)

    def predict(self, mu, beta, X):
        beta_ = torch.reshape(beta, (self.covariate_count, self.dim))
        log_unnorm_exp = mu + torch.matmul(self.X, beta_)
        if self.pivot:
            log_unnorm_exp = torch.column_stack((log_unnorm_exp, torch.zeros(self.sample_count)))
        norm = torch.logsumexp(log_unnorm_exp, 1)
        norm_expr = torch.exp(log_unnorm_exp - norm[:,None])
        pi = norm_expr
        return(pi, log_unnorm_exp)



