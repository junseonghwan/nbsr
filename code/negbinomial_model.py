import torch

import os
import math
import numpy as np
import pandas as pd
import time

from distributions import *

class NegativeBinomialRegressionModel(torch.nn.Module):
    def __init__(self, X, Y, m0=0, s0=2, shape=2, scale=1, pivot=False):
        super(NegativeBinomialRegressionModel, self).__init__()
        self.X = torch.tensor(X, dtype=torch.float64)
        self.Y = torch.tensor(Y, dtype=torch.float64)
        self.pivot = pivot
        self.softplus = torch.nn.Softplus()
        self.sample_count = Y.shape[0]
        self.covariate_count = X.shape[1]
        self.rna_count = Y.shape[1]
        self.m0 = torch.tensor(m0, requires_grad=False)
        self.s0 = torch.tensor(s0, requires_grad=False)
        self.shape = torch.tensor(shape, requires_grad=False)
        self.scale = torch.tensor(scale, requires_grad=False)

        # The parameters we adjust during training.
        self.dim = self.rna_count - 1 if pivot else self.rna_count
        self.mu = torch.nn.Parameter(torch.randn(self.dim), requires_grad=True)
        self.beta = torch.nn.Parameter(torch.randn(self.covariate_count * self.dim), requires_grad=True)
        self.phi = torch.nn.Parameter(torch.randn(self.rna_count), requires_grad=True)

    def log_likelihood(self, mu, beta):
        # reshape beta:
        beta_ = torch.reshape(beta, (self.covariate_count, self.dim))
        log_unnorm_exp = mu + torch.matmul(self.X, beta_)
        if self.pivot:
            log_unnorm_exp = torch.column_stack((torch.zeros(self.sample_count), log_unnorm_exp))
        norm = torch.logsumexp(log_unnorm_exp, 1)
        norm_expr = torch.exp(log_unnorm_exp - norm[:,None])
        log_lik = torch.zeros(1)
        for (pi, y) in zip(norm_expr, self.Y):
            s = torch.sum(y)
            log_lik += torch.sum(log_negbinomial(y, s * pi, self.softplus(self.phi)))
        return(log_lik)

    def log_posterior(self, mu, beta):
        log_lik = self.log_likelihood(mu, beta)
        log_prior = torch.sum(log_normal(self.mu, self.m0, self.s0)) + torch.sum(log_normal(self.beta, self.m0, self.s0))
        log_prior += torch.sum(log_gamma(self.softplus(self.phi), self.shape, self.scale))
        log_posterior = log_lik + log_prior
        return(log_posterior)

    