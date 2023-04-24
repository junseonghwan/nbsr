import torch

import os
import math
import numpy as np
import pandas as pd
import time

class MultinomialRegressionModel(torch.nn.Module):
    def __init__(self, covariate_count, rna_count, m0, s0, pivot=False):
        super(MultinomialRegressionModel, self).__init__()
        self.pivot = pivot
        self.softplus = torch.nn.Softplus()
        self.covariate_count = covariate_count
        self.rna_count = rna_count
        self.m0 = torch.tensor(m0, requires_grad=False)
        self.s0 = torch.tensor(s0, requires_grad=False)

        # The parameters we adjust during training.
        dim = rna_count - 1 if pivot else rna_count
        self.mu = torch.nn.Parameter(torch.randn(dim), requires_grad=True)
        self.beta = torch.nn.Parameter(torch.randn(self.covariate_count, dim), requires_grad=True)

    def log_likelihood(self, X, Y):
        log_unnorm_exp = self.mu + torch.matmul(X, self.beta)
        if self.pivot:
            sample_count = Y.shape[0]
            log_unnorm_exp = torch.column_stack((torch.zeros(sample_count), log_unnorm_exp))
        norm = torch.logsumexp(log_unnorm_exp, 1)
        norm_expr = torch.exp(log_unnorm_exp - norm[:,None])
        log_lik = torch.zeros(1)
        for (pi, y) in zip(norm_expr, Y):
            log_lik += torch.sum(log_multinomial(y, pi))
        return(log_lik)

    def log_posterior(self, X, Y):
        log_lik = self.log_likelihood(X, Y)
        log_prior = torch.sum(log_normal(self.mu, self.m0, self.s0)) + torch.sum(log_normal(self.beta, self.m0, self.s0))
        log_posterior = log_lik + log_prior
        return(log_posterior)
