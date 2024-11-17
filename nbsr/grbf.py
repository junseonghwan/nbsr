import torch

import os
import math
import numpy as np
import pandas as pd
from scipy.stats import invgamma
import time

from nbsr.distributions import log_normal, log_lognormal, softplus_inv, softplus

# x is K or Kx1.
def torch_gaussian_rbf(x, center, scale):
    if not torch.is_tensor(x):
        x = torch.tensor(x, dtype=torch.float32)

    if x.dim() == 1:
        x = x.unsqueeze(-1)  # Add an extra dimension for broadcasting: K x 1

    # center is a vector of length L. Reshape for broadcasting
    center = center.reshape(1, -1)  # 1 x 1 x L
    # scale is either a vector of length L or a scalar.
    scale = scale.reshape(1, -1)  # 1 x 1 x L or 1 x 1 x 1
    
    # Broadcasting will align dimensions automatically.
    val = (x - center) / scale
    val = torch.exp(-0.5 * val**2)
    return(val)

# mu: not in log scale (will take the log in the function)
# centers: in log scale.
def evaluate_mean(mu, beta, log_centers=None, scales=None):
    # mu is a tensor of dimension K (or Kx1).
    log_mu = torch.log(mu)
    if log_mu.dim() == 1:
        # Unsqueeze to add a dimension, making it 2D
        # Using 0 to add a dimension at the beginning; use 1 to add at the end
        log_mu = log_mu.unsqueeze(-1)
    K = log_mu.shape[0]
    if log_centers is not None:
        # g is K x L
        g = torch_gaussian_rbf(log_mu, log_centers, scales)
        # concatenate intercept and log_mu
        X = torch.concat([torch.ones(K, 1), log_mu, g], dim = 1)
    else:
        X = torch.concat([torch.ones(K, 1), log_mu], dim = 1)
    f = torch.matmul(X, beta)
    return(f, X)

# Fit GRBF on log phi ~ log mu_bar
class GaussianRBF(torch.nn.Module):
    def __init__(self, min_value, max_value, sd=None, knot_count = 10, width=1.2):
        super().__init__()
        # the number of parameters is knot_count + 2  for intercept and slope.
        self.beta = torch.nn.Parameter(torch.randn(knot_count + 2, dtype=torch.float64), requires_grad=True)
        if sd is None:
            self.psi = torch.nn.Parameter(torch.randn(1, dtype=torch.float64), requires_grad=True)
        else:
            self.psi = softplus_inv(torch.tensor(sd, requires_grad=False))
        self.width = width
        self.knot_count = knot_count
        if knot_count > 0:
            self.centers = torch.linspace(min_value, max_value, knot_count)
            delta = torch.diff(self.centers)
            self.h = torch.tensor([delta[0] * self.width]).repeat(knot_count)
        else:
            self.centers = None
            self.h = None

    def forward(self, pi):
        return evaluate_mean(pi, self.beta, self.centers, self.h)[0]

    # phi ~ LogNormal(f(mean_expr), sd).
    def log_density(self, phi, pi):
        assert(phi.shape == pi.shape)
        # log_phi: log of dispersion values of dimension Kx1.
        # mean_expr: mean expression values of dimension Kx1.
        f_mean_expr, _ = evaluate_mean(pi, self.beta, self.centers, self.h)
        sd = softplus(self.psi)
        log_lik = log_lognormal(phi, f_mean_expr, sd)
        log_prior0 = log_normal(self.beta, torch.tensor(0), torch.tensor(1))
        return(log_lik.sum() + log_prior0.sum())

    def sample(self, pi):
        f_mean, _ = evaluate_mean(pi, self.beta, self.centers, self.h)
        z = torch.randn(pi.shape)
        sd = softplus(self.psi)
        return(torch.exp(f_mean + sd*z))
