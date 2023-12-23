import torch

import os
import math
import numpy as np
import pandas as pd
from scipy.stats import invgamma
import time

from nbsr.distributions import log_negbinomial, log_gamma, log_normal, log_invgamma, softplus_inv
from nbsr.utils import torch_rbf

# Model the relationship between the dispersion and mean.
# Estimates Gaussian RBF parameters given dispersion (phi) and mean (mu).
# \phi ~ LogNormal(f(\mu), \tau)
class GaussianRBF(torch.nn.Module):
    def __init__(self, min, max, knot_count = 10):
        super().__init__()
        self.softplus = torch.nn.Softplus()
        self.a = torch.nn.Parameter(torch.randn(knot_count, dtype=torch.float64), requires_grad=True)
        delta = (max - min) / knot_count
        self.b = 1. / (2*(delta**2))
        self.c = torch.arange(0, max, step = delta)

    def evaluate(self, mu):
        phi = torch_rbf(torch.log(mu), self.softplus(self.a), self.b, self.c)
        assert(torch.sum(phi < 0) == 0)
        return(phi)
