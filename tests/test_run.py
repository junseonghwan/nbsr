import unittest

import numpy as np
import pandas as pd
import torch

import nbsr.negbinomial_model as nbm
import nbsr.nbsr_dispersion as nbsrd
import nbsr.dispersion as dm
from nbsr.utils import *
from tests.test_gradients import generate_data
from nbsr.distributions import log_invgamma

class TestNBSRGradients(unittest.TestCase):

    def test_MCEM(self):
        # d = 3
        # N = 20
        # J = 5
        # (Y, X, phi) = generate_data(d, N, J)
        
        # tensorY = torch.tensor(Y)

        counts_pd = pd.read_csv("/Users/seonghwanjun/Dropbox/seong/miRNA/output/validation2/t_lymphocyte_cd4/rep1/Y.csv")
        coldata_pd = pd.read_csv("/Users/seonghwanjun/Dropbox/seong/miRNA/output/validation2/t_lymphocyte_cd4/rep1/X.csv", na_filter=False, skipinitialspace=True)
        tensorY = torch.tensor(counts_pd.transpose().to_numpy(), dtype=torch.float64)
        X, x_map = construct_tensor_from_coldata(coldata_pd, ["trt"], counts_pd.shape[1])


        disp_model = dm.DispersionModel(tensorY)
        model = nbsrd.NBSRDispersion(X, tensorY, disp_model=disp_model, pivot=True)
        model.specify_beta_prior(1., 3., 2.)
        optimizer = torch.optim.Adam(model.parameters(), lr = 0.05)

        for em_iter in range(1):
            log_weights, phi_samples = model.log_likelihood_samples(10)
            phi_samples = phi_samples.detach()
            log_weights_sum = torch.logsumexp(log_weights.detach(), dim=0, keepdim=True)  # Shape: 1 x N x K
            normalized_log_weights = log_weights - log_weights_sum  # Shape: M x N x K
            normalized_weights = torch.exp(normalized_log_weights).data
            #err = torch.sum(normalized_weights.sum(0) - 1., dtype=torch.float32)
            #self.assertTrue(torch.allclose(err, torch.tensor([0.]), atol=1e-6))

            for iter in range(100):
                # Construct the objective function to optimize.
                pi,_ = model.predict(model.beta, model.X)
                log_obs_lik = torch.stack([model.log_likelihood2(model.beta, phi_m) for phi_m in phi_samples])
                log_phi_lik = torch.stack([model.disp_model.log_density(phi_m, pi) for phi_m in phi_samples])
                # Compute the log of prior.
                sd = model.softplus(model.psi)
                # normal prior on beta -- 0 mean and sd = softplus(psi).
                log_beta_prior = model.log_beta_prior(model.beta)
                # inv gamma prior on var = sd^2 -- hyper parameters specified to the model.
                log_var_prior = torch.sum(log_invgamma(sd**2, model.beta_var_shape, model.beta_var_scale))
                log_dispersion_prior = model.disp_model.log_prior()
                log_prior = log_beta_prior + log_var_prior + log_dispersion_prior

                objective_q = (normalized_weights * (log_obs_lik + log_phi_lik)).sum() + log_prior

                # Optimize the posterior
                loss = -objective_q
                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()
                            
            print(loss.detach())

        
        pi,_ = model.predict(model.beta, model.X)
        log_pi = torch.log(pi.detach())
        log_phi_mean = model.disp_model.forward(log_pi)
        #print(torch.exp(log_phi_mean).mean(0))
        print(torch.exp(log_phi_mean)[0:5,:].mean(0))
        print(torch.exp(log_phi_mean)[5:10,:].mean(0))
