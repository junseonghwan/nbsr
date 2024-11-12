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

        output_path = "/Users/seonghwanjun/Dropbox/seong/miRNA/output/validation2/t_lymphocyte_cd4/rep1/"

        counts_pd = pd.read_csv("/Users/seonghwanjun/Dropbox/seong/miRNA/output/validation2/t_lymphocyte_cd4/rep1/Y.csv")
        coldata_pd = pd.read_csv("/Users/seonghwanjun/Dropbox/seong/miRNA/output/validation2/t_lymphocyte_cd4/rep1/X.csv", na_filter=False, skipinitialspace=True)
        mean_expr = pd.read_csv(os.path.join(output_path, "deseq2_mu.csv"))
        mean_expr = torch.tensor(mean_expr.transpose().to_numpy())
        tensorY = torch.tensor(counts_pd.transpose().to_numpy(), dtype=torch.float64)
        X, x_map = construct_tensor_from_coldata(coldata_pd, ["trt"], counts_pd.shape[1])
        row_sums = mean_expr.sum(dim=1, keepdim=True)
        pi_hat = mean_expr / row_sums

        # Fomulate GRBF prior by fitting \phi_{ij}
        log_pi = torch.log(pi_hat)
        min_val = torch.min(log_pi)
        max_val = torch.max(log_pi)
        K = tensorY.shape[1]
        log_R = torch.log(tensorY.sum(1))
        grbf_disp_model = dm.DispersionGRBF(min_val, max_val, Z=log_R.unsqueeze(1), sd_dimension=K)
        model = nbsrd.NBSRDispersion(X, tensorY, disp_model=grbf_disp_model, pivot=False)
        model.specify_beta_prior(1., 3., 2.)
        for name, param in grbf_disp_model.named_parameters():
            print(name)
        optimizer = torch.optim.Adam(grbf_disp_model.parameters(), lr = 0.01)
        # We optimize log likelihood + GRBF prior. We don't optimize kappa here.
        for i in range(3000):
            phi_hat = torch.exp(grbf_disp_model.forward(pi_hat))
            log_lik = model.log_obs_likelihood(pi_hat, phi_hat).sum()
            log_prior = grbf_disp_model.log_prior().sum()
            loss = -(log_lik + log_prior)
            optimizer.zero_grad()
            loss.backward(retain_graph=False)
            optimizer.step()
            if i % 100 == 0:
                print(f"Iter {i}: loss {loss.detach().numpy()}")
                #print(f"Kappa: {grbf_disp_model.get_sd().detach().numpy()}")

        phi = torch.exp(grbf_disp_model.forward(pi_hat))
        np.savetxt(os.path.join(output_path, "nbsr_dispersion_prior.csv"), phi.data.numpy().transpose(), delimiter=',')

        # Formulate prior on kappa.
        
        # Use MC-EM NBSR Dispersion with the GRBF prior (do not update GRBF parameters).
        model_params = []
        for name, param in model.named_parameters():
            print(name)
            if 'disp_model.beta' in name or 'phi' in name:
                continue
            model_params.append(param)

        # We will optimize kappa parameter to get a sense of variance around dispersion.
        optimizer = torch.optim.Adam(model_params, lr=0.01)
        mc_samples = 20
        for em_iter in range(50):
            log_weights, phi_samples = model.log_likelihood_samples(mc_samples)
            phi_samples = phi_samples.detach()
            log_weights = log_weights.detach()
            log_weights_sum = torch.logsumexp(log_weights, dim=0, keepdim=True)  # Shape: 1 x N x K
            normalized_log_weights = log_weights - log_weights_sum  # Shape: M x N x K
            normalized_weights = torch.exp(normalized_log_weights).data

            for iter in range(100):
                # Construct the objective function to optimize.
                pi,_ = model.predict(model.beta, model.X)
                log_obs_lik = torch.stack([model.log_obs_likelihood2(model.beta, phi_m) for phi_m in phi_samples])
                log_phi_lik = torch.stack([model.disp_model.log_density(phi_m, pi) for phi_m in phi_samples])
                # Compute the log prior.
                log_nbsr_prior = model.log_prior(model.beta)
                log_dispersion_prior = model.disp_model.log_prior()
                log_prior = log_nbsr_prior + log_dispersion_prior

                objective_q = (normalized_weights * (log_obs_lik + log_phi_lik)).sum() + log_prior

                # Optimize the posterior
                loss = -objective_q
                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()

            if em_iter % 1 == 0:
                pi, _ = model.predict(model.beta, X)
                phi = torch.exp(model.disp_model.forward(pi))
                mu = model.s[:,None] * pi
                err = torch.mean(torch.abs(mu - tensorY))
                print(f"Mean absolute error: {err.detach().numpy()}")
                print(f"Current kappa: {model.disp_model.get_sd().detach().numpy()}")
                curr_log_posterior = model.log_posterior(model.beta)
                print(f"EM iter {em_iter}. Log posterior: {curr_log_posterior}")
                if torch.isnan(curr_log_posterior).any():
                    #import pdb; pdb.set_trace()
                    print("Break")

        # Output pi and phi.
        pi, _ = model.predict(model.beta, X)
        phi = torch.exp(model.disp_model.forward(pi))
        mu = model.s[:,None] * pi
        err = torch.mean(torch.abs(mu - tensorY))
        print(f"Mean absolute error: {err.detach().numpy()}")
        np.savetxt(os.path.join(output_path, "nbsr_pi.csv"), pi.data.numpy().transpose(), delimiter=',')
        np.savetxt(os.path.join(output_path, "nbsr_dispersion.csv"), phi.data.numpy().transpose(), delimiter=',')
        np.savetxt(os.path.join(output_path, "kappa.csv"), grbf_disp_model.get_sd().detach().numpy(), delimiter=',')


