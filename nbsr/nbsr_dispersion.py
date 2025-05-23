import torch

import numpy as np
from scipy.special import digamma, polygamma

from nbsr.distributions import log_negbinomial, log_invgamma
from nbsr.negbinomial_model import NegativeBinomialRegressionModel
from nbsr.utils import hessian_trended_nbsr

# This model extends the basic NBSR model with trended dispersion given by disp_model.forward(pi).
class NBSRTrended(NegativeBinomialRegressionModel):

    def __init__(self, X, Y, disp_model, lam, shape, scale, pivot=False):
        super().__init__(X, Y, lam=lam, shape=shape, scale=scale, dispersion_prior=disp_model, dispersion=None, pivot=pivot)
        self.phi = None

    def log_likelihood(self, pi, phi):
        # Define log_liklihood that uses the new architecture.
        mu = self.s[:,None] * pi
        log_lik_vals = log_negbinomial(self.Y, mu, phi)
        return log_lik_vals.sum()

    def log_posterior(self, beta):
        pi,_ = self.predict(beta, self.X)
        phi = torch.exp(self.disp_model.forward(pi))

        # Compute the log likelihood of Y
        log_lik = self.log_likelihood(pi, phi)
        # Compute the log of prior.
        sd = self.softplus(self.psi)
        # normal prior on beta -- 0 mean and sd = softplus(psi).
        log_beta_prior = self.log_beta_prior(beta)
        # inv gamma prior on var = sd^2 -- hyper parameters specified to the model.
        log_var_prior = torch.sum(log_invgamma(sd**2, self.beta_var_shape, self.beta_var_scale))
        log_dispersion_prior = self.disp_model.log_prior()
        log_posterior = log_lik + log_beta_prior + log_var_prior + log_dispersion_prior
        return log_posterior
    
    def forward(self, beta):
        return self.log_posterior(beta)

    def log_likelihood2(self, beta):
        # Define log_liklihood that uses the new architecture.
        pi,_ = self.predict(beta, self.X)
        mu = self.s[:, None] * pi
        phi = torch.exp(self.disp_model.forward(pi))
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
        pi,_ = self.predict(beta, self.X) # N x K
        phi = torch.exp(self.disp_model.forward(pi)) # N x K
        #J = self.rna_count-1 if self.pivot else self.rna_count
        I_K = torch.eye(self.rna_count, device=beta.device) # K x K
        b1_term = (1 + self.disp_model.b1) # scalar

        # grad[i,k] = \sum_j \nabla_k \log P(Y_{ij}).
        grad = torch.zeros(self.sample_count, self.dim * self.covariate_count, device=beta.device)
        for i in range(self.sample_count):
        #for idx, (pi_i, phi_i, x, y) in enumerate(zip(pi, phi, self.X, self.Y)):
            pi_i = pi[i]
            phi_i = phi[i]
            x = self.X[i]
            y = self.Y[i]
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
            grad[i,:] = grad_idx.transpose(0,1).flatten()
        return grad

    def log_posterior_gradient(self, beta):
        """
        Computes the gradient of the log posterior distribution with respect to the model parameters.

        Args:
            beta (torch.Tensor): A tensor of shape (dim * covariate_count,) representing the model parameters.

        Returns:
            torch.Tensor: A tensor of the same shape as `beta` representing the gradient of the log posterior distribution.
        """
        self.to_device(beta.device)
        log_prior_grad = self.log_beta_prior_gradient(beta)
        
        log_lik_grad = self.log_lik_gradient_persample(beta).sum(0)
        return log_lik_grad + log_prior_grad

    def log_likelihood_hessian(self, beta):
        pi = self.predict(beta, self.X)[0].detach()
        phi = torch.exp(self.disp_model.forward(pi))
        mu = self.s[:,None] * pi
        var = mu + phi * (mu ** 2)
        r = 1.0 / phi
        p = mu / var

        b1 = self.disp_model.b1[0].detach()
        aa = digamma(self.Y + r) - digamma(r) + torch.log(p)
        r_np = r.numpy()
        cc = polygamma(1, self.Y.numpy() + r_np) - polygamma(1, r_np)

        return hessian_trended_nbsr(self.X.numpy(), 
                                    self.Y.numpy(), 
                                    pi.numpy(), 
                                    p.numpy(),
                                    r.numpy(),
                                    aa.numpy(), 
                                    cc.numpy(), 
                                    b1.numpy(),
                                    self.pivot)

    def log_posterior_hessian(self, beta):
        sd = self.softplus(self.psi).repeat(self.dim)
        return self.log_likelihood_hessian(beta) + (1/sd**2) * torch.eye(self.dim * self.covariate_count)
