import torch

import os
import math
import numpy as np
import pandas as pd
from scipy.stats import invgamma
import time

from nbsr.distributions import log_negbinomial, log_gamma, log_normal, log_invgamma, softplus_inv

class NegativeBinomialRegressionModel(torch.nn.Module):
    def __init__(self, X, Y, dispersion=None, pivot=False):
        super(NegativeBinomialRegressionModel, self).__init__()
        # Assume X is a pandas dataframe.
        assert(isinstance(X, pd.DataFrame))
        assert(isinstance(Y, pd.DataFrame))
        self.X_df = X
        self.Y_df = Y
        self.XX = torch.tensor(pd.get_dummies(X, drop_first=True, dtype=int).to_numpy(), dtype=torch.float64)
        self.Y = torch.tensor(Y.transpose().to_numpy(), dtype=torch.float64)
        self.pivot = pivot
        self.softplus = torch.nn.Softplus()
        self.sample_count = self.Y.shape[0]
        self.covariate_count = self.XX.shape[1] + 1 # +1 for the intercept term.
        self.rna_count = self.Y.shape[1]
        self.converged = False
        self.X = torch.cat([torch.ones(self.sample_count, 1), self.XX], dim = 1)
        print("RNA count:", self.rna_count)
        print("Sample count:", self.sample_count)
        print("Covariate count:", self.covariate_count)

        # The parameters we adjust during training.
        self.dim = self.rna_count - 1 if pivot else self.rna_count
        self.beta = torch.nn.Parameter(torch.randn(self.covariate_count * self.dim, dtype=torch.float64), requires_grad=True)
        if dispersion is None:
            self.phi = torch.nn.Parameter(torch.randn(self.rna_count, dtype=torch.float64), requires_grad=True)
        else:
            self.phi = softplus_inv(torch.tensor(dispersion + 1e-9, requires_grad=False))
        self.psi = torch.nn.Parameter(torch.randn(self.covariate_count, dtype=torch.float64), requires_grad=True)

        self.I = None

    def compute_observed_information(self, recompute=True):
        if self.I is not None and not recompute:
            return self.I
        
        log_post_grad = self.log_posterior_gradient(self.beta)
        gradient_matrix = torch.zeros(log_post_grad.size(0), self.beta.size(0))
        # Compute the gradient for each component of log_post_grad w.r.t. beta
        for k in range(log_post_grad.size(0)):
            # Zero previous gradient
            if self.beta.grad is not None:
                self.beta.grad.zero_()

            # Backward on the k-th component of y
            log_post_grad[k].backward(retain_graph=True)

            # Store the gradient
            gradient_matrix[k,:] = self.beta.grad

        self.I = -gradient_matrix
        return self.I

    def specify_beta_prior(self, lam, beta_var_shape, beta_var_scale):
        self.lam = torch.tensor(lam, requires_grad=False)
        self.beta_var_shape = torch.tensor(beta_var_shape, requires_grad=False)
        self.beta_var_scale = torch.tensor(beta_var_scale, requires_grad=False)
        sd = np.sqrt(invgamma.rvs(a=self.beta_var_shape, scale=self.beta_var_scale, size=self.covariate_count))
        print("Initial sd:", sd)
        self.psi = torch.nn.Parameter(softplus_inv(torch.tensor(sd)), requires_grad=True)
        print("Initial psi:", self.psi)
        #self.psi = softplus_inv(torch.tensor(invgamma.rvs(self.beta_var_shape, self.beta_var_scale, size=self.covariate_count+1)))

    def log_likelihood(self, beta):
            """
            Computes the log-likelihood of the negative binomial model.

            Args:
                beta (torch.Tensor): A tensor of shape (covariate_count * dim, 1) containing the model parameters.

            Returns:
                torch.Tensor: A tensor of shape (1,) containing the log-likelihood of the model.
            """
            # reshape beta:
            beta_ = torch.reshape(beta, (self.covariate_count, self.dim))
            log_unnorm_exp = torch.matmul(self.X, beta_)
            if self.pivot:
                log_unnorm_exp = torch.column_stack((log_unnorm_exp, torch.zeros(self.sample_count)))
            norm = torch.logsumexp(log_unnorm_exp, 1)
            norm_expr = torch.exp(log_unnorm_exp - norm[:,None])
            
            s = torch.sum(self.Y, dim=1)  # Summing over rows
            log_lik_vals = log_negbinomial(self.Y, s[:, None] * norm_expr, self.softplus(self.phi))
            log_lik = torch.sum(log_lik_vals)  # Sum all values

            return(log_lik)
    
    def log_beta_prior(self, beta):
        beta_ = beta.reshape(self.dim, self.covariate_count)
        sd = self.softplus(self.psi)
        log_prior1 = torch.sum(log_normal(beta_, torch.zeros_like(sd), sd/self.lam))
        return(log_prior1)

    def log_posterior(self, beta):
        """
        Computes the log posterior probability of the negative binomial regression model
        with normal prior on the regression coefficients.

        Args:
            beta (torch.Tensor): A tensor of shape (dim * covariate_count,) representing the
                flattened regression coefficients.

        Returns:
            torch.Tensor: A scalar tensor representing the log posterior probability.
        """
        log_lik = self.log_likelihood(beta)
        # normal prior on beta -- 0 mean and var given as input from glmGamPoi.
        sd = self.softplus(self.psi)
        beta_ = beta.reshape(self.dim, self.covariate_count)
        log_prior1 = self.log_beta_prior(beta)
        log_prior2 = torch.sum(log_invgamma(sd**2, self.beta_var_shape, self.beta_var_scale))
        log_posterior = log_lik + log_prior1 + log_prior2
        return(log_posterior)

    def log_lik_gradient_persample(self, beta):
        """
        Computes the gradient of the log-likelihood function with respect to the model parameters for each sample.

        Args:
            beta (torch.Tensor): A tensor of shape (covariate_count * dim,) containing the model parameters.

        Returns:
            torch.Tensor: A tensor of shape (sample_count, covariate_count * dim) containing the gradient of the log-likelihood function with respect to the model parameters for each sample.
        """        
        beta_ = torch.reshape(beta, (self.covariate_count, self.dim))
        dispersion = self.softplus(self.phi)
        J = self.rna_count
        log_unnorm_exp = torch.matmul(self.X, beta_)
        if self.pivot:
            log_unnorm_exp = torch.column_stack((log_unnorm_exp, torch.zeros(self.sample_count)))
        norm = torch.logsumexp(log_unnorm_exp, 1)
        norm_expr = torch.exp(log_unnorm_exp - norm[:,None])

        grad = torch.zeros(self.sample_count, self.dim * self.covariate_count)
        for idx, (pi, x, y) in enumerate(zip(norm_expr, self.X, self.Y)):
            s = torch.sum(y)
            mean = s * pi
            sigma2 = mean + dispersion * (mean ** 2)
            p = mean / sigma2
            r = 1 / dispersion
            A = torch.eye(J) - pi.repeat((J, 1))
            A = A.transpose(0, 1)
            temp0 = (mean + 2 * dispersion * (mean ** 2))/sigma2
            temp1 = 1 - temp0
            temp2 = 2 - temp0
            temp = (r * temp1 + y * temp2)
            ret1 = x.repeat((J, 1)).transpose(0,1) * temp
            ret2 = ret1.unsqueeze(1).repeat(1, J, 1)
            ret3 = ret2 * A
            grad[idx,:] = torch.sum(ret3, 2).flatten()
        return grad
    
    def log_lik_gradient_persample2(self, beta):
        """
        Computes the gradient of the log-likelihood function with respect to the model parameters for each sample.
        Uses tensor operations instead of for loops.
        """
        beta = self.beta
        beta_ = torch.reshape(beta, (self.covariate_count, self.dim))
        dispersion = self.softplus(self.phi)
        J = self.rna_count
        log_unnorm_exp = torch.matmul(self.X, beta_)
        if self.pivot:
            log_unnorm_exp = torch.column_stack((log_unnorm_exp, torch.zeros(self.sample_count)))
        norm = torch.logsumexp(log_unnorm_exp, 1)
        norm_expr = torch.exp(log_unnorm_exp - norm[:, None])

        s = torch.sum(self.Y, dim=1)
        mean = s[:, None] * norm_expr
        sigma2 = mean + dispersion * (mean ** 2)
        r = 1 / dispersion
        D = dispersion * (mean ** 2) / sigma2

        identity_matrix = torch.eye(J, device=norm_expr.device).unsqueeze(0).repeat(self.sample_count, 1, 1)
        # Pi is NxJxJ with elements Pi_{n,j,k} = 1[j = k] - norm_expr_{n,k}.
        # Transpose is necessary to get norm_expr_{n,k}.
        Pi = (identity_matrix - norm_expr.unsqueeze(2)).transpose(1,2)
        XPi = torch.einsum('np,njk->njkp', self.X, Pi)

        rD = r * D
        yDD = torch.einsum('nj,nj->nj', self.Y, (1 - D))
        c0 = (-rD + yDD)
        ret = torch.einsum('nj,njkp->njkp', c0, XPi)
        grad = torch.sum(ret.transpose(2,3), 1).reshape(self.sample_count, J * self.covariate_count)
        return grad

    def log_lik_gradient(self, beta, tensorized=True):
        """
        Computes the gradient of the log-likelihood function with respect to the model parameters.

        Args:
            beta (torch.Tensor): The model parameters.
            tensorized (bool): Whether to use the tensorized version of the gradient computation.

        Returns:
            torch.Tensor: The gradient of the log-likelihood function with respect to the model parameters.
        """
        if tensorized:
            return torch.sum(self.log_lik_gradient_persample2(beta), 0)
        else:
            return torch.sum(self.log_lik_gradient_persample(beta), 0)

    def log_beta_prior_gradient(self, beta):
        beta_ = beta.reshape(self.dim, self.covariate_count)
        sd = self.softplus(self.psi)
        log_prior_grad = -(self.lam**2) * beta_ / sd**2
        return(log_prior_grad.flatten())

    def log_posterior_gradient(self, beta, tensorized=True):
        """
        Computes the gradient of the log posterior distribution with respect to the model parameters.

        Args:
            beta (torch.Tensor): A tensor of shape (dim * covariate_count,) representing the model parameters.
            tensorized (bool): Whether to use the tensorized version of the gradient computation.

        Returns:
            torch.Tensor: A tensor of the same shape as `beta` representing the gradient of the log posterior distribution.
        """
        log_prior_grad = self.log_beta_prior_gradient(beta)
        log_lik_grad = self.log_lik_gradient(beta, tensorized)
        return log_lik_grad + log_prior_grad
    
    def log_lik_hessian_persample(self, beta):
        beta_ = torch.reshape(beta, (self.covariate_count, self.dim))
        dispersion = self.softplus(self.phi)
        J = self.dim
        P = self.covariate_count
        log_unnorm_exp = torch.matmul(self.X, beta_)
        if self.pivot:
            log_unnorm_exp = torch.column_stack((log_unnorm_exp, torch.zeros(self.sample_count)))
        norm = torch.logsumexp(log_unnorm_exp, 1)
        norm_expr = torch.exp(log_unnorm_exp - norm[:,None])
        total_dim = J*P
        r = 1 / dispersion

        hessian = torch.zeros(self.sample_count, total_dim, total_dim)
        for idx, (pi, x, y) in enumerate(zip(norm_expr, self.X, self.Y)):
            s = torch.sum(y)
            mean = s * pi
            sigma2 = mean + dispersion * (mean ** 2)
            Di = dispersion * (mean ** 2) / sigma2

            rD = r * Di
            yD = y * (1-Di)
            rDD = r * Di * (1-Di)
            yDD = y * Di * (1-Di)

            A = pi.repeat((J, 1))
            B = torch.eye(J) - A

            AA = A.unsqueeze(2) * x
            BB = B.unsqueeze(2) * x
            prod1 = torch.einsum('jkd,kle->jkdle', AA, BB)
            prod1 = prod1.transpose(3, 4).transpose(1, 2)
            prod1 = prod1.reshape(J, J*P, J*P)

            prod2 = torch.einsum('jkd,jle->jkdle', BB, BB)
            prod2 = prod2.transpose(3, 4).transpose(1, 2)
            prod2 = prod2.reshape(J, J*P, J*P)

            C1_ = torch.einsum('j,jkl->jkl', (rD - yD), prod1)
            C2_ = torch.einsum('j,jkl->jkl', -(rDD + yDD) , prod2)

            hessian[idx,:,:] = torch.sum(C1_ + C2_, 0)
        return hessian

    def log_posterior_hessian(self, beta):
        sd = self.softplus(self.psi).repeat(self.dim)
        return torch.sum(self.log_lik_hessian_persample(beta), 0) + (1/sd**2) * torch.eye(self.dim * self.covariate_count)

    def predict(self, beta, X):
        beta_ = torch.reshape(beta, (self.covariate_count, self.dim))
        log_unnorm_exp = torch.matmul(X, beta_)
        if self.pivot:
            log_unnorm_exp = torch.column_stack((log_unnorm_exp, torch.zeros(self.sample_count)))
        norm = torch.logsumexp(log_unnorm_exp, 1)
        norm_expr = torch.exp(log_unnorm_exp - norm[:,None])
        pi = norm_expr
        return(pi, log_unnorm_exp)



