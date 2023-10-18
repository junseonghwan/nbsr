import unittest

import numpy as np
import pandas as pd
import torch

import nbsr.negbinomial_model as nbm

def generate_data(d, N, J):
    # Generate data for testing.
    softplus = lambda x: np.log1p(np.exp(x))

    phi = softplus(np.random.randn(J))
    beta = np.random.randn(J * d)
    beta_reshape = beta.reshape(d, J)
    Y = np.zeros((N, J))
    X = np.zeros((N, d))
    s = np.random.poisson(10000, N)
    for i in range(N):
        x = np.random.randn(d)
        exp_xbeta = np.exp(np.matmul(x, beta_reshape))
        pi = exp_xbeta/np.sum(exp_xbeta)
        #mu = s[i] * pi
        #sigma2 = mu + phi * (mu ** 2)
        #p = mu / sigma2 # equivalent to r / (mu + r).
        #r = 1 / phi
        #y = torch.tensor(np.random.negative_binomial(r.data.numpy(), p.data.numpy()))
        y = np.random.multinomial(s[i], pi)
        Y[i,:] = y
        X[i,:] = x

    return(Y, X, phi)

class TestNegBinModel(unittest.TestCase):

    def test_log_lik_gradient(self):
        d = 3
        N = 20
        J = 5
        (Y, X, phi) = generate_data(d, N, J)
    
        Y_df = pd.DataFrame(Y.transpose(), dtype="int32")
        X_df = pd.DataFrame(X)
        model = nbm.NegativeBinomialRegressionModel(X_df, Y_df, dispersion = phi, pivot=False)
        z = model.log_likelihood(model.beta)
        if model.beta.grad is not None:
            model.beta.grad.zero_()
        z.backward(retain_graph=True)
        grad_expected = model.beta.grad.data.numpy()
        grad_actual = model.log_lik_gradient(model.beta, tensorized=False).data.numpy()
        self.assertTrue(np.allclose(grad_expected, grad_actual))
        print(grad_expected)
        print(grad_actual)

    def test_log_lik_gradient_tensorized(self):
        d = 3
        N = 20
        J = 5
        (Y, X, phi) = generate_data(d, N, J)
    
        Y_df = pd.DataFrame(Y.transpose(), dtype="int32")
        X_df = pd.DataFrame(X)
        model = nbm.NegativeBinomialRegressionModel(X_df, Y_df, dispersion = phi, pivot=False)
        grad_expected = model.log_lik_gradient(model.beta, tensorized=False).data.numpy()
        grad_actual = model.log_lik_gradient(model.beta, tensorized=True).data.numpy()
        print(grad_expected)
        print(grad_actual) 
        self.assertTrue(np.allclose(grad_expected, grad_actual))

    def test_log_beta_prior_gradient(self):
        d = 3
        N = 20
        J = 5
        (Y, X, phi) = generate_data(d, N, J)

        print("Test gradient of log prior over beta...")
        Y_df = pd.DataFrame(Y.transpose(), dtype="int32")
        X_df = pd.DataFrame(X)
        model = nbm.NegativeBinomialRegressionModel(X_df, Y_df, dispersion = phi, pivot=False)
        model.specify_beta_prior(1, 3, 2)
        z = model.log_beta_prior(model.beta)
        if model.beta.grad is not None:
            model.beta.grad.zero_()
        z.backward(retain_graph=True)
        grad_expected = model.beta.grad.data.numpy()
        grad_actual = model.log_beta_prior_gradient(model.beta).data.numpy()
        print(grad_expected)
        print(grad_actual)
        self.assertTrue(np.allclose(grad_expected, grad_actual))

    def test_log_posterior_gradient(self):
        d = 3
        N = 20
        J = 5
        (Y, X, phi) = generate_data(d, N, J)

        print("Test log posterior gradient")
        Y_df = pd.DataFrame(Y.transpose(), dtype="int32")
        X_df = pd.DataFrame(X)
        model = nbm.NegativeBinomialRegressionModel(X_df, Y_df, dispersion = phi, pivot=False)
        model.specify_beta_prior(1, 3, 2)
        z = model.log_posterior(model.beta)
        if model.beta.grad is not None:
            model.beta.grad.zero_()
        z.backward(retain_graph=True)
        grad_expected = model.beta.grad.data.numpy()
        grad_actual = model.log_posterior_gradient(model.beta, tensorized=False).data.numpy()

        # log_lik_grad = model.log_lik_gradient(model.beta)
        # log_prior_grad = model.log_beta_prior_gradient(model.beta)
        # print(log_lik_grad)
        # print(log_prior_grad)

        print(grad_expected)
        print(grad_actual)
        self.assertTrue(np.allclose(grad_expected, grad_actual))

    def test_log_lik_hessian(self):
        d = 3
        N = 20
        J = 5
        (Y, X, phi) = generate_data(d, N, J)
    
        Y_df = pd.DataFrame(Y.transpose(), dtype="int32")
        X_df = pd.DataFrame(X)
        model = nbm.NegativeBinomialRegressionModel(X_df, Y_df, dispersion = phi, pivot=False)

        log_lik_grad = model.log_lik_gradient(model.beta)
        hess_expected = torch.zeros(log_lik_grad.size(0), model.beta.size(0))
        # Compute the gradient for each component of y w.r.t. beta
        for k in range(log_lik_grad.size(0)):
            # Zero previous gradient
            if model.beta.grad is not None:
                model.beta.grad.zero_()

            # Backward on the k-th component of y
            log_lik_grad[k].backward(retain_graph=True)

            # Store the gradient
            hess_expected[k,:] = model.beta.grad

        hess_realized = torch.sum(model.log_lik_hessian_persample(model.beta),0).data.numpy()
        #print(hess_expected)
        #print(hess_realized)
        print("Testing Hessian computation...")
        self.assertTrue(np.allclose(hess_expected, hess_realized))

if __name__ == '__main__':
    unittest.main()