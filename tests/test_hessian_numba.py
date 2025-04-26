import unittest
import time

import numpy as np
import torch

import nbsr.negbinomial_model as nbm
import nbsr.nbsr_dispersion as nbsrd
import nbsr.dispersion as dm
import nbsr.utils as utils

def setup_module(module):
    print("Testing Hessian computation speed.")

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

class TestNBSRHessian(unittest.TestCase):
    def test_log_lik_hessian(self):
        print("==============Testing Hessian computation==============")
        d = 3
        N = 30
        J = 128 # Runtime is cubic in J.
        (Y, X, phi) = generate_data(d, N, J)

        model = nbm.NegativeBinomialRegressionModel(torch.tensor(X), torch.tensor(Y), dispersion = phi, pivot=False)

        # Compute Hessian using numba.
        pi = model.predict(model.beta, model.X)[0].data.numpy()
        s = np.sum(model.Y.data.numpy(), 1)
        mu = s[:,None] * pi
        start = time.perf_counter()
        grad1, hess1 = utils.log_lik_gradients(X, Y, pi, mu, phi, model.pivot)
        end = time.perf_counter()
        print("Elapsed with numba1 = {}s".format((end - start)))
        
        start = time.perf_counter()
        hess2 = utils.log_lik_gradients2(X, Y, pi, mu, phi, model.pivot)
        end = time.perf_counter()
        print("Elapsed with numba2 = {}s".format((end - start)))
        #dim = J-1 if model.pivot else J
        #grad2 = grad_realized2.sum(axis=0).T[:dim].flatten()
        #print(grad1)
        #print(grad2)
        #self.assertTrue(np.allclose(grad1, grad2))
        #print(hess1)
        #print(hess2)
        self.assertTrue(np.allclose(hess1, hess2))

        

