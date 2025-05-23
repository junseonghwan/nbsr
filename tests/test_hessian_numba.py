import unittest
import time

import numpy as np
from scipy.special import digamma, polygamma
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

# class TestNBSRHessian(unittest.TestCase):
#     def test_log_lik_gradients(self):
#         print("==============Testing Hessian computation==============")
#         d = 3
#         N = 30
#         J = 100 # Runtime is cubic in J. J=1024 takes about 800 seconds.
#         (Y, X, phi) = generate_data(d, N, J)

#         model = nbm.NegativeBinomialRegressionModel(torch.tensor(X), torch.tensor(Y), 
#                                                     lam=1., shape=3., scale=2.,
#                                                     dispersion = phi, pivot=False)

#         # Compute Hessian using numba.
#         pi = model.predict(model.beta, model.X)[0].data.numpy()
#         s = np.sum(model.Y.data.numpy(), 1)
#         mu = s[:,None] * pi
#         start = time.perf_counter()
#         grad1, hess1 = utils.hessian_nbsr(X, Y, pi, mu, phi, model.pivot)
#         end = time.perf_counter()
#         print("Elapsed with numba1 = {}s".format((end - start)))
        
#         start = time.perf_counter()
#         grad2, hess2 = utils.log_lik_gradients2(X, Y, pi, mu, phi, model.pivot)
#         end = time.perf_counter()
#         print("Elapsed with numba2 = {}s".format((end - start)))
#         dim = J-1 if model.pivot else J
#         grad2 = grad2.sum(axis=0).T[:dim].flatten()
#         self.assertTrue(np.allclose(grad1, grad2))
#         self.assertTrue(np.allclose(hess1, hess2))

# class TestNBSRTrendedHessian(unittest.TestCase):
#     def test_log_lik_gradients_trended(self):
#         print("==============Testing NBSRTrended Hessian computation==============")
#         d = 3
#         N = 30
#         J = 200 # Runtime is cubic in J.
#         (Y, X, phi) = generate_data(d, N, J)

#         disp_model = dm.DispersionModel(torch.tensor(Y))
#         model = nbsrd.NBSRTrended(torch.tensor(X), torch.tensor(Y), disp_model, lam=1., shape=3., scale=2., pivot=False)
#         z = model.log_likelihood2(model.beta)
#         if model.beta.grad is not None:
#             model.beta.grad.zero_()
#         z.backward(retain_graph=True)
#         grad_expected = model.beta.grad.data.numpy()
#         log_lik_grad = model.log_lik_gradient_persample(model.beta).sum(0)
#         grad_actual = log_lik_grad.data.numpy()
#         #self.assertTrue(np.allclose(grad_expected, grad_actual))
#         #print(grad_expected)
#         #print(grad_actual)

#         # Compute gradients and Hessian using numba.
#         b1 = model.disp_model.b1.data.numpy()
#         pi = model.predict(model.beta, model.X)[0].detach()
#         phi = torch.exp(model.disp_model.forward(pi)).detach()
#         s = torch.sum(model.Y, 1)
#         mu = s[:,None] * pi
#         var = mu + phi * (mu ** 2)
#         r = 1.0 / phi
#         p = mu / var
#         a = digamma(model.Y + r) - digamma(r) + torch.log(p)
#         r_np = r.detach().numpy()
#         trigamma = polygamma(1, model.Y.data.numpy() + r_np) - polygamma(1, r_np)
#         start = time.perf_counter()
#         hess_realized = utils.hessian_trended_nbsr(X, Y, pi.numpy(), p.numpy(), r.numpy(), a.numpy(), trigamma, b1[0], model.pivot)
#         end = time.perf_counter()
#         print("Elapsed with numba = {}s".format((end - start)))
#         #print(grad_realized)
#         #self.assertTrue(np.allclose(grad_expected, grad_realized))

#         # compute Hessian.
#         hess_expected = torch.zeros(log_lik_grad.size(0), log_lik_grad.size(0))
#         start = time.perf_counter()
#         # Compute the gradient for each component of y w.r.t. beta
#         for k in range(log_lik_grad.size(0)):
#             # Zero previous gradient
#             if model.beta.grad is not None:
#                 model.beta.grad.zero_()

#             # Backward on the k-th component of y
#             log_lik_grad[k].backward(retain_graph=True)

#             # Store the gradient
#             hess_expected[k,:] = model.beta.grad
#         end = time.perf_counter()
#         print("Elapsed with torch = {}s".format((end - start)))

#         #print(hess_expected)
#         #print(hess_realized)

#         self.assertTrue(np.allclose(hess_expected, hess_realized))

