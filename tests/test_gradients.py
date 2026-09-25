import unittest
import time

import numpy as np
import torch

import nbsr.negbinomial_model as nbm
import nbsr.nbsr_dispersion as nbsrd
import nbsr.dispersion as dm
import nbsr.utils as utils
from tests import reference_hessians

def setup_module(module):
    print("Testing gradients and Hessian computation.")

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

def set_distinct_sd(model):
    """Give each covariate a different prior sd on beta.

    psi is initialised so that every covariate has sd = 1. With equal sd the
    layout of beta inside the prior cannot matter, so the prior and its
    gradient must also be tested with distinct sd values.
    """
    with torch.no_grad():
        model.psi.copy_(torch.linspace(-1.0, 2.0, model.covariate_count, dtype=torch.float64))

class TestNBSRGradients(unittest.TestCase):

    def test_log_lik_gradient(self):
        d = 3
        N = 20
        J = 5
        (Y, X, phi) = generate_data(d, N, J)
    
        print("==============Test gradient of log likelihood with no pivot==============")
        #Y_df = pd.DataFrame(Y.transpose(), dtype="int32")
        #X_df = pd.DataFrame(X)
        model = nbm.NegativeBinomialRegressionModel(torch.tensor(X), torch.tensor(Y), 
                                                    lam=1., shape=3., scale=2.,
                                                    dispersion = phi, pivot=False)
        z = model.log_likelihood(model.beta)
        if model.beta.grad is not None:
            model.beta.grad.zero_()
        z.backward(retain_graph=True)
        grad_expected = model.beta.grad.data.numpy()
        start = time.perf_counter()
        grad_actual = model.log_lik_gradient(model.beta).data.numpy()
        end = time.perf_counter()
        print("Elapsed with numpy = {}s".format((end - start)))
        print(grad_expected)
        print(grad_actual)
        self.assertTrue(np.allclose(grad_expected, grad_actual))
        
        # TODO: MARKED FOR REMOVAL.
        # Numba related computation -- not using numba.
        # pi = model.predict(model.beta, model.X)[0].data.numpy()
        # s = np.sum(model.Y.data.numpy(), 1)
        # mu = s[:,None] * pi
        # start = time.perf_counter()
        # grad_actual, _ = utils.hessian_nbsr(X, Y, pi, mu, phi, model.pivot)
        # end = time.perf_counter()
        # print("Elapsed with numba compilation = {}s".format((end - start)))
        # print(grad_actual)
        # self.assertTrue(np.allclose(grad_expected, grad_actual))

        # # Timing should improve on the second call as compiled code will be called.
        # start = time.perf_counter()
        # grad_actual, _ = utils.hessian_nbsr(X, Y, pi, mu, phi, model.pivot)
        # end = time.perf_counter()
        # print("Elapsed with post compilation = {}s".format((end - start)))
        # print(grad_actual)
        # self.assertTrue(np.allclose(grad_expected, grad_actual))

    def test_log_lik_gradient_pivot(self):
        print("==============Test gradient of log likelihood with pivot==============")
        d = 3
        N = 20
        J = 5
        (Y, X, phi) = generate_data(d, N, J)
    
        print(Y.shape)
        print(X.shape)
        #Y_df = pd.DataFrame(Y.transpose(), dtype="int32")
        #X_df = pd.DataFrame(X)
        model = nbm.NegativeBinomialRegressionModel(torch.tensor(X), torch.tensor(Y), 
                                                    lam=1., shape=3., scale=2.,
                                                    dispersion = phi, pivot=True)
        z = model.log_likelihood(model.beta)
        if model.beta.grad is not None:
            model.beta.grad.zero_()
        z.backward(retain_graph=True)
        grad_expected = model.beta.grad.data.numpy()
        grad_actual = model.log_lik_gradient(model.beta).data.numpy()
        print(grad_expected)
        print(grad_actual)
        self.assertTrue(np.allclose(grad_expected, grad_actual))

        # TODO: MARKED FOR REMOVAL.
        # Numba related stuff.
        # pi = model.predict(model.beta, model.X)[0].data.numpy()
        # s = np.sum(model.Y.data.numpy(), 1)
        # mu = s[:,None] * pi
        # start = time.perf_counter()
        # grad_actual, _ = utils.hessian_nbsr(X, Y, pi, mu, phi, model.pivot)
        # end = time.perf_counter()
        # print("Elapsed with numba compilation = {}s".format((end - start)))
        # print(grad_actual)
        # self.assertTrue(np.allclose(grad_expected, grad_actual))

    def test_log_beta_prior_gradient(self):
        d = 3
        N = 20
        J = 5
        (Y, X, phi) = generate_data(d, N, J)

        print("==============Test gradient of log prior over beta==============")
        #Y_df = pd.DataFrame(Y.transpose(), dtype="int32")
        #X_df = pd.DataFrame(X)
        model = nbm.NegativeBinomialRegressionModel(torch.tensor(X), torch.tensor(Y), 
                                                    lam=1., shape=3., scale=2.,
                                                    dispersion = phi, pivot=False)
        set_distinct_sd(model)
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
        print("==============Test log posterior gradient==============")
        d = 3
        N = 20
        J = 5
        (Y, X, phi) = generate_data(d, N, J)

        #Y_df = pd.DataFrame(Y.transpose(), dtype="int32")
        #X_df = pd.DataFrame(X)
        model = nbm.NegativeBinomialRegressionModel(torch.tensor(X), torch.tensor(Y), 
                                                    lam=1., shape=3., scale=2.,
                                                    dispersion = phi, pivot=False)
        set_distinct_sd(model)
        z = model.log_posterior(model.beta)
        if model.beta.grad is not None:
            model.beta.grad.zero_()
        z.backward(retain_graph=True)
        grad_expected = model.beta.grad.data.numpy()
        grad_actual = model.log_posterior_gradient(model.beta).data.numpy()

        # log_lik_grad = model.log_lik_gradient(model.beta)
        # log_prior_grad = model.log_beta_prior_gradient(model.beta)
        # print(log_lik_grad)
        # print(log_prior_grad)

        print(grad_expected)
        print(grad_actual)
        self.assertTrue(np.allclose(grad_expected, grad_actual))

    def test_log_lik_hessian(self):
        print("==============Testing Hessian computation==============")
        d = 3
        N = 20
        J = 8
        (Y, X, phi) = generate_data(d, N, J)
        model = nbm.NegativeBinomialRegressionModel(torch.tensor(X), torch.tensor(Y),
                                                    lam=1., shape=3., scale=2.,
                                                    dispersion = phi, pivot=False)
        beta = model.beta.detach().clone()
        hess_expected = torch.autograd.functional.hessian(model.log_likelihood, beta).numpy()
        # closed form (Kronecker structure) and the loop reference must both match autograd.
        hess_realized = model.log_likelihood_hessian(beta).numpy()
        self.assertTrue(np.allclose(hess_expected, hess_realized))
        pi = model.predict(beta, model.X)[0].numpy()
        mu = np.sum(Y, 1)[:, None] * pi
        hess_loop = reference_hessians.hessian_nbsr(X, Y, pi, mu, phi, model.pivot)
        self.assertTrue(np.allclose(hess_expected, hess_loop))

    def test_log_posterior_hessian(self):
        print("==============Testing posterior Hessian (likelihood + beta prior)==============")
        d = 3
        N = 20
        J = 5
        (Y, X, phi) = generate_data(d, N, J)
        model = nbm.NegativeBinomialRegressionModel(torch.tensor(X), torch.tensor(Y),
                                                    lam=2., shape=3., scale=2.,
                                                    dispersion = phi, pivot=False)
        set_distinct_sd(model)
        beta = model.beta.detach().clone()
        hess_expected = torch.autograd.functional.hessian(model.log_posterior, beta).detach().numpy()
        hess_realized = model.log_posterior_hessian(beta).detach().numpy()
        self.assertTrue(np.allclose(hess_expected, hess_realized))

class TestNBSRTrendedGradients(unittest.TestCase):

    def test_log_lik_gradient(self):
        d = 3
        N = 20
        J = 5
        (Y, X, phi) = generate_data(d, N, J)
    
        tensorY = torch.tensor(Y)
        disp_model = dm.DispersionModel(tensorY.shape[1])
        model = nbsrd.NBSRTrended(torch.tensor(X), tensorY, disp_model=disp_model, lam=1., shape=3., scale=2.)
        z = model.log_likelihood_beta(model.beta)
        if model.beta.grad is not None:
            model.beta.grad.zero_()
        z.backward(retain_graph=True)
        grad_expected = model.beta.grad.data.numpy()
        grad_actual = model.log_lik_gradient_persample(model.beta).sum(0).data.numpy()
        print(grad_expected)
        print(grad_actual)
        self.assertTrue(np.allclose(grad_expected, grad_actual))

    def test_log_lik_gradient_pivot(self):
        d = 3
        N = 20
        J = 5
        (Y, X, phi) = generate_data(d, N, J)
    
        tensorY = torch.tensor(Y)
        disp_model = dm.DispersionModel(tensorY.shape[1])
        model = nbsrd.NBSRTrended(torch.tensor(X), tensorY, disp_model=disp_model, lam=1., shape=3., scale=2., pivot=True)
        z = model.log_likelihood_beta(model.beta)
        if model.beta.grad is not None:
            model.beta.grad.zero_()
        z.backward(retain_graph=True)
        grad_expected = model.beta.grad.data.numpy()
        grad_actual = model.log_lik_gradient_persample(model.beta).sum(0).data.numpy()
        print(grad_expected)
        print(grad_actual)
        self.assertTrue(np.allclose(grad_expected, grad_actual))

    def test_log_posterior_gradient(self):
        d = 3
        N = 20
        J = 5
        (Y, X, phi) = generate_data(d, N, J)
    
        tensorY = torch.tensor(Y)
        disp_model = dm.DispersionModel(tensorY.shape[1])
        model = nbsrd.NBSRTrended(torch.tensor(X), tensorY, disp_model=disp_model, lam=1., shape=3., scale=2.)
        set_distinct_sd(model)
        z = model.log_posterior(model.beta)
        if model.beta.grad is not None:
            model.beta.grad.zero_()
        z.backward(retain_graph=True)
        grad_expected = model.beta.grad.data.numpy()
        grad_actual = model.log_posterior_gradient(model.beta).data.numpy()
        print(grad_expected.shape)
        print(grad_actual.shape)
        self.assertTrue(np.allclose(grad_expected, grad_actual))

    def test_log_posterior_hessian(self):
        d = 3
        N = 20
        J = 5
        (Y, X, phi) = generate_data(d, N, J)

        tensorY = torch.tensor(Y)
        disp_model = dm.DispersionModel(tensorY.shape[1])
        model = nbsrd.NBSRTrended(torch.tensor(X), tensorY, disp_model=disp_model, lam=2., shape=3., scale=2.)
        set_distinct_sd(model)
        beta = model.beta.detach().clone()
        hess_expected = torch.autograd.functional.hessian(model.log_posterior, beta).detach().numpy()
        hess_realized = model.log_posterior_hessian(beta).detach().numpy()
        self.assertTrue(np.allclose(hess_expected, hess_realized))


if __name__ == '__main__':
    unittest.main()