import unittest
import time

import numpy as np
import torch

import nbsr.negbinomial_model as nbm
import nbsr.nbsr_dispersion as nbsrd
import nbsr.dispersion as dm
import nbsr.utils as utils

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

class TestNBSRGradients(unittest.TestCase):

    def test_log_lik_gradient(self):
        d = 3
        N = 20
        J = 5
        (Y, X, phi) = generate_data(d, N, J)
    
        print("==============Test gradient of log likelihood with no pivot==============")
        #Y_df = pd.DataFrame(Y.transpose(), dtype="int32")
        #X_df = pd.DataFrame(X)
        model = nbm.NegativeBinomialRegressionModel(torch.tensor(X), torch.tensor(Y), dispersion = phi, pivot=False)
        z = model.log_likelihood(model.beta)
        if model.beta.grad is not None:
            model.beta.grad.zero_()
        z.backward(retain_graph=True)
        grad_expected = model.beta.grad.data.numpy()
        start = time.perf_counter()
        grad_actual = model.log_lik_gradient(model.beta, tensorized=False).data.numpy()
        end = time.perf_counter()
        print("Elapsed with numpy = {}s".format((end - start)))
        print(grad_expected)
        print(grad_actual)
        self.assertTrue(np.allclose(grad_expected, grad_actual))
        
        pi = model.predict(model.beta, model.X)[0].data.numpy()
        s = np.sum(model.Y.data.numpy(), 1)
        mu = s[:,None] * pi
        start = time.perf_counter()
        grad_actual, _ = utils.log_lik_gradients(X, Y, pi, mu, phi, model.pivot)
        end = time.perf_counter()
        print("Elapsed with numba compilation = {}s".format((end - start)))
        print(grad_actual)
        self.assertTrue(np.allclose(grad_expected, grad_actual))

        # Timing should improve on the second call as compiled code will be called.
        start = time.perf_counter()
        grad_actual, _ = utils.log_lik_gradients(X, Y, pi, mu, phi, model.pivot)
        end = time.perf_counter()
        print("Elapsed with post compilation = {}s".format((end - start)))
        print(grad_actual)
        self.assertTrue(np.allclose(grad_expected, grad_actual))

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
        model = nbm.NegativeBinomialRegressionModel(torch.tensor(X), torch.tensor(Y), dispersion = phi, pivot=True)
        z = model.log_likelihood(model.beta)
        if model.beta.grad is not None:
            model.beta.grad.zero_()
        z.backward(retain_graph=True)
        grad_expected = model.beta.grad.data.numpy()
        grad_actual = model.log_lik_gradient(model.beta, tensorized=False).data.numpy()
        print(grad_expected)
        print(grad_actual)
        self.assertTrue(np.allclose(grad_expected, grad_actual))

        pi = model.predict(model.beta, model.X)[0].data.numpy()
        s = np.sum(model.Y.data.numpy(), 1)
        mu = s[:,None] * pi
        start = time.perf_counter()
        grad_actual, _ = utils.log_lik_gradients(X, Y, pi, mu, phi, model.pivot)
        end = time.perf_counter()
        print("Elapsed with numba compilation = {}s".format((end - start)))
        print(grad_actual)
        self.assertTrue(np.allclose(grad_expected, grad_actual))

    def test_log_beta_prior_gradient(self):
        d = 3
        N = 20
        J = 5
        (Y, X, phi) = generate_data(d, N, J)

        print("==============Test gradient of log prior over beta==============")
        #Y_df = pd.DataFrame(Y.transpose(), dtype="int32")
        #X_df = pd.DataFrame(X)
        model = nbm.NegativeBinomialRegressionModel(torch.tensor(X), torch.tensor(Y), dispersion = phi, pivot=False)
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
        print("==============Test log posterior gradient==============")
        d = 3
        N = 20
        J = 5
        (Y, X, phi) = generate_data(d, N, J)

        #Y_df = pd.DataFrame(Y.transpose(), dtype="int32")
        #X_df = pd.DataFrame(X)
        model = nbm.NegativeBinomialRegressionModel(torch.tensor(X), torch.tensor(Y), dispersion = phi, pivot=False)
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
        print("==============Testing Hessian computation==============")
        d = 3
        N = 20
        J = 5
        (Y, X, phi) = generate_data(d, N, J)
    
        #Y_df = pd.DataFrame(Y.transpose(), dtype="int32")
        #X_df = pd.DataFrame(X)
        model = nbm.NegativeBinomialRegressionModel(torch.tensor(X), torch.tensor(Y), dispersion = phi, pivot=False)

        log_lik_grad = model.log_lik_gradient(model.beta)
        hess_expected = torch.zeros(log_lik_grad.size(0), model.beta.size(0))
        start = time.perf_counter()
        # Compute the gradient for each component of y w.r.t. beta
        for k in range(log_lik_grad.size(0)):
            # Zero previous gradient
            if model.beta.grad is not None:
                model.beta.grad.zero_()

            # Backward on the k-th component of y
            log_lik_grad[k].backward(retain_graph=True)

            # Store the gradient
            hess_expected[k,:] = model.beta.grad
        end = time.perf_counter()
        print("Elapsed with torch = {}s".format((end - start)))

        hess_realized = torch.sum(model.log_lik_hessian_persample(model.beta),0).data.numpy()
        print(hess_expected.data.numpy()[0,:])
        print(hess_realized[0,:])
        self.assertTrue(np.allclose(hess_expected, hess_realized))

        # Compute Hessian using numba.
        pi = model.predict(model.beta, model.X)[0].data.numpy()
        s = np.sum(model.Y.data.numpy(), 1)
        mu = s[:,None] * pi
        start = time.perf_counter()
        _, hess_realized = utils.log_lik_gradients(X, Y, pi, mu, phi, model.pivot)
        end = time.perf_counter()
        print("Elapsed with numba compilation = {}s".format((end - start)))
        print(hess_realized[0,:])
        self.assertTrue(np.allclose(hess_expected, hess_realized))

class TestNBSRTrendedGradients(unittest.TestCase):

    def test_log_lik_gradient(self):
        d = 3
        N = 20
        J = 5
        (Y, X, phi) = generate_data(d, N, J)
    
        tensorY = torch.tensor(Y)
        disp_model = dm.DispersionModel(tensorY)
        model = nbsrd.NBSRTrended(torch.tensor(X), tensorY, disp_model=disp_model)
        z = model.log_likelihood2(model.beta)
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
        disp_model = dm.DispersionModel(tensorY)
        model = nbsrd.NBSRTrended(torch.tensor(X), tensorY, disp_model=disp_model, pivot=True)
        z = model.log_likelihood2(model.beta)
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
        disp_model = dm.DispersionModel(tensorY)
        model = nbsrd.NBSRTrended(torch.tensor(X), tensorY, disp_model=disp_model)
        model.specify_beta_prior(1, 3, 2)
        z = model.log_posterior(model.beta)
        if model.beta.grad is not None:
            model.beta.grad.zero_()
        z.backward(retain_graph=True)
        grad_expected = model.beta.grad.data.numpy()
        grad_actual = model.log_posterior_gradient(model.beta).data.numpy()
        print(grad_expected.shape)
        print(grad_actual.shape)
        self.assertTrue(np.allclose(grad_expected, grad_actual))


if __name__ == '__main__':
    unittest.main()