import unittest

import numpy as np
import pandas as pd

import nbsr.negbinomial_model as nbm

class TestNegBinModel(unittest.TestCase):
    
    def test_log_lik_gradient(self):
        # Generate data for testing.
        d = 3
        N = 20
        J = 5
        softplus = lambda x: np.log1p(np.exp(x))

        phi = softplus(np.random.randn(J))
        beta = np.random.randn(J * d)
        beta_reshape = beta.reshape(d, J)
        Y = np.zeros((N, J))
        X = np.zeros((N, d))
        s = np.random.poisson(10000, N)
        print(s)
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
    
        Y_df = pd.DataFrame(Y.transpose(), dtype="int32")
        X_df = pd.DataFrame(X)
        model = nbm.NegativeBinomialRegressionModel(X_df, Y_df, dispersion = phi, pivot=False)
        z = model.log_likelihood(model.beta)
        if model.beta.grad is not None:
            model.beta.grad.zero_()
        z.backward(retain_graph=True)
        grad_expected = model.beta.grad.data.numpy()
        grad_actual = model.log_lik_gradient(model.beta).data.numpy()
        self.assertTrue(np.allclose(grad_expected, grad_actual))
 
if __name__ == '__main__':
    unittest.main()