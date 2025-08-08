import torch

from nbsr.distributions import log_negbinomial, log_normal, log_invgamma, softplus_inv
from nbsr.utils import hessian_nbsr

# This model implements free to vary dispersion parameterization.
class NegativeBinomialRegressionModel(torch.nn.Module):
    # when dispersion prior is unspecified, default to no prior.
    def __init__(self, X, Y, lam, shape, scale, dispersion_prior=None, dispersion=None, pivot=False):
        super().__init__()
        assert(isinstance(X, torch.Tensor))
        assert(isinstance(Y, torch.Tensor))
        # Place X, Y on buffer so that they can be moved to GPU.
        self.register_buffer("X", X)
        self.register_buffer("Y", Y)
        self.register_buffer("lam", torch.tensor(lam, dtype=torch.float64))
        self.register_buffer("beta_var_shape", torch.tensor(shape, dtype=torch.float64))
        self.register_buffer("beta_var_scale", torch.tensor(scale, dtype=torch.float64))
        
        self.s = torch.sum(self.Y, dim=1)  # Summing over rows
        self.pivot = pivot
        self.softplus = torch.nn.Softplus()
        self.sample_count = self.Y.shape[0]
        #self.covariate_count = self.XX.shape[1] + 1 # +1 for the intercept term.
        self.covariate_count = self.X.shape[1]
        self.rna_count = self.Y.shape[1]
        self.converged = False
        #self.X = torch.cat([torch.ones(self.sample_count, 1), self.XX], dim = 1)
        print("RNA count:", self.rna_count)
        print("Sample count:", self.sample_count)
        print("Covariate count:", self.covariate_count)

        # The parameters we adjust during training.
        self.dim = self.rna_count - 1 if pivot else self.rna_count
        self.beta = torch.nn.Parameter(torch.randn(self.covariate_count * self.dim, dtype=torch.float64), requires_grad=True)
        #self.beta = torch.nn.Parameter(torch.zeros(self.covariate_count * self.dim, dtype=torch.float64), requires_grad=True)
        self.disp_model = dispersion_prior
        if dispersion is None:
            self.phi = torch.nn.Parameter(torch.randn(self.rna_count, dtype=torch.float64), requires_grad=True)
        else:
            self.phi = softplus_inv(torch.tensor(dispersion + 1e-9, requires_grad=False))
        self.psi = torch.nn.Parameter(softplus_inv(torch.ones(self.covariate_count, dtype=torch.float64)), requires_grad=True)

    # TODO: MARKED FOR REMOVAL
    # def specify_beta_prior(self, lam, beta_var_shape, beta_var_scale):
    #     self.lam = torch.tensor(lam, requires_grad=False)
    #     self.beta_var_shape = torch.tensor(beta_var_shape, requires_grad=False)
    #     self.beta_var_scale = torch.tensor(beta_var_scale, requires_grad=False)
    #     print("Initial psi:", self.psi)

    def to_device(self, device):
        self.to(device)

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
        
        log_lik_vals = log_negbinomial(self.Y, self.s[:, None] * norm_expr, self.softplus(self.phi))
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
        pi,_ = self.predict(beta, self.X)
        log_lik = self.log_likelihood(beta)
        sd = self.softplus(self.psi)
        dispersion = self.softplus(self.phi)
        # normal prior on beta -- 0 mean and sd = softplus(psi).
        log_beta_prior = self.log_beta_prior(beta)
        # inv gamma prior on var = sd^2 -- hyper parameters specified to the model.
        log_var_prior = torch.sum(log_invgamma(sd**2, self.beta_var_shape, self.beta_var_scale))
        log_dispersion_prior = 0
        if self.disp_model is not None:
            # Take sample average of pi's since we only support feature wise dispersion for standard NBSR.
            pi_bar = pi.mean(0)
            log_dispersion_prior = torch.sum(self.disp_model.log_density(dispersion, pi_bar))
        log_posterior = log_lik + log_beta_prior + log_var_prior + log_dispersion_prior
        return(log_posterior)

    def forward(self, beta):
        return self.log_posterior(beta)

    def predict(self, beta, X):
        beta_ = torch.reshape(beta, (self.covariate_count, self.dim))
        log_unnorm_exp = torch.matmul(X, beta_)
        if self.pivot:
            log_unnorm_exp = torch.column_stack((log_unnorm_exp, torch.zeros(self.sample_count, device=beta.device)))
        norm = torch.logsumexp(log_unnorm_exp, 1)
        norm_expr = torch.exp(log_unnorm_exp - norm[:,None])
        pi = norm_expr
        return(pi, log_unnorm_exp)

    def log_lik_gradient_persample_tensorized(self, beta):
        beta_ = beta.view(self.covariate_count, self.dim)
        dispersion = self.softplus(self.phi)
        J, N, P = self.rna_count, self.sample_count, self.covariate_count
        device = beta.device

        log_unnorm_exp = self.X @ beta_
        if self.pivot:
            log_unnorm_exp = torch.column_stack(
                (log_unnorm_exp, torch.zeros(N, device=device))
            )

        norm = torch.logsumexp(log_unnorm_exp, dim=1)
        norm_expr = torch.exp(log_unnorm_exp - norm[:, None]) # N × J

        mean = self.s[:, None] * norm_expr
        sigma2 = mean + dispersion * mean**2
        r = 1.0 / dispersion
        D = dispersion * mean**2 / sigma2 # N × J

        # Pi is (N × J × J)
        eye_J = torch.eye(J, device=device).unsqueeze(0).expand(N, -1, -1)
        Pi = eye_J - norm_expr.unsqueeze(1)

        # XPi is (N × J × P × J)
        XPi = torch.einsum('np,njk->njkp', self.X, Pi)

        c0 = -r * D + self.Y * (1 - D) # N × J
        ret = torch.einsum('nj,njkp->njkp', c0, XPi) # N × J × P × J

        grad = torch.sum(ret, dim=1).transpose(1, 2) # N × P × J

        if self.pivot: # drop last column
            grad = grad[:, :, :-1] # N × P × dim

        grad = grad.reshape(N, self.dim * P)
        return grad
    
    ### Gradient of the model
    def log_lik_gradient_persample(self, beta):
        """
        Computes the gradient of the log-likelihood function with respect to the model parameters for each sample.

        Args:
            beta (torch.Tensor): A tensor of shape (covariate_count * dim,) containing the model parameters.

        Returns:
            torch.Tensor: A tensor of shape (sample_count, covariate_count * dim) containing the gradient of the log-likelihood function with respect to the model parameters for each sample.
        """        
        device = beta.device
        dtype  = beta.dtype

        N, J, P = self.sample_count, self.rna_count, self.covariate_count
        dim = self.dim

        beta_ = beta.view(P, dim)
        dispersion = self.softplus(self.phi)
        r = 1. / dispersion
        grad = torch.empty(N, P * dim, device=device, dtype=dtype)
        
        # log_unnorm_exp = torch.matmul(self.X, beta_)
        # if self.pivot:
        #     log_unnorm_exp = torch.column_stack((log_unnorm_exp, torch.zeros(self.sample_count, device=beta.device)))
        # norm = torch.logsumexp(log_unnorm_exp, 1)
        # norm_expr = torch.exp(log_unnorm_exp - norm[:,None])
        # I = torch.eye(J, device = beta.device)

        # grad = torch.zeros(self.sample_count, self.dim * self.covariate_count, device=beta.device)
        #for idx, (pi, x, y) in enumerate(zip(norm_expr, self.X, self.Y)):
        for n in range(N):
            x = self.X[n]
            y = self.Y[n]
            s = torch.sum(y)

            logits = x @ beta_
            if self.pivot:
                # append baseline (zero) column for softmax over J classes
                logits = torch.cat([logits, torch.zeros(1, device=device, dtype=dtype)], dim=0)
            m  = torch.logsumexp(logits, dim=0)
            pi = torch.exp(logits - m)    # (J,)

            mean = s * pi
            sigma2 = mean + dispersion * (mean ** 2)

            # temp_j = r*(1 - t0_j) + y_j*(2 - t0_j), where t0_j = (mean + 2*disp*mean^2)/sigma2
            t0   = (mean + 2 * dispersion * (mean ** 2)) / sigma2
            temp = r * (1.0 - t0) + y * (2.0 - t0)     # (J,)

            # res = outer(x, temp) - [sum_j temp_j] * outer(x, pi)
            # (rank-1 correction: (I - pi 1^T) applied without J×J)
            xt  = torch.outer(x, temp)                  # (P, J)
            res = xt - torch.outer(x, pi) * temp.sum()  # (P, J)

            # Drop the pivot column to match parameter dimension when pivoting
            if self.pivot:
                res = res[:, :-1]                       # (P, dim) with dim = J-1

            grad[n] = res.reshape(-1)
        return grad

    def log_lik_gradient(self, beta):
        """
        Computes the gradient of the log-likelihood function with respect to the model parameters.

        Args:
            beta (torch.Tensor): The model parameters.

        Returns:
            torch.Tensor: The gradient of the log-likelihood function with respect to the model parameters.
        """
        return torch.sum(self.log_lik_gradient_persample(beta), 0)

    def log_beta_prior_gradient(self, beta):
        beta_ = beta.reshape(self.dim, self.covariate_count)
        sd = self.softplus(self.psi)
        log_prior_grad = -(self.lam**2) * beta_ / sd**2
        return(log_prior_grad.flatten())

    def log_posterior_gradient(self, beta):
        """
        Computes the gradient of the log posterior distribution with respect to the model parameters.

        Args:
            beta (torch.Tensor): A tensor of shape (dim * covariate_count,) representing the model parameters.

        Returns:
            torch.Tensor: A tensor of the same shape as `beta` representing the gradient of the log posterior distribution.
        """
        log_prior_grad = self.log_beta_prior_gradient(beta)
        
        log_lik_grad = self.log_lik_gradient(beta)
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

    def log_likelihood_hessian(self, beta):
        pi = self.predict(beta, self.X)[0].detach()
        phi = self.softplus(self.phi.detach())
        mu = self.s[:,None] * pi
        return hessian_nbsr(self.X.numpy(), 
                            self.Y.numpy(), 
                            pi.numpy(), 
                            mu.numpy(), 
                            phi.numpy(), 
                            self.pivot)

    def log_posterior_hessian(self, beta):
        sd = self.softplus(self.psi).repeat(self.dim)
        return self.log_likelihood_hessian(beta) + (1/sd**2) * torch.eye(self.dim * self.covariate_count)

    # def log_posterior_hessian(self, beta):
    #     sd = self.softplus(self.psi).repeat(self.dim)
    #     return torch.sum(self.log_lik_hessian_persample(beta), 0) + (1/sd**2) * torch.eye(self.dim * self.covariate_count)

