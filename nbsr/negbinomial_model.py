"""Negative Binomial Softmax Regression with one free dispersion per feature (or fixed dispersions).

    pi_i = softmax(x_i' beta)   (feature J fixed at 0 when pivot=True)
    y_ij ~ NB(s_i pi_ij, phi_j)

Gradients and Hessians of the log-likelihood with respect to the flat, covariate-major beta are in closed
form. Every term factorizes through u_ij = log pi_ij, whose derivative d u_ij / d eta_ik = 1[j=k] - pi_ik,
so the Hessian has the Kronecker structure handled by utils.kron_hessian.
"""
import torch

from nbsr.distributions import log_negbinomial, log_normal, log_invgamma, softplus_inv, nb_log_density_derivatives
from nbsr.utils import kron_hessian


class NegativeBinomialRegressionModel(torch.nn.Module):
    # when dispersion prior is unspecified, default to no prior.
    def __init__(self, X, Y, lam, shape, scale, dispersion_prior=None, dispersion=None, pivot=False, beta_prior_sd=None):
        """beta_prior_sd: fix the prior sd of beta per covariate (scalar or length-P) instead of learning it by
        empirical Bayes through psi. Mirrors sigma_beta2 given as data in the NBSR-HMC Stan model."""
        super().__init__()
        assert isinstance(X, torch.Tensor) and isinstance(Y, torch.Tensor)
        # Place X, Y on buffer so that they can be moved to GPU.
        self.register_buffer("X", X.to(torch.float64))
        self.register_buffer("Y", Y.to(torch.float64))
        self.register_buffer("lam", torch.tensor(lam, dtype=torch.float64))
        self.register_buffer("beta_var_shape", torch.tensor(shape, dtype=torch.float64))
        self.register_buffer("beta_var_scale", torch.tensor(scale, dtype=torch.float64))
        self.register_buffer("s", self.Y.sum(dim=1))  # library sizes

        self.pivot = pivot
        self.softplus = torch.nn.Softplus()
        self.sample_count = self.Y.shape[0]
        self.covariate_count = self.X.shape[1]
        self.rna_count = self.Y.shape[1]
        self.converged = False
        print("RNA count:", self.rna_count)
        print("Sample count:", self.sample_count)
        print("Covariate count:", self.covariate_count)

        # The parameters we adjust during training.
        self.dim = self.rna_count - 1 if pivot else self.rna_count
        self.beta = torch.nn.Parameter(torch.randn(self.covariate_count * self.dim, dtype=torch.float64), requires_grad=True)
        self.disp_model = dispersion_prior
        if dispersion is None:
            self.phi = torch.nn.Parameter(torch.randn(self.rna_count, dtype=torch.float64), requires_grad=True)
        else:
            self.phi = softplus_inv(torch.as_tensor(dispersion, dtype=torch.float64) + 1e-9)
        if beta_prior_sd is None:
            self.psi = torch.nn.Parameter(softplus_inv(torch.ones(self.covariate_count, dtype=torch.float64)), requires_grad=True)
            self.learn_beta_prior_sd = True
        else:
            sd = torch.as_tensor(beta_prior_sd, dtype=torch.float64).reshape(-1)
            if sd.numel() == 1:
                sd = sd.expand(self.covariate_count)
            assert sd.numel() == self.covariate_count, "beta_prior_sd must be a scalar or one value per covariate"
            self.register_buffer("psi", softplus_inv(sd.clone()))
            self.learn_beta_prior_sd = False

    def to_device(self, device):
        self.to(device)

    # ------------------------------------------------------------------ model
    def predict(self, beta, X):
        """Composition pi (N, J) and the linear predictor (N, J) for design X."""
        beta_ = torch.reshape(beta, (self.covariate_count, self.dim))
        log_unnorm_exp = torch.matmul(X, beta_)
        if self.pivot:
            log_unnorm_exp = torch.column_stack((log_unnorm_exp, torch.zeros(X.shape[0], dtype=beta.dtype, device=beta.device)))
        norm = torch.logsumexp(log_unnorm_exp, 1)
        pi = torch.exp(log_unnorm_exp - norm[:, None])
        return pi, log_unnorm_exp

    def dispersion(self, pi):
        """phi broadcastable against (N, J): here one free/fixed value per feature."""
        return self.softplus(self.phi)

    def log_likelihood(self, beta):
        """Log-likelihood of Y at beta (a scalar tensor)."""
        pi, _ = self.predict(beta, self.X)
        return log_negbinomial(self.Y, self.s[:, None] * pi, self.dispersion(pi)).sum()

    log_likelihood_beta = log_likelihood

    def log_beta_prior(self, beta):
        # Flat beta is covariate-major (see predict: X @ beta.reshape(covariate_count, dim)).
        # Transpose so that column d holds covariate d and broadcasts against sd[d].
        beta_ = beta.reshape(self.covariate_count, self.dim).T
        sd = self.softplus(self.psi)
        return torch.sum(log_normal(beta_, torch.zeros_like(sd), sd / self.lam))

    def log_posterior(self, beta):
        pi, _ = self.predict(beta, self.X)
        log_lik = self.log_likelihood(beta)
        sd = self.softplus(self.psi)
        # normal prior on beta -- 0 mean and sd = softplus(psi) / lam; inverse-gamma prior on sd^2.
        log_beta_prior = self.log_beta_prior(beta)
        log_var_prior = torch.sum(log_invgamma(sd ** 2, self.beta_var_shape, self.beta_var_scale)) if self.learn_beta_prior_sd else 0.0
        log_dispersion_prior = 0
        if self.disp_model is not None:
            # The dispersion model acts as a log-normal prior on the free per-feature dispersions, evaluated
            # at the sample-average composition.
            pi_bar = pi.mean(0, keepdim=True)
            log_dispersion_prior = torch.sum(self.disp_model.log_density(self.softplus(self.phi), pi_bar))
        return log_lik + log_beta_prior + log_var_prior + log_dispersion_prior

    def forward(self, beta):
        return self.log_posterior(beta)

    # ------------------------------------------------------------------ derivatives w.r.t. beta
    def _dispersion_terms(self, pi):
        """phi, and the first and second derivatives of log phi_ij with respect to u_ij = log pi_ij
        (c and c2, (N, J) or 0). Fixed/free per-feature dispersions do not depend on pi."""
        return self.dispersion(pi), 0.0, 0.0

    def _score_terms(self, beta, second=False):
        pi, _ = self.predict(beta, self.X)
        pi = pi.detach()
        mu = self.s[:, None] * pi
        phi, c, c2 = self._dispersion_terms(pi)
        phi = phi.detach() if torch.is_tensor(phi) else phi
        trended = torch.is_tensor(c)
        l_u, l_v, l_uu, l_uv, l_vv = nb_log_density_derivatives(self.Y, mu, phi, second=second)
        # d l_ij / d u_ij along the model, where log phi may move with u through c.
        G = l_u + c * l_v if trended else l_u
        if not second:
            return pi, G, None
        A = l_uu + 2.0 * c * l_uv + c ** 2 * l_vv + c2 * l_v if trended else l_uu
        return pi, G, A

    def log_lik_gradient_persample(self, beta):
        """(N, P*dim) gradient of each sample's log-likelihood w.r.t. the flat covariate-major beta.

        d l_i / d beta_{d,k} = x_id sum_j G_ij (1[j=k] - pi_ik) = x_id (G_ik - pi_ik sum_j G_ij).
        """
        pi, G, _ = self._score_terms(beta)
        Gk = (G - pi * G.sum(1, keepdim=True))[:, :self.dim]                # (N, dim)
        return torch.einsum("nd,nk->ndk", self.X, Gk).reshape(self.sample_count, -1)

    def log_lik_gradient(self, beta):
        return self.log_lik_gradient_persample(beta).sum(0)

    def log_beta_prior_gradient(self, beta):
        beta_ = beta.reshape(self.covariate_count, self.dim).T
        sd = self.softplus(self.psi)
        log_prior_grad = -(self.lam ** 2) * beta_ / sd ** 2
        # beta_ is (dim, covariate_count); transpose back so the flat gradient is covariate-major like beta.
        return log_prior_grad.T.flatten()

    def log_posterior_gradient(self, beta):
        return self.log_lik_gradient(beta) + self.log_beta_prior_gradient(beta)

    def log_likelihood_hessian(self, beta):
        """Closed-form Hessian of the log-likelihood w.r.t. the flat beta (covariate-major), on beta's device.

        d^2 l_ij / d eta_ik d eta_ik' = A_ij (1[j=k]-pi_ik)(1[j=k']-pi_ik') - G_ij pi_ik (1[k=k']-pi_ik'),
        which utils.kron_hessian sums over j and i.
        """
        pi, G, A = self._score_terms(beta, second=True)
        return kron_hessian(self.X, pi, A=A, B=-G, dim=self.dim)

    def log_posterior_hessian(self, beta):
        H_lik = self.log_likelihood_hessian(beta)
        # Flat beta is covariate-major (index d*dim + k), so covariate d's sd repeats dim times consecutively.
        sd = self.softplus(self.psi.detach()).repeat_interleave(self.dim).to(H_lik.dtype)
        # log N(beta; 0, sd/lam) has Hessian -lam^2/sd^2 on the diagonal.
        return H_lik - torch.diag(self.lam ** 2 / sd ** 2)
