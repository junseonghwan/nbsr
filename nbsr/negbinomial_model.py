"""Negative Binomial Softmax Regression with one free dispersion per feature (or fixed dispersions).

    pi_i = softmax(x_i' beta)   (feature J fixed at 0 when pivot=True)
    y_ij ~ NB(s_i pi_ij, phi_j)

Gradients and Hessians of the log-likelihood with respect to the flat, covariate-major beta are in closed
form. Every term factorizes through u_ij = log pi_ij, whose derivative d u_ij / d eta_ik = 1[j=k] - pi_ik,
so the Hessian has the Kronecker structure handled by utils.kron_hessian.
"""
import torch

from nbsr.distributions import log_negbinomial, log_normal, softplus_inv, nb_log_density_derivatives
from nbsr.utils import kron_hessian


class NegativeBinomialRegressionModel(torch.nn.Module):
    # when dispersion prior is unspecified, default to no prior.
    def __init__(self, X, Y, beta_prior_sd=10.0, dispersion_prior=None, dispersion=None, pivot=False):
        """
        X : (N, P) design with intercept; Y : (N, J) counts.
        beta_prior_sd : prior sd of beta, a scalar or one value per covariate (intercept first). Fixed, not
            learned: the run() driver sets it from a wide-prior stage-1 fit (DESeq2-style quantile matching)
            unless the user supplies it, mirroring sigma_beta2 given as data in the NBSR-HMC Stan model.
        dispersion_prior : a DispersionModel acting as a log-normal prior on free per-feature dispersions.
        dispersion : fixed per-feature dispersions (array of length J); None = free parameters.
        pivot : fix the last feature's coefficients at zero (reference category).
        """
        super().__init__()
        assert isinstance(X, torch.Tensor) and isinstance(Y, torch.Tensor)
        # Place X, Y on buffer so that they can be moved to GPU.
        self.register_buffer("X", X.to(torch.float64))
        self.register_buffer("Y", Y.to(torch.float64))
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

        sd = torch.as_tensor(beta_prior_sd, dtype=torch.float64).reshape(-1)
        if sd.numel() == 1:
            sd = sd.expand(self.covariate_count)
        assert sd.numel() == self.covariate_count, "beta_prior_sd must be a scalar or one value per covariate"
        self.register_buffer("beta_prior_sd", sd.clone())

        # The parameters we adjust during training.
        self.dim = self.rna_count - 1 if pivot else self.rna_count
        self.beta = torch.nn.Parameter(torch.randn(self.covariate_count * self.dim, dtype=torch.float64), requires_grad=True)
        self.disp_model = dispersion_prior
        if dispersion is None:
            self.phi = torch.nn.Parameter(torch.randn(self.rna_count, dtype=torch.float64), requires_grad=True)
        else:
            self.phi = softplus_inv(torch.as_tensor(dispersion, dtype=torch.float64) + 1e-9)

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
        sd = self.beta_prior_sd
        return torch.sum(log_normal(beta_, torch.zeros_like(sd), sd))

    def log_posterior(self, beta):
        pi, _ = self.predict(beta, self.X)
        log_lik = self.log_likelihood(beta)
        log_beta_prior = self.log_beta_prior(beta)
        log_dispersion_prior = 0
        if self.disp_model is not None:
            # The dispersion model acts as a log-normal prior on the free per-feature dispersions, evaluated
            # at the sample-average composition.
            pi_bar = pi.mean(0, keepdim=True)
            log_dispersion_prior = torch.sum(self.disp_model.log_density(self.softplus(self.phi), pi_bar))
        return log_lik + log_beta_prior + log_dispersion_prior

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
        log_prior_grad = -beta_ / self.beta_prior_sd ** 2
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
        sd = self.beta_prior_sd.repeat_interleave(self.dim).to(H_lik.dtype)
        # log N(beta; 0, sd) has Hessian -1/sd^2 on the diagonal.
        return H_lik - torch.diag(1.0 / sd ** 2)
