"""Dispersion model of the NBSR (softmax) family, in the NB2 parameterization of the NBSR-HMC Stan model:

    Var(y_ij) = mu_ij + mu_ij^2 / phi_ij,     log phi_ij = b_0 + b_j + b_pi * logit(pi_ij) + w_i' b_w

phi here is the NB2 precision ("size"); NBSR's dispersion (Var = mu + phi_nbsr mu^2) is its inverse, so
log_dispersion(pi) = -forward(pi) is what the likelihood uses. Keeping Stan's convention makes the fitted
b_0, b_pi, b_j, b_w directly comparable with NBSR-HMC (in particular b_pi ~ N(1, 0.1): precision rises,
i.e. dispersion falls, with abundance).

pi_ij is the model's own composition (so the dispersion moves with beta), b_j is a per-feature offset with a
hierarchical scale, and W holds external per-sample covariates (e.g. log library size, log capture rate).
This matches the NBSR-HMC Stan model; the MAP is taken in the Stan model's non-centred parameterization,
b_j = sigma_bj * z_j, so that the hierarchical scale has a proper mode.

Priors: b_0 ~ N(0, 1), b_pi ~ N(1, 0.1), z_j ~ N(0, 1), sigma_bj ~ half-N(0, 0.5), b_w,q ~ N(0, sigma_b[q]).
"""
import math

import torch
from torch.func import grad, vmap

from nbsr.distributions import log_lognormal, log_normal, softplus_inv


class DispersionModel(torch.nn.Module):
    def __init__(self, feature_count, W=None, sigma_b=1.0, b_pi_prior=(1.0, 0.1), sigma_bj_prior_sd=0.5,
                 sigma_bj_init=0.1, link="logit", feature_offsets=True, estimate_sd=False, dtype=torch.float64):
        """
        feature_count : J
        link : the transform f of pi_ij that b_pi multiplies: "logit" (NBSR-HMC), "log", or any callable
            f(pi) built from differentiable torch operations (its derivatives with respect to log pi, which
            the closed-form beta gradient/Hessian need, are then taken by autograd elementwise). With
            link="log", feature_offsets=False and log total counts in W this is the previous NBSR dispersion
            model b0 + b1 log pi + b2 log R (up to the NB2 sign convention).
        feature_offsets : include the hierarchical per-feature offsets b_j.
        W : (N, Q) external dispersion covariates, or None. Give them on the scale you want in the model
            (the Stan model uses log w_i); no transform is applied here.
        sigma_b : prior sd of b_w, a scalar or a length-Q sequence.
        b_pi_prior : (mean, sd) of the normal prior on b_pi.
        sigma_bj_prior_sd : scale of the half-normal prior on sigma_bj.
        estimate_sd : also carry a per-feature sd (softplus(kappa)) for the log-normal density used when this
            model acts as a prior on free per-feature dispersions (NegativeBinomialRegressionModel with
            dispersion_prior=...). Not part of the trended model itself.
        """
        super().__init__()
        assert link in ("logit", "log") or callable(link), f"link must be 'logit', 'log' or a callable f(pi), got {link!r}"
        self.feature_count = feature_count
        self.link = link
        self.feature_offsets = feature_offsets
        self.softplus = torch.nn.Softplus()
        if W is not None:
            W = torch.as_tensor(W, dtype=dtype)
            assert W.ndim == 2, "W must be (samples, covariates)"
            self.register_buffer("W", W)
            self.covariate_count = W.shape[1]
        else:
            self.W = None
            self.covariate_count = 0
        sigma_b = torch.as_tensor(sigma_b, dtype=dtype).reshape(-1)
        if sigma_b.numel() == 1:
            sigma_b = sigma_b.expand(self.covariate_count)
        assert sigma_b.numel() == self.covariate_count, "sigma_b must be a scalar or have one entry per column of W"
        self.register_buffer("sigma_b", sigma_b.clone())
        self.register_buffer("b_pi_prior_mean", torch.tensor(float(b_pi_prior[0]), dtype=dtype))
        self.register_buffer("b_pi_prior_sd", torch.tensor(float(b_pi_prior[1]), dtype=dtype))
        self.register_buffer("sigma_bj_prior_sd", torch.tensor(float(sigma_bj_prior_sd), dtype=dtype))

        self.b_0 = torch.nn.Parameter(torch.zeros(1, dtype=dtype))
        self.b_pi = torch.nn.Parameter(torch.full((1,), float(b_pi_prior[0]), dtype=dtype))
        self.z_bj = torch.nn.Parameter(torch.zeros(feature_count, dtype=dtype))
        self.kappa_bj = torch.nn.Parameter(softplus_inv(torch.tensor(float(sigma_bj_init), dtype=dtype)).reshape(1))
        self.b_w = torch.nn.Parameter(torch.zeros(self.covariate_count, dtype=dtype))

        self.estimate_sd = estimate_sd
        self.kappa = torch.nn.Parameter(torch.randn(feature_count, dtype=dtype)) if estimate_sd else None

    # ---- derived parameters
    @property
    def sigma_bj(self):
        return self.softplus(self.kappa_bj)

    @property
    def b_j(self):
        if not self.feature_offsets:
            return torch.zeros_like(self.z_bj)
        return self.sigma_bj * self.z_bj

    def external_predictor(self):
        """w_i' b_w for every sample, (N,), or 0 when there are no external covariates."""
        if self.W is None:
            return 0.0
        return self.W @ self.b_w

    @staticmethod
    def logit(pi):
        return torch.log(pi) - torch.log1p(-pi)

    def link_fn(self, pi):
        """f(pi), elementwise."""
        if callable(self.link):
            return self.link(pi)
        return self.logit(pi) if self.link == "logit" else torch.log(pi)

    def forward(self, pi):
        """log NB2 precision phi_ij (Stan convention), (N, J), for a composition pi (N, J)."""
        log_phi = self.b_0 + self.b_j.unsqueeze(0) + self.b_pi * self.link_fn(pi)
        if self.W is not None:
            log_phi = log_phi + self.external_predictor().unsqueeze(1)
        if self.estimate_sd:
            log_phi = log_phi + 0.5 * self.get_sd() ** 2  # mean of the log-normal, see log_density.
        return log_phi

    def log_dispersion(self, pi):
        """log of NBSR's dispersion (Var = mu + phi mu^2) = -forward(pi)."""
        return -self.forward(pi)

    def link_derivatives(self, pi):
        """First and second derivatives of h(u) = f(exp(u)) with respect to u = log pi, elementwise.
        logit: 1/(1-pi) and pi/(1-pi)^2;  log: 1 and 0;  callable f: by autograd."""
        if callable(self.link):
            f = self.link
            h = lambda u: f(torch.exp(u))
            dh, d2h = grad(h), grad(grad(h))
            u = torch.log(pi).detach().reshape(-1)
            return vmap(dh)(u).reshape(pi.shape), vmap(d2h)(u).reshape(pi.shape)
        if self.link == "log":
            return torch.ones_like(pi), torch.zeros_like(pi)
        one_minus = 1.0 - pi
        return 1.0 / one_minus, pi / one_minus ** 2

    def log_prior(self):
        dtype = self.b_0.dtype
        zero, one = torch.zeros((), dtype=dtype), torch.ones((), dtype=dtype)
        lp = log_normal(self.b_0, zero, one).sum()
        lp = lp + log_normal(self.b_pi, self.b_pi_prior_mean, self.b_pi_prior_sd).sum()
        if self.feature_offsets:
            lp = lp + log_normal(self.z_bj, zero, one).sum()
            # half-normal on sigma_bj > 0: log 2 + log N(sigma; 0, s).
            lp = lp + math.log(2.0) + log_normal(self.sigma_bj, zero, self.sigma_bj_prior_sd).sum()
        if self.covariate_count:
            lp = lp + log_normal(self.b_w, torch.zeros_like(self.b_w), self.sigma_b).sum()
        return lp

    # ---- use as a prior on free per-feature dispersions (legacy NBSR workflow)
    def get_sd(self):
        assert self.estimate_sd, "construct with estimate_sd=True to use the log-normal density"
        return self.softplus(self.kappa)

    def log_density(self, dispersion, pi):
        """log-normal density of per-feature NBSR dispersions around the trended log dispersion at composition pi."""
        return log_lognormal(dispersion, self.log_dispersion(pi), self.get_sd().unsqueeze(0))
