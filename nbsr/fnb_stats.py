"""Feature-wise negative binomial regression with a covariate-dependent dispersion model.

For each gene j the model is

    y_ij ~ NB(mu_ij, phi_ij)
    log mu_ij  = log s_i + x_i' beta_j
    log phi_ij = a_j + b_j log mu_ij + w_i' gamma_j

with independent normal priors on beta_j (per covariate), b_j and gamma_j, and the DESeq2 dispersion-trend
prior on a_j. Every gene is fitted to its MAP by a batched damped-Newton solver (all genes at once, closed-form
gradients and Hessians), and a Wald test is built from either the model-based covariance (inverse negative
Hessian at the mode) or a sandwich covariance that stays valid when the NB variance is misspecified.
"""
import warnings
from typing import Optional

import numpy as np
import pandas as pd
import scipy.stats as ss
import torch
from joblib import Parallel, delayed
from torch.func import functional_call, grad, hessian, vmap
from tqdm import tqdm

from nbsr.dataset import Dataset
from nbsr.distributions import log_negbinomial, log_normal, nb_log_density_derivatives
from nbsr.featurewise_dispersion import LogDispersionTrendPrior, MeanPowerCovariateDispersion
from nbsr.featurewise_nb_model import FeaturewiseNegBinom


class FeaturewiseNBStats:
    """Fit the feature-wise NB model to every gene of a Dataset and test contrasts of the mean model.

    Parameter layout (matches FeaturewiseNegBinom.pack()): [beta (P), a, b, gamma (D)].

    Parameters
    ----------
    dataset : Dataset
    b_prior_sd : prior sd of b_j, the within-gene slope of log dispersion on log mean. In a two-group design
        with the same factor in the dispersion model, b_j and gamma_j are nearly collinear (log mu takes only
        two values apart from the size factor), so this prior is what keeps them identified.
    beta_prior_sd : prior sd of the non-intercept mean coefficients, natural-log scale. The default is flat
        for any identifiable log fold change; it only keeps the MAP finite for separated genes (one group all
        zeros). Pass a smaller value for deliberate shrinkage. The intercept always gets INTERCEPT_PRIOR_SD.
    gamma_prior_sd : prior sd of gamma_j, the dispersion-covariate coefficients.
    """

    # The intercept is the log reference-group mean and is always well identified by the counts, so its prior
    # is effectively flat (this equals DESeq2's 1e-6 ridge) instead of pulling it toward zero.
    INTERCEPT_PRIOR_SD = 1e3

    def __init__(self, dataset: Dataset, b_prior_sd=0.1, beta_prior_sd=5.0, gamma_prior_sd=1.0):
        self.dataset = dataset
        self.dispersion_prior = LogDispersionTrendPrior(dataset.a0, dataset.a1, dataset.dispersion_prior_var)
        self.b_prior_sd = b_prior_sd
        self.beta_prior_sd = beta_prior_sd
        self.gamma_prior_sd = gamma_prior_sd
        # Per-gene problems are tiny; fit and differentiate in float64 regardless of the dataset dtype
        # (in float32 the optimizers stall well short of the optimum).
        self.fit_dtype = torch.float64
        self.results_ = None

    # ------------------------------------------------------------------ model set-up

    @property
    def n_parameters(self):
        return self.dataset.mean_covariate_count + 2 + self.dataset.dispersion_covariate_count

    def _tensors(self):
        """Dataset tensors in the fitting dtype: Y (N, G), X (N, P), W (N, D) or None, sf (N,), mu_bar (G,)."""
        dtype = self.fit_dtype
        ds = self.dataset
        W = ds.W.to(dtype) if ds.W is not None else None
        return ds.Y.to(dtype), ds.X.to(dtype), W, ds.size_factors.to(dtype), ds.mu_bar.to(dtype)

    def _beta_prior_sd_vector(self):
        """Per-covariate prior sd for beta: INTERCEPT_PRIOR_SD on the intercept, beta_prior_sd elsewhere."""
        names = self.dataset.covariate_names
        sd = torch.full((len(names),), float(self.beta_prior_sd), dtype=self.fit_dtype)
        if "Intercept" in names:
            sd[names.index("Intercept")] = self.INTERCEPT_PRIOR_SD
        return sd

    def _build_model(self, mu_bar):
        """A FeaturewiseNegBinom module for one gene. Used as a template (priors, pack/unpack layout) and by
        the per-gene L-BFGS path; parameter values are set by the caller."""
        disp_model = MeanPowerCovariateDispersion(
            self.dataset.dispersion_covariate_count, self.dispersion_prior, mu_bar,
            self.b_prior_sd, self.gamma_prior_sd)
        model = FeaturewiseNegBinom(self.dataset.mean_covariate_count, disp_model, self._beta_prior_sd_vector())
        return model.to(self.fit_dtype)

    def _initial_params(self, Y, X, sf, mu_bar):
        """(G, p) warm start: beta from least squares of log((y + 0.5) / sf) on X, a_j at its trend-prior
        mean, b_j and gamma_j at zero. Starting beta at zero, or the intercept alone at log(mu_bar), leaves
        genes with a large fold change far enough from the optimum that Newton/L-BFGS steps overshoot to NaN."""
        dtype = self.fit_dtype
        n_genes = Y.shape[1]
        Z = torch.log((Y + 0.5) / sf.unsqueeze(-1))                        # (N, G)
        beta0 = torch.linalg.lstsq(X, Z).solution.T                          # (G, P)
        a0 = self.dispersion_prior.mean(mu_bar).to(dtype).reshape(-1, 1)     # (G, 1)
        b0 = torch.zeros(n_genes, 1, dtype=dtype)
        gamma0 = torch.zeros(n_genes, self.dataset.dispersion_covariate_count, dtype=dtype)
        return torch.cat([beta0, a0, b0, gamma0], dim=1)

    # ------------------------------------------------------------------ log posterior and derivatives

    @staticmethod
    def _linear_predictors(theta, X, W, sf):
        """eta = log mu and v = log phi for all genes, (G, N) each, plus the pieces the derivatives reuse."""
        P = X.shape[1]
        beta, a, b, gamma = theta[:, :P], theta[:, P], theta[:, P + 1], theta[:, P + 2:]
        eta = beta @ X.T + torch.log(sf)
        v = a.unsqueeze(1) + b.unsqueeze(1) * eta
        if W is not None:
            v = v + gamma @ W.T
        mu, phi, r = torch.exp(eta), torch.exp(v), torch.exp(-v)
        q = 1.0 / (1.0 + phi * mu)                                           # = r / (r + mu)
        return eta, v, mu, phi, r, q

    def _prior_terms(self, template, mu_bar):
        dm = template.dispersion_model
        return dict(beta_sd=template.beta_prior_sd,
                    a_mean=dm.disp_trend_prior.mean(mu_bar),
                    a_var=dm.disp_trend_prior.disp_prior_var,
                    b_sd=dm.b_prior_sd,
                    gamma_sd=dm.gamma_prior_sd)

    def _log_posterior_terms(self, theta, YT, mu_bar, X, W, sf, template, want_grad=True, want_hessian=True):
        """Log posterior and, optionally, its gradient and Hessian for all genes at once, in closed form.

        theta: (G, p); YT: (G, N) counts. The log posterior depends on theta only through eta (linear in beta)
        and v = a + b eta + w'gamma, so with the per-sample derivatives of the NB log density w.r.t. (eta, v)
        the gradient and Hessian are weighted sums of x_i, w_i and eta_i. The single second-order term of the
        map theta -> v is d^2 v / (db dbeta) = x_i. Returns (f (G,), g (G, p) or None, H (G, p, p) or None).
        """
        P = X.shape[1]
        D = W.shape[1] if W is not None else 0
        beta, a, b, gamma = theta[:, :P], theta[:, P], theta[:, P + 1], theta[:, P + 2:]
        y = YT
        eta, v, mu, phi, r, q = self._linear_predictors(theta, X, W, sf)
        pr = self._prior_terms(template, mu_bar)

        f = log_negbinomial(y, mu, phi).sum(1)
        f = f + log_normal(beta, torch.zeros_like(beta), pr["beta_sd"]).sum(1)
        f = f + log_normal(a, pr["a_mean"], torch.sqrt(pr["a_var"]))
        f = f + log_normal(b, torch.zeros_like(b), pr["b_sd"])
        if D:
            f = f + log_normal(gamma, torch.zeros_like(gamma), pr["gamma_sd"]).sum(1)
        if not want_grad and not want_hessian:
            return f, None, None

        l_u, l_v, l_uu, l_uv, l_vv = nb_log_density_derivatives(y, mu, phi, second=want_hessian)
        bb = b.unsqueeze(1)
        g = torch.empty_like(theta)
        g[:, :P] = (l_u + bb * l_v) @ X - beta / pr["beta_sd"] ** 2
        g[:, P] = l_v.sum(1) - (a - pr["a_mean"]) / pr["a_var"]
        g[:, P + 1] = (l_v * eta).sum(1) - b / pr["b_sd"] ** 2
        if D:
            g[:, P + 2:] = l_v @ W - gamma / pr["gamma_sd"] ** 2
        if not want_hessian:
            return f, g, None

        c = l_uv + bb * l_vv                          # weight of x_i in the beta x (a, b, gamma) blocks
        wbb = l_uu + 2.0 * bb * l_uv + bb ** 2 * l_vv  # weight of x_i x_i' in the beta block

        p = theta.shape[1]
        H = torch.zeros(theta.shape[0], p, p, dtype=theta.dtype)
        H[:, :P, :P] = torch.einsum("gn,ni,nj->gij", wbb, X, X)
        H[:, :P, P] = c @ X
        H[:, :P, P + 1] = (c * eta + l_v) @ X         # + l_v x_i from d^2 v / (db dbeta)
        H[:, P, P] = l_vv.sum(1)
        H[:, P, P + 1] = (l_vv * eta).sum(1)
        H[:, P + 1, P + 1] = (l_vv * eta ** 2).sum(1)
        if D:
            H[:, :P, P + 2:] = torch.einsum("gn,ni,nd->gid", c, X, W)
            H[:, P, P + 2:] = l_vv @ W
            H[:, P + 1, P + 2:] = (l_vv * eta) @ W
            H[:, P + 2:, P + 2:] = torch.einsum("gn,nd,ne->gde", l_vv, W, W)
        H = torch.triu(H) + torch.triu(H, diagonal=1).transpose(-1, -2)
        prior_precision = torch.cat([1.0 / pr["beta_sd"] ** 2,
                                     (1.0 / pr["a_var"]).reshape(1),
                                     (1.0 / pr["b_sd"] ** 2).reshape(1),
                                     torch.full((D,), (1.0 / pr["gamma_sd"] ** 2).item(), dtype=theta.dtype)])
        idx = torch.arange(p)
        H[:, idx, idx] -= prior_precision
        return f, g, H

    def _score_outer_products(self, theta, YT, mu_bar, X, W, sf):
        """sum_i s_i s_i' per gene, (G, p, p), where s_i is sample i's contribution to the log-likelihood
        score. This is the middle of the sandwich covariance H^-1 (sum_i s_i s_i') H^-1."""
        P = X.shape[1]
        b = theta[:, P + 1]
        eta, v, mu, phi, r, q = self._linear_predictors(theta, X, W, sf)
        l_u, l_v, _, _, _ = nb_log_density_derivatives(YT, mu, phi, second=False)
        # d eta_i / d theta = [x_i, 0, 0, 0];  d v_i / d theta = [b x_i, 1, eta_i, w_i]
        S = torch.zeros(YT.shape[0], YT.shape[1], theta.shape[1], dtype=theta.dtype)
        S[:, :, :P] = (l_u + b.unsqueeze(1) * l_v).unsqueeze(-1) * X.unsqueeze(0)
        S[:, :, P] = l_v
        S[:, :, P + 1] = l_v * eta
        if W is not None:
            S[:, :, P + 2:] = l_v.unsqueeze(-1) * W.unsqueeze(0)
        return torch.einsum("gnp,gnq->gpq", S, S)

    def _make_log_posterior_fn(self, model):
        """Per-gene log posterior as a pure function of the packed parameter vector, for autograd. This is
        the reference implementation that the closed-form derivatives are tested against."""
        def log_posterior_fn(theta, y, mu_bar, X, W, sf):
            params = model.unpack(theta)
            params["dispersion_model.mu_bar"] = mu_bar
            mu, log_phi = functional_call(model, params, (X, W, sf))
            log_lik = log_negbinomial(y, mu, torch.exp(log_phi)).sum()
            log_prior_beta = log_normal(params["beta"], torch.zeros_like(params["beta"]), model.beta_prior_sd).sum()
            log_prior_dispersion = model.dispersion_model.compute_log_prior(
                a=params["dispersion_model.a"], b=params["dispersion_model.b"],
                mu_bar=mu_bar, gamma=params.get("dispersion_model.gamma"))
            return log_lik + log_prior_beta + log_prior_dispersion
        return log_posterior_fn

    # ------------------------------------------------------------------ fitting

    def fit(self, method: str = "newton", **kwargs):
        """Fit all genes. method="newton" (default) runs the batched damped-Newton solver, see fit_newton;
        method="lbfgs" runs per-gene L-BFGS in parallel, see fit_lbfgs. kwargs go to the chosen method."""
        if method == "newton":
            return self.fit_newton(**kwargs)
        if method == "lbfgs":
            return self.fit_lbfgs(**kwargs)
        raise ValueError(f"unknown method {method!r}; use 'newton' or 'lbfgs'")

    def fit_newton(self, max_iter: int = 100, tol_grad: float = 1e-4, max_backtrack: int = 12,
                   derivatives: str = "analytic", verbose: bool = False):
        """Damped (Levenberg-Marquardt) Newton ascent on the log posterior of every gene at once.

        Each iteration solves (-H_g + lam_g I) step = grad_g for every active gene in one batched Cholesky,
        accepts the step where it raises the log posterior, and otherwise raises lam_g and retries. A gene is
        converged when the sup-norm of its gradient drops below tol_grad; converged genes leave the active
        set. derivatives="autograd" uses torch.func on the reference log posterior instead of the closed
        forms (several times slower; kept for testing).
        """
        dtype = self.fit_dtype
        Y, X, W, sf, mu_bar = self._tensors()
        YT = Y.T.contiguous()
        n_genes, p = Y.shape[1], self.n_parameters
        eye = torch.eye(p, dtype=dtype)
        template = self._build_model(mu_bar[0])

        if derivatives == "analytic":
            def val_fn(th, y, mb):
                return self._log_posterior_terms(th, y, mb, X, W, sf, template, want_grad=False, want_hessian=False)[0]
            def grad_fn(th, y, mb):
                return self._log_posterior_terms(th, y, mb, X, W, sf, template, want_hessian=False)[1]
            def hess_fn(th, y, mb):
                return self._log_posterior_terms(th, y, mb, X, W, sf, template)[2]
        elif derivatives == "autograd":
            log_post = self._make_log_posterior_fn(template)
            in_dims = (0, 0, 0, None, None, None)
            def val_fn(th, y, mb):
                return vmap(log_post, in_dims=in_dims)(th, y, mb, X, W, sf)
            def grad_fn(th, y, mb):
                return vmap(grad(log_post), in_dims=in_dims)(th, y, mb, X, W, sf)
            def hess_fn(th, y, mb):
                return vmap(hessian(log_post), in_dims=in_dims)(th, y, mb, X, W, sf)
        else:
            raise ValueError(f"derivatives must be 'analytic' or 'autograd', got {derivatives!r}")

        theta = self._initial_params(Y, X, sf, mu_bar)
        assert theta.shape == (n_genes, p) and template.pack().shape == (p,)

        f = val_fn(theta, YT, mu_bar)
        lam = torch.full((n_genes,), 1e-3, dtype=dtype)
        converged = torch.zeros(n_genes, dtype=torch.bool)
        n_iter_used = torch.zeros(n_genes, dtype=torch.int64)
        grad_norm = torch.full((n_genes,), torch.inf, dtype=dtype)

        n_iter = 0
        for it in range(max_iter):
            idx = (~converged).nonzero().squeeze(-1)
            if idx.numel() == 0:
                break
            th, y_a, mb_a, f_a, lam_a = theta[idx], YT[idx], mu_bar[idx], f[idx], lam[idx]

            g = grad_fn(th, y_a, mb_a)
            gn = g.abs().amax(dim=1)
            grad_norm[idx] = gn
            done = (gn < tol_grad) & torch.isfinite(f_a)
            converged[idx[done]] = True
            if verbose:
                print(f"newton iter {it:3d} | active {int((~done).sum()):6d} | "
                      f"max |grad| {gn[~done].max().item() if (~done).any() else 0.0:.3e}")
            if done.all():
                break
            n_iter = it + 1
            n_iter_used[idx[~done]] += 1

            H = hess_fn(th, y_a, mb_a)
            pending = ~done
            for _ in range(max_backtrack):
                A = -H + lam_a.reshape(-1, 1, 1) * eye
                L, info = torch.linalg.cholesky_ex(A)
                L = torch.where((info == 0).reshape(-1, 1, 1), L, eye)   # placeholder where A is indefinite
                step = torch.cholesky_solve(g.unsqueeze(-1), L).squeeze(-1)
                th_new = th + step
                f_new = val_fn(th_new, y_a, mb_a)
                # Near the optimum a step changes f by less than float64 roundoff on |f| ~ 1e2-1e4; without
                # this slack good steps would be rejected and lam would grow without bound.
                f_slack = 1e-10 * torch.clamp(f_a.abs(), min=1.0)
                accept = pending & (info == 0) & torch.isfinite(f_new) & (f_new >= f_a - f_slack)
                th = torch.where(accept.unsqueeze(-1), th_new, th)
                f_a = torch.where(accept, f_new, f_a)
                lam_a = torch.where(accept, torch.clamp(lam_a / 3.0, min=1e-8), lam_a)
                pending = pending & ~accept
                if not pending.any():
                    break
                lam_a = torch.where(pending, lam_a * 10.0, lam_a)
            theta[idx], f[idx], lam[idx] = th, f_a, lam_a

        # Genes updated on the last iteration have not had their gradient checked yet.
        idx = (~converged).nonzero().squeeze(-1)
        if idx.numel():
            gn = grad_fn(theta[idx], YT[idx], mu_bar[idx]).abs().amax(dim=1)
            grad_norm[idx] = gn
            converged[idx[(gn < tol_grad) & torch.isfinite(f[idx])]] = True
        if verbose:
            print(f"newton done: {int(converged.sum())}/{n_genes} genes converged in <= {n_iter} iterations")

        self.results_ = {
            "params": theta,
            "log_posterior": f,
            "converged": converged,
            "n_iter": n_iter_used,
            "grad_norm": grad_norm,
        }
        self._compute_hessians(derivatives=derivatives)

    def fit_lbfgs(self, n_cpus: int = 1, max_iter: int = 100, lr: float = 1.0, tol: float = 1e-6):
        """Per-gene L-BFGS (strong Wolfe line search) over the joblib/loky pool. Much slower than fit_newton
        for these tiny problems and converges less tightly (it stops on relative loss change); kept as an
        independent check."""
        Y, X, W, sf, mu_bar = self._tensors()
        n_genes = Y.shape[1]
        theta0 = self._initial_params(Y, X, sf, mu_bar)

        # loky returns results in submission order, so position == gene index.
        job_results = list(tqdm(
            Parallel(n_jobs=n_cpus, backend="loky", batch_size="auto", return_as="generator")(
                delayed(self._fit_gene_lbfgs)(Y[:, i], X, W, sf, mu_bar[i], theta0[i], max_iter, lr, tol)
                for i in range(n_genes)),
            total=n_genes, desc="Fitting genes"))

        params = torch.stack([theta for theta, _ in job_results])
        converged = torch.tensor([ok for _, ok in job_results])
        self.results_ = {"params": params, "converged": converged}
        self._compute_hessians()

    def _fit_gene_lbfgs(self, y, X, W, sf, mu_bar, theta0, max_iter, lr, tol):
        model = self._build_model(mu_bar)
        with torch.no_grad():
            for name, value in model.unpack(theta0).items():
                dict(model.named_parameters())[name].copy_(value)
        optimizer = torch.optim.LBFGS(model.parameters(), lr=lr, max_iter=20, tolerance_grad=tol,
                                      tolerance_change=tol, history_size=10, line_search_fn="strong_wolfe")

        def closure():
            optimizer.zero_grad()
            loss = model.loss(y, X, W, sf)
            loss.backward()
            return loss

        previous, converged = None, False
        for _ in range(max_iter):
            loss = optimizer.step(closure).item()
            if not np.isfinite(loss):
                break
            if previous is not None and abs(loss - previous) / (abs(previous) + 1e-10) < tol:
                converged = True
                break
            previous = loss
        return model.pack(), converged

    def _compute_hessians(self, derivatives="analytic"):
        """Cholesky factor of the negative Hessian of the log posterior at the fitted parameters, per gene.
        Genes whose negative Hessian is not positive definite (typically non-converged fits) are marked
        not converged and get an identity placeholder."""
        Y, X, W, sf, mu_bar = self._tensors()
        theta = self.results_["params"]
        template = self._build_model(mu_bar[0])
        if derivatives == "analytic":
            H = self._log_posterior_terms(theta, Y.T.contiguous(), mu_bar, X, W, sf, template, want_grad=False)[2]
        else:
            H = vmap(hessian(self._make_log_posterior_fn(template)), in_dims=(0, 0, 0, None, None, None))(
                theta, Y.T.contiguous(), mu_bar, X, W, sf)
        H = 0.5 * (H + H.transpose(-1, -2))
        eye = torch.eye(H.shape[-1], dtype=H.dtype)
        L, info = torch.linalg.cholesky_ex(-H + 1e-6 * eye)
        bad = info != 0
        if bad.any():
            warnings.warn(f"Negative Hessian not positive definite for {int(bad.sum())} gene(s); "
                          "marking them as not converged.")
            self.results_["converged"][bad] = False
            L[bad] = eye
        self.results_["hess_L"] = L

    # ------------------------------------------------------------------ inference and output

    def results(self, factor_name, reference_level, test_level, robust=False):
        """Wald test of test_level vs reference_level for a factor of the mean model.

        Returns a DataFrame indexed by gene with estimate (log fold change), se, z, pval, padj (BH) and
        converged; non-converged genes are NaN. robust=True replaces the model-based covariance H^-1 by the
        HC1-scaled sandwich N/(N-P) H^-1 (sum_i s_i s_i') H^-1, which keeps the test calibrated when the NB
        variance function is misspecified (heavier tails, unmodelled sample heterogeneity).
        """
        assert self.results_ is not None, "call fit() first"
        names = self.dataset.covariate_names
        ref_col, test_col = f"{factor_name}[T.{reference_level}]", f"{factor_name}[T.{test_level}]"
        if ref_col not in names and test_col not in names:
            raise ValueError(f"'{ref_col}' and '{test_col}' not found. Available: {names}")
        contrast = torch.zeros(len(names), dtype=self.fit_dtype)
        if ref_col in names:
            contrast[names.index(ref_col)] = -1.0
        if test_col in names:
            contrast[names.index(test_col)] = 1.0

        params, L = self.results_["params"], self.results_["hess_L"]
        n_genes, n_beta = params.shape[0], len(names)
        converged = self.results_["converged"].numpy()

        cov = torch.cholesky_solve(torch.eye(self.n_parameters, dtype=L.dtype), L)
        if robust:
            Y, X, W, sf, mu_bar = self._tensors()
            S = self._score_outer_products(params, Y.T.contiguous(), mu_bar, X, W, sf)
            n_samples, n_mean_cov = X.shape
            cov = (n_samples / (n_samples - n_mean_cov)) * (cov @ S @ cov)

        out = {k: np.full(n_genes, np.nan) for k in ["estimate", "se", "z", "pval", "padj"]}
        out["estimate"][converged] = (params[converged, :n_beta] @ contrast).numpy()
        out["se"][converged] = torch.sqrt(
            torch.einsum("i,nij,j->n", contrast, cov[converged, :n_beta, :n_beta], contrast)).numpy()
        out["z"][converged] = out["estimate"][converged] / out["se"][converged]
        out["pval"][converged] = 2 * ss.norm.cdf(-np.abs(out["z"][converged]))
        out["padj"][converged] = ss.false_discovery_control(out["pval"][converged], method="bh")
        out["converged"] = converged
        return pd.DataFrame(out, index=self.dataset.var_names[:n_genes])

    def _store_results(self):
        """Write fitted parameters and the Hessian factor into the dataset's AnnData (varm / var)."""
        assert self.results_ is not None, "call fit() first"
        params = self.results_["params"].numpy()
        n_genes, n_beta = params.shape[0], self.dataset.mean_covariate_count
        adata = self.dataset.adata
        adata.varm["beta"] = params[:, :n_beta]
        adata.varm["a"] = params[:, n_beta].reshape(-1, 1)
        adata.varm["b"] = params[:, n_beta + 1].reshape(-1, 1)
        if self.dataset.dispersion_covariate_count > 0:
            adata.varm["gamma"] = params[:, n_beta + 2:]
        adata.varm["hess_L"] = self.results_["hess_L"].numpy().reshape(n_genes, -1)
        adata.var["converged"] = self.results_["converged"].numpy()

    def save(self, path):
        self.dataset.adata.write_h5ad(path)
