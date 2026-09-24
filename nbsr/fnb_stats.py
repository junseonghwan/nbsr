from typing import Optional
from joblib import Parallel, delayed

import numpy as np
import pandas as pd
import scipy.stats as ss
import torch
from torch.func import hessian, vmap, functional_call
from tqdm import tqdm

from nbsr.dataset import Dataset
from nbsr.distributions import log_negbinomial, log_normal
from nbsr.dispersion import LogDispersionTrendPrior, MeanPowerCovariateDispersion
from nbsr.featurewise_nb_model import FeaturewiseNegBinom

class FeaturewiseNBStats:
    # Prior sd on the mean-model intercept. The intercept is the log reference-group mean, which is always
    # well identified by the counts, so the prior is made effectively flat rather than pulling it toward 0.
    INTERCEPT_PRIOR_SD = 1e3

    def __init__(self, 
                 dataset : Dataset,
                 b_prior_sd=0.1,
                 beta_prior_sd=5.0):
        """
        b_prior_sd: prior sd on b_j, the within-gene slope of log dispersion on log mean.
        beta_prior_sd: prior sd on the non-intercept mean-model coefficients (natural-log scale).
            The default is weak enough to be flat for any identifiable log fold change; its only role is to
            keep the MAP finite for separated genes (one group all zeros). Pass a smaller value for deliberate
            shrinkage of the log fold changes.
        """
        self.dataset = dataset
        self.dispersion_prior = LogDispersionTrendPrior(self.dataset.a0, self.dataset.a1, self.dataset.dispersion_prior_var)        
        self.b_prior_sd = b_prior_sd
        self.beta_prior_sd = beta_prior_sd
        self.results_ = None
        # Per-gene problems are tiny, so fit and differentiate in float64 regardless of the dataset dtype.
        # In float32 L-BFGS stalls with gradient norms ~10 (loss ~1e2-1e6 vs relative tolerance 1e-6),
        # leaving some genes short of the optimum with an indefinite Hessian.
        self.fit_dtype = torch.float64

    def _beta_prior_sd_vector(self):
        """Per-covariate prior sd for beta: flat on the intercept, beta_prior_sd elsewhere."""
        sd = torch.full((self.dataset.mean_covariate_count,), float(self.beta_prior_sd), dtype=self.fit_dtype)
        if "Intercept" in self.dataset.covariate_names:
            sd[self.dataset.covariate_names.index("Intercept")] = self.INTERCEPT_PRIOR_SD
        return sd

    def _initial_beta(self, y, X, sf, mu_bar):
        """Least-squares fit of log normalized counts on X, the usual GLM warm start.

        Starting from beta = 0 (mu = size factor), or from the intercept alone at log(mu_bar), leaves genes
        with a large fold change far from the optimum: mu is off by orders of magnitude for one group, and
        with a weak beta prior the first L-BFGS steps overshoot and the loss goes NaN. One least-squares pass
        on log((y + 0.5) / sf) lands every coefficient within a few tenths of the MAP.
        """
        if y is None:
            beta = torch.zeros(self.dataset.mean_covariate_count, dtype=self.fit_dtype)
            if "Intercept" in self.dataset.covariate_names:
                beta[self.dataset.covariate_names.index("Intercept")] = torch.log(
                    torch.as_tensor(mu_bar, dtype=self.fit_dtype))
            return beta
        z = torch.log((y + 0.5) / sf)
        return torch.linalg.lstsq(X, z.unsqueeze(-1)).solution.squeeze(-1)

    def _build_model(self, mu_bar, y=None, X=None, sf=None):
        """Build a fresh model for a single gene, initialised near the optimum.

        When y, X and sf are given, beta starts at the least-squares warm start; otherwise only the
        intercept is set (to log mu_bar), which is enough for a template model whose values are not used.
        """
        n_mean_cov = self.dataset.mean_covariate_count
        n_disp_cov = self.dataset.dispersion_covariate_count
        disp_model = MeanPowerCovariateDispersion(
            n_disp_cov,
            self.dispersion_prior,
            mu_bar,
            self.b_prior_sd
        )
        model = FeaturewiseNegBinom(n_mean_cov, disp_model, self._beta_prior_sd_vector()).to(self.fit_dtype)
        with torch.no_grad():
            model.beta.copy_(self._initial_beta(y, X, sf, mu_bar))
        return model
    
    def fit_gene_LBFGS(
            self,
            y_np: np.ndarray,
            X_np: np.ndarray,
            W_np: Optional[np.ndarray],
            sf_np: np.ndarray,
            mu_bar_np: np.ndarray,
            n_iter: int = 100,
            lr: float = 1.0,
            tol: float = 1e-6,
            print_every: int = 0,
        ):

        dtype = self.fit_dtype

        y = torch.as_tensor(y_np, dtype=dtype)
        X = torch.as_tensor(X_np, dtype=dtype)
        W = torch.as_tensor(W_np, dtype=dtype) if W_np is not None else None
        sf = torch.as_tensor(sf_np, dtype=dtype)
        mu_bar = torch.as_tensor(mu_bar_np, dtype=dtype)

        model = self._build_model(mu_bar, y, X, sf)
        optimizer = torch.optim.LBFGS(
            model.parameters(),
            lr=lr,
            max_iter=20,          # inner iterations per step
            tolerance_grad=tol,
            tolerance_change=tol,
            history_size=10,
            line_search_fn="strong_wolfe"  # important for stability
        )

        loss_history = []
        converged    = False

        model.train()
        def closure():
            optimizer.zero_grad()
            loss = model.loss(y, X, W, sf)
            loss.backward()
            return loss

        for i in range(n_iter):
            loss = optimizer.step(closure)
            loss_val = loss.item()
            loss_history.append(loss_val)

            if print_every and i % print_every == 0:
                print(f"iter {i:5d} | loss {loss_val:.4f}")

            if not np.isfinite(loss_val):
                break  # diverged; leave converged = False.

            if i > 0:
                rel_change = abs(loss_val - loss_history[-2]) / (abs(loss_history[-2]) + 1e-10)
                if rel_change < tol:
                    converged = True
                    break

        model.eval()
        return model, loss_history, converged

    def fit_gene_ADAM(
            self,
            y: torch.tensor,
            X: torch.tensor,
            W: Optional[torch.tensor],
            size_factors: torch.tensor,
            mu_bar: torch.tensor,
            n_iter: int = 20000,
            lr: float = 5e-1,
            tol: float = 1e-6,
            patience: int = 50,
            print_every: int = 100,
        ):
        """
        Fit FeaturewiseNegBinom via Adam.

        Parameters
        ----------
        Y : (N, K) matrix of counts
        X : (N, P) mean model covariates
        W : (N, D) dispersion model covariates (or None)
        size_factors : (N,)
        tol : convergence tolerance on relative loss change
        patience : stop if no improvement for this many iterations
        """
        model = self._build_model(mu_bar)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        
        best_loss = torch.inf
        no_improve = 0
        loss_history = []
        converged = False

        model.train()
        for i in range(n_iter):
            optimizer.zero_grad()
            loss = model.loss(y, X, W, size_factors)
            loss.backward()
            optimizer.step()

            loss_val = loss.item()
            loss_history.append(loss_val)

            if print_every and i % print_every == 0:
                print(f"iter {i:5d} | loss {loss_val:.4f}")

            # Convergence check
            rel_change = abs(loss_val - best_loss) / (abs(best_loss) + 1e-10)
            if loss_val < best_loss:
                best_loss = loss_val
                no_improve = 0
            else:
                no_improve += 1

            if rel_change < tol or no_improve >= patience:
                print(f"Converged at iter {i} | loss {loss_val:.4f}")
                converged = True
                break

        model.eval()
        return model, loss_history, converged
    
    def _make_log_posterior_fn(self, model):
        def log_posterior_fn(theta, y, mu_bar, X, W, sf):
            params = model.unpack(theta)
            params["dispersion_model.mu_bar"] = mu_bar
            # functional_call injects params into model.forward without mutating state
            # submodule params use dot notation e.g. "dispersion_model.a" -- which model.unpack does automatically.
            mu, log_phi = functional_call(model, params, (X, W, sf))
            phi = torch.exp(log_phi)

            log_lik = log_negbinomial(y, mu, phi).sum()
            log_prior_beta = log_normal(params["beta"],
                                        torch.zeros_like(params["beta"]),
                                        model.beta_prior_sd).sum()
            log_prior_dispersion = model.dispersion_model.compute_log_prior(
                a=params["dispersion_model.a"],
                b=params["dispersion_model.b"],
                mu_bar=mu_bar,
                gamma=params.get("dispersion_model.gamma"))
            return log_lik + log_prior_beta + log_prior_dispersion
        return log_posterior_fn

    def _compute_hessians(self):
        dtype  = self.fit_dtype
        Y      = self.dataset.Y.to(dtype)
        X      = self.dataset.X.to(dtype)
        W      = self.dataset.W.to(dtype) if self.dataset.W is not None else None
        sf     = self.dataset.size_factors.to(dtype)
        mu_bar = self.dataset.mu_bar.to(dtype)

        assert self.results_ is not None

        n_genes = len(self.results_["models"])

        # template_model is used just for computing the Hessian. 
        template_model = self._build_model(mu_bar[0])  # any gene works: mu_bar is overridden per gene in log_post.
        log_post = self._make_log_posterior_fn(template_model)

        # vmap will batch the computation so that it's faster.
        # in_dims indicates the dimension to loop over for each argument in the batch.
        # None if fixed parameter.
        H_all = vmap(hessian(log_post, argnums=0),
                     in_dims=(0, 0, 0, None, None, None))(
            self.results_["params"],  # (n_genes, p)
            Y.T.contiguous(),         # (n_genes, n_samples)
            mu_bar,
            X,
            W,
            sf
        )                             # -> (n_genes, p, p)

        H_all = 0.5 * (H_all + H_all.transpose(-1, -2))
        Hn    = -H_all + 1e-6 * torch.eye(H_all.shape[-1], dtype=H_all.dtype)
        # cholesky_ex returns info != 0 for genes whose negative Hessian is not positive definite
        # (typically non-converged fits) instead of raising for the whole batch.
        L_all, info = torch.linalg.cholesky_ex(Hn)  # batched, (n_genes, p, p)
        bad = info != 0
        if bad.any():
            print(f"Negative Hessian not positive definite for {int(bad.sum())} gene(s); marking them as not converged.")
            self.results_["converged"][bad] = False
            L_all[bad] = torch.eye(H_all.shape[-1], dtype=H_all.dtype)  # placeholder so cholesky_solve stays well defined; masked downstream.
        self.results_["hess_L"] = L_all

    def fit(self, 
            n_cpus : int = 1, 
            max_iter : int = 100, 
            lr : float = 1.0, 
            print_iter : int = 0):
        # Fit for each gene.
        n_genes = self.dataset.gene_count
        n_mean_covariates = self.dataset.mean_covariate_count
        n_disp_covariates = self.dataset.dispersion_covariate_count
        n_parameters = n_mean_covariates + 2 + n_disp_covariates # +2 for intercept term and the slope for log \mu_{ij} in the dispersion model.

        Y = self.dataset.Y.numpy()
        X = self.dataset.X.numpy()
        W = self.dataset.W.numpy() if self.dataset.W is not None else None
        sf = self.dataset.size_factors.numpy()
        mu_bar = self.dataset.mu_bar.numpy()

        self.results_ = {
            "models": [None] * n_genes,
            "params": torch.zeros(n_genes, n_parameters, dtype=self.fit_dtype),
            "converged": torch.zeros(n_genes, dtype=torch.bool),
        }

        # loky returns results in submission order, so gene_idx == position in job_results.
        job_results = list(tqdm(
            Parallel(n_jobs=n_cpus, backend="loky", batch_size="auto", return_as="generator")(
                delayed(self.fit_gene_LBFGS)(
                    Y[:, i], X, W, sf, mu_bar[i],
                    n_iter=int(max_iter), lr=float(lr), print_every=int(print_iter))
                for i in range(n_genes)
            ),
            total=n_genes,
            desc="Fitting genes"
        ))

        for gene_idx, (model, loss_history, converged) in enumerate(job_results):
            self.results_["models"][gene_idx]    = model
            self.results_["params"][gene_idx]    = model.pack()
            self.results_["converged"][gene_idx] = converged



        # Compute the Hessian matrix over the mean and dispersion parameters and save it to file.
        self._compute_hessians() # this will set self.results_["hess_L"].

    def results(self, factor_name, reference_level, test_level):

        assert self.results_ is not None

        # Look up the factor_name in the dataset and retrieve the corresponding column index.
        # patsy column names are 
        ref_column_name = f"{factor_name}[T.{reference_level}]"
        test_column_name = f"{factor_name}[T.{test_level}]"
        covariate_names = self.dataset.covariate_names

        # If none of the column names are in covariate_names, print error.
        if ref_column_name not in covariate_names and test_column_name not in covariate_names:
            raise ValueError(f"'{ref_column_name}' and '{test_column_name}' not found. Available: {covariate_names}")

        # covariate_count includes the intercept term.
        contrast = torch.zeros(self.dataset.mean_covariate_count, dtype=self.fit_dtype)

        if ref_column_name in covariate_names:
            contrast[covariate_names.index(ref_column_name)] = -1.0
        if test_column_name in covariate_names:
            contrast[covariate_names.index(test_column_name)] = 1.0

        n_parameters = self.dataset.mean_covariate_count + 2 + self.dataset.dispersion_covariate_count
        L_all = self.results_["hess_L"]

        n_genes   = len(self.results_["models"])
        n_beta    = self.dataset.mean_covariate_count
        converged = self.results_["converged"].numpy()

        estimates = np.full(n_genes, np.nan)
        ses       = np.full(n_genes, np.nan)
        zs        = np.full(n_genes, np.nan)
        pvals     = np.full(n_genes, np.nan)
        padjs     = np.full(n_genes, np.nan)

        beta_all  = torch.stack([m.beta.detach() for m in self.results_["models"]])
        cov_all   = torch.cholesky_solve(torch.eye(n_parameters, dtype=L_all.dtype), L_all)

        estimates[converged] = (beta_all[converged] @ contrast).numpy()
        ses[converged]       = torch.sqrt(
            torch.einsum("i,nij,j->n", contrast, cov_all[converged, :n_beta, :n_beta], contrast)
        ).numpy()

        zs[converged]    = estimates[converged] / ses[converged]
        pvals[converged] = 2 * ss.norm.cdf(-np.abs(zs[converged]))
        padjs[converged]  = ss.false_discovery_control(pvals[converged], method="bh")

        return pd.DataFrame({
            "estimate": estimates,
            "se":       ses,
            "z":        zs,
            "pval":     pvals,
            "padj":     padjs,
            "converged": self.results_["converged"].numpy(),
        }, index=self.dataset.var_names[:n_genes])

    def _store_results(self):
        assert self.results_ is not None

        n_genes      = len(self.results_["models"])
        n_beta       = self.dataset.mean_covariate_count
        n_disp_cov   = self.dataset.dispersion_covariate_count
        #n_parameters = self.results_["params"].shape[1]

        # per-gene matrices -> varm
        self.dataset.adata.varm["beta"]   = self.results_["params"][:, :n_beta].numpy()
        self.dataset.adata.varm["a"]      = self.results_["params"][:, n_beta].numpy().reshape(-1, 1)
        self.dataset.adata.varm["b"]      = self.results_["params"][:, n_beta+1].numpy().reshape(-1, 1)
        self.dataset.adata.varm["hess_L"] = self.results_["hess_L"].numpy().reshape(n_genes, -1)  # flatten p x p
        if n_disp_cov > 0:
            self.dataset.adata.varm["gamma"] = self.results_["params"][:, n_beta+2:].numpy()

        # per-gene scalars -> var
        self.dataset.adata.var["converged"] = self.results_["converged"].numpy()

    def save(self, path):
        self.dataset.adata.write_h5ad(path)
