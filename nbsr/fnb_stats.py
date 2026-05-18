from typing import Optional

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
    def __init__(self, 
                 dataset : Dataset,
                 b_prior_sd=0.1):
        self.dataset = dataset
        self.dispersion_prior = LogDispersionTrendPrior(self.dataset.a0, self.dataset.a1, self.dataset.dispersion_prior_var)        
        self.b_prior_sd = b_prior_sd
        self.results_ = None

    def _build_model(self, mu_bar):
        """Build a fresh model for a single gene."""
        n_mean_cov = self.dataset.mean_covariate_count
        n_disp_cov = self.dataset.dispersion_covariate_count
        disp_model = MeanPowerCovariateDispersion(
            n_disp_cov,
            self.dispersion_prior,
            mu_bar,
            self.b_prior_sd
        )
        return FeaturewiseNegBinom(n_mean_cov, disp_model)

    def fit_gene(
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
        # n_mean_covariates = X.shape[1]
        # n_disp_covariates = W.shape[1] if W is not None else 0
        # disp_model = MeanPowerCovariateDispersion(
        #     n_disp_covariates, 
        #     self.dispersion_prior, 
        #     mu_bar, 
        #     self.b_prior_sd)
        # model = FeaturewiseNegBinom(n_mean_covariates, disp_model)
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
        Y      = self.dataset.Y
        X      = self.dataset.X
        W      = self.dataset.W
        sf     = self.dataset.size_factors
        mu_bar = self.dataset.mu_bar

        assert self.results_ is not None

        n_genes = len(self.results_["models"])

        # template_model is used just for computing the Hessian. 
        template_model = self._build_model() 
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
        Hn    = -H_all + 1e-6 * torch.eye(H_all.shape[-1])
        L_all = torch.linalg.cholesky(Hn)  # batched, (n_genes, p, p)
        self.results_["hess_L"] = L_all

    def fit(self, max_iter, lr=1e-2):
        # Fit for each gene.
        n_genes = self.dataset.gene_count
        n_mean_covariates = self.dataset.mean_covariate_count
        n_disp_covariates = self.dataset.dispersion_covariate_count
        n_parameters = n_mean_covariates + 2 + n_disp_covariates # +2 for intercept term and the slope for log \mu_{ij} in the dispersion model.

        Y = self.dataset.Y
        X = self.dataset.X
        W = self.dataset.W
        sf = self.dataset.size_factors
        mu_bar = self.dataset.mu_bar

        self.results_ = {
            "models": [None] * n_genes,
            "params": torch.zeros(n_genes, n_parameters),
            "converged": torch.zeros(n_genes, dtype=torch.bool),
        }

        for gene_idx in tqdm(range(n_genes)):
            y = Y[:, gene_idx]
            
            model, loss_history, converged = self.fit_gene(
                y=y,
                X=X,
                W=W, # W_torch is None if no dispersion covariates are specified.
                size_factors=sf,
                mu_bar=mu_bar[gene_idx],
                n_iter=max_iter,
                lr=lr,
                print_every=1000)
            
            # Save the model.
            with torch.no_grad():
                self.results_["models"][gene_idx] = model
                self.results_["params"][gene_idx] = model.pack()
                self.results_["converged"][gene_idx]  = converged


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
        contrast = torch.zeros(self.dataset.mean_covariate_count)

        if ref_column_name in covariate_names:
            contrast[covariate_names.index(ref_column_name)] = -1.0
        if test_column_name in covariate_names:
            contrast[covariate_names.index(test_column_name)] = 1.0

        n_parameters = self.dataset.mean_covariate_count + 2 + self.dataset.dispersion_covariate_count
        L_all = self.results_["hess_L"]
        cov_all = torch.cholesky_solve(torch.eye(n_parameters), L_all)

        n_genes   = len(self.results_["models"])
        n_beta    = self.dataset.mean_covariate_count
        converged = self.results_["converged"].numpy()

        estimates = np.full(n_genes, np.nan)
        ses       = np.full(n_genes, np.nan)
        zs        = np.full(n_genes, np.nan)
        pvals     = np.full(n_genes, np.nan)
        padjs     = np.full(n_genes, np.nan)

        beta_all  = torch.stack([m.beta.detach() for m in self.results_["models"]])
        cov_all   = torch.cholesky_solve(torch.eye(n_parameters), L_all)

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
