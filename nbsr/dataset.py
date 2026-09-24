import anndata as ad
import numpy as np
import pandas as pd
import patsy
import torch

from pydeseq2.dds import DeseqDataSet
from pydeseq2.default_inference import DefaultInference

class Dataset:

    def __init__(self, counts, metadata, mean_formula, disp_formula=None, n_cpus=4, dtype=torch.float32):
        """
        counts: N x K count matrix (pandas DataFrame)
        metadata: N x P metadata (pandas DataFrame)
        mean_formula: formula for the mean model (e.g. "~ x1 + x2")
        disp_formula: formula for the dispersion model
        dtype: data type for the tensors
        """

        # TODO: add checks for the input data 
        # check that the number of samples in counts and metadata match
        # check that the covariates specified in the formulas are present in the metadata
        # check that the counts are non-negative integers etc.

        self.dtype = dtype

        self.adata = ad.AnnData(X=counts.values,
                                obs=metadata,
                                var=pd.DataFrame(index=counts.columns))
        self.adata.obs_names = counts.index.tolist()
        self.adata.var_names = counts.columns.tolist()
        self.adata.uns["mean_formula"] = mean_formula
        self.adata.uns["disp_formula"] = disp_formula

        # Store the design matrix for mean model in adata.obsm["X"] and the design matrix for dispersion model in adata.obsm["W"].
        self.adata.obsm["X"] = patsy.dmatrix(mean_formula, self.adata.obs, return_type="dataframe")
        if disp_formula is not None and len(disp_formula) > 0:
            # Default is it will add an intercept term to the design matrix.
            W_df = patsy.dmatrix(disp_formula, self.adata.obs, return_type="dataframe")
            # Remove the intercept column from the design matrix for dispersion model since we will have a separate intercept parameter for the dispersion model.
            self.adata.obsm["W"] = W_df.iloc[:,1:]

        inference = DefaultInference(n_cpus=n_cpus)
        dds = DeseqDataSet(
            counts=counts,
            metadata=metadata,
            design=mean_formula,
            refit_cooks=True,
            inference=inference,
        )
        dds.fit_size_factors()
        dds.fit_genewise_dispersions()
        dds.fit_dispersion_trend()
        dds.fit_dispersion_prior()
        #dds.fit_MAP_dispersions()
        a0, a1 = dds.uns["trend_coeffs"]
        disp_prior_var = dds.uns['prior_disp_var']
        print(f"DESeq2 dispersion trend coefficients: a0={a0}, a1={a1}, prior variance={disp_prior_var}")

        self.adata.obs["size_factors"] = dds.obs["size_factors"]
        self.adata.uns["a0"] = a0
        self.adata.uns["a1"] = a1
        self.adata.uns["dispersion_prior_var"] = disp_prior_var

        self.adata.layers["normalized_counts"] = self.adata.X / self.adata.obs["size_factors"].values[:, np.newaxis]
        self.adata.var["mu_bar"] = self.adata.layers["normalized_counts"].mean(axis=0)

    @classmethod
    def from_adata(cls, adata, dtype=torch.float32):
        """Reconstruct Dataset from a saved AnnData without re-fitting."""
        obj = object.__new__(cls)
        obj.adata = adata
        obj.dtype = dtype
        return obj
    
    @property
    def a0(self):
        return self.adata.uns["a0"]
    
    @property
    def a1(self):
        return self.adata.uns["a1"]
    
    @property
    def dispersion_prior_var(self):
        return self.adata.uns["dispersion_prior_var"]
    
    @property
    def var_names(self):
        return self.adata.var_names

    @property
    def gene_count(self):
        return self.adata.n_vars
    
    @property
    def sample_count(self):
        return self.adata.n_obs
    
    @property
    def mean_covariate_count(self):
        return self.adata.obsm["X"].shape[1]
    
    @property
    def dispersion_covariate_count(self):
        return self.adata.obsm["W"].shape[1] if "W" in self.adata.obsm else 0
    
    @property
    def Y(self):
        return torch.tensor(self.adata.X, dtype=self.dtype)
    
    @property
    def X(self):
        return torch.tensor(self.adata.obsm["X"].values, dtype=self.dtype)
    
    @property
    def W(self):
        return torch.tensor(self.adata.obsm["W"].values, dtype=self.dtype) if "W" in self.adata.obsm else None

    @property
    def mu_bar(self):
        return torch.tensor(self.adata.var["mu_bar"].values, dtype=self.dtype)

    @property
    def size_factors(self):
        return torch.tensor(self.adata.obs["size_factors"].values, dtype=self.dtype)

    @property
    def covariate_names(self):
        return self.adata.obsm["X"].columns.tolist()