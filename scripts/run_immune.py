"""Fit the featurewise NB model to the immune data (data/immune_data/dataset.h5ad) and write results.

Run from the repository root: python scripts/run_immune.py
"""
from pathlib import Path

import anndata as ad
import pandas as pd
import torch

from nbsr.dataset import Dataset
from nbsr.fnb_stats import FeaturewiseNBStats

X = pd.read_csv("data/immune_data/X.csv", index_col="sample_name")
Y = pd.read_csv("data/immune_data/Y.csv")

Y.columns = Y.columns.astype(str)
X["trt"] = pd.Categorical(X["trt"], categories=["Pre", "On"])

outpath = Path("data/immune_data/fnb_run/")
mean_formula = "~trt"
disp_formula = "~trt"

adata = ad.read_h5ad("data/immune_data/dataset.h5ad")
ds = Dataset.from_adata(adata, torch.float32)

n_cpus = 4
b_prior_sd = 0.5
fnb_stats = FeaturewiseNBStats(ds, b_prior_sd)
fnb_stats.fit()

factor_name = "trt"
reference_level = "Pre"
test_level = "On"
results_df = fnb_stats.results(factor_name, reference_level, test_level)
results_df.to_csv(outpath / "results.csv")
fnb_stats._store_results()
fnb_stats.save(outpath / "ds_fit.h5ad")


