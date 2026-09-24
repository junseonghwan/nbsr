"""End-to-end check of the featurewise NB path: Dataset -> FeaturewiseNBStats.fit -> results.

Uses the first 40 features of data/test (20 samples, one two-level factor `trt`).
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from nbsr.dataset import Dataset
from nbsr.fnb_stats import FeaturewiseNBStats

DATA = Path(__file__).resolve().parents[1] / "data" / "test"
N_FEATURES = 40


@pytest.fixture(scope="module")
def dataset():
    # Y.csv is features x samples; Dataset wants samples x features.
    counts = pd.read_csv(DATA / "Y.csv", index_col=0).T.iloc[:, :N_FEATURES]
    # na_filter=False: pandas would otherwise read the level "null" as NaN.
    metadata = pd.read_csv(DATA / "X.csv", index_col="sample", na_filter=False).loc[counts.index]
    metadata["trt"] = pd.Categorical(metadata["trt"], categories=["null", "alt"])
    return Dataset(counts, metadata, mean_formula="~trt", n_cpus=1)


def test_dataset_shapes(dataset):
    assert dataset.gene_count == N_FEATURES
    assert dataset.sample_count == 20
    assert dataset.covariate_names == ["Intercept", "trt[T.alt]"]
    assert dataset.dispersion_covariate_count == 0
    assert torch.all(dataset.size_factors > 0)


def test_fit_and_results(dataset):
    stats = FeaturewiseNBStats(dataset, b_prior_sd=0.5)
    stats.fit(n_cpus=2, max_iter=100)

    n_params = dataset.mean_covariate_count + 2  # beta, a, b
    assert stats.results_["params"].shape == (N_FEATURES, n_params)
    assert stats.results_["hess_L"].shape == (N_FEATURES, n_params, n_params)

    converged = stats.results_["converged"].numpy()
    # The argument-order bug made this exactly 0; a healthy run converges on nearly every gene.
    assert converged.all(), f"only {converged.sum()}/{N_FEATURES} genes converged"

    res = stats.results("trt", reference_level="null", test_level="alt")
    assert list(res.index) == list(dataset.var_names)
    ok = res["converged"].to_numpy()
    for col in ["estimate", "se", "z", "pval", "padj"]:
        assert np.all(np.isfinite(res.loc[ok, col])), f"non-finite {col} for converged genes"
    assert np.all(res.loc[ok, "se"] > 0)
    assert np.all((res.loc[ok, "pval"] >= 0) & (res.loc[ok, "pval"] <= 1))
    assert np.all((res.loc[ok, "padj"] >= 0) & (res.loc[ok, "padj"] <= 1))

    # estimate is the log fold change alt vs null of beta_trt, i.e. the coefficient itself here.
    beta_trt = torch.stack([m.beta.detach() for m in stats.results_["models"]])[:, 1].numpy()
    np.testing.assert_allclose(res.loc[ok, "estimate"], beta_trt[ok], rtol=1e-6)


def test_beta_prior_is_flat_on_intercept(dataset):
    stats = FeaturewiseNBStats(dataset, b_prior_sd=0.5, beta_prior_sd=5.0)
    model = stats._build_model(dataset.mu_bar[0])
    sd = model.beta_prior_sd
    assert sd.shape == (dataset.mean_covariate_count,)
    assert sd[dataset.covariate_names.index("Intercept")] == FeaturewiseNBStats.INTERCEPT_PRIOR_SD
    assert sd[dataset.covariate_names.index("trt[T.alt]")] == 5.0
    # The prior is elementwise, so the log prior must broadcast against the vector sd.
    assert torch.isfinite(model.log_prior())


def test_scalar_beta_prior_sd_is_broadcast():
    from nbsr.dispersion import LogDispersionTrendPrior, MeanPowerCovariateDispersion
    from nbsr.featurewise_nb_model import FeaturewiseNegBinom
    prior = LogDispersionTrendPrior(0.1, 1.0, 0.5)
    disp = MeanPowerCovariateDispersion(0, prior, torch.tensor(10.0))
    model = FeaturewiseNegBinom(3, disp, beta_prior_sd=2.0)
    assert torch.equal(model.beta_prior_sd, torch.full((3,), 2.0))
    with pytest.raises(AssertionError):
        FeaturewiseNegBinom(3, disp, beta_prior_sd=torch.ones(2))
