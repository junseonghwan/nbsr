# NBSR: Negative Binomial Softmax Regression

NBSR is a compositional model for sequencing counts (microRNA, microbiome, surface proteins, any assay whose
counts sum to a library size). The composition `pi_i = softmax(x_i' beta)` of a sample is a function of its
covariates, and each count is negative binomial with mean `s_i pi_ij` and its own dispersion. Inference is on
log relative-abundance ratios between covariate levels, with standard errors from the Hessian of the log
posterior at the mode. The package also contains a feature-wise NB regression (`nbsr.fnb_stats`) used for
method comparisons.

Paper: https://doi.org/10.1093/biostatistics/kxag012

## Installation

```bash
git clone https://github.com/junseonghwan/nbsr.git
cd nbsr
python -m pip install -e .      # -e for development
pytest tests                    # 43 tests
```

`requirements.txt` pins a set of versions known to work together.

## Input files

A data directory with

- `Y.csv`: counts, one row per feature and one column per sample (first column holds the feature names).
- `X.csv`: sample metadata, one row per sample in the same order as the columns of `Y.csv`, with a header
  naming the covariates. Categorical covariates are dummy-coded with the first level (alphabetical) as the
  reference.

## Models

`NegativeBinomialRegressionModel` (`nbsr/negbinomial_model.py`) has one dispersion per feature, either free
(with an optional log-normal prior around the dispersion model) or fixed from a `dispersion.csv` file.

`NBSRTrended` (`nbsr/nbsr_dispersion.py`) ties the dispersion to the fitted composition through the
dispersion model of `nbsr/dispersion.py`:

    log phi_ij = b_0 + b_j + b_pi * logit(pi_ij) + w_i' b_w

with a per-feature offset `b_j = sigma_bj z_j` (hierarchical scale, half-normal prior on `sigma_bj`),
`b_pi ~ N(1, 0.1)` and external per-sample covariates `W` taken from columns of `X.csv`. This is the model of
the NBSR-HMC Stan implementation; the CLI fits its posterior mode.

Both models take `--pivot`, which fixes the last feature's coefficients at zero (reference category, as in the
Stan model). Without it every feature has coefficients and the normal prior on `beta` resolves the softmax
invariance.

Gradients and Hessians of the log posterior are in closed form (`utils.kron_hessian`), so the Hessian of a
model with a few hundred features takes well under a second; it is a dense `(P * J)^2` matrix, so the
approach targets panels of hundreds to a few thousand features.

## Fitting

```
python nbsr/main.py train /path/to/data var1 var2 [options]
```

Fits the posterior mode by Adam, writes `nbsr_beta.csv`, `nbsr_beta_sd.csv`, `nbsr_pi.csv`,
`nbsr_dispersion.csv`, the dispersion model parameters (`nbsr_dispersion_params.csv`, `nbsr_dispersion_bj.csv`,
`nbsr_dispersion_bw.csv`), `checkpoint.pth`, `config.json` and `hessian.npy` into `/path/to/data/runK` for
each run, and copies the best run to `/path/to/data`.

| option | default | meaning |
|---|---|---|
| `-i, --iterations` | 10000 | Adam iterations |
| `-l, --lr` | 0.05 | learning rate (values above 0.2 are rarely stable) |
| `-r, --runs` | 1 | independent initialisations; the best log posterior is kept |
| `--trended_dispersion` | off | use `NBSRTrended` (recommended from about 10 samples per condition) |
| `--z_columns NAME` | none | columns of `X.csv` used as external dispersion covariates `W` (repeat the flag for several) |
| `--z_log` | off | log-transform the `--z_columns` first (e.g. library sizes, capture rates) |
| `--dispersion_model_file FILE` | none | load a fitted dispersion model (`disp_model.pth` from `eb`) and hold it fixed |
| `--estimate_dispersion_sd` | off | estimate a per-feature sd for the log-normal dispersion prior |
| `--beta_prior_sd SD [SD ...]` | empirical | fix the prior sd of `beta`, one value or one per covariate (intercept first) |
| `--stage1_prior_sd`, `--stage1_iterations`, `--prior_quantile` | 10, half of `-i`, 0.95 | settings of the empirical prior's stage-1 fit and quantile matching |
| `--pivot` | off | reference-category parameterisation |

A `dispersion.csv` in the data directory (one value per feature) switches to fixed dispersions and ignores the
dispersion options.

### Prior on the coefficients

`beta ~ N(0, sd_d)` with one fixed sd per covariate. By default the sd is set empirically, DESeq2-style:
a stage-1 fit with a wide prior (`--stage1_prior_sd`, default 10, at half the iterations, written to
`stage1/`), then for each covariate the sd of a zero-centred normal whose upper tail matches the
precision-weighted `--prior_quantile` (default 0.95) of |beta| across features, then the main fit with those
sds fixed, warm-started from stage 1. The matched sds are printed, written to `nbsr_beta_sd.csv`, and stored in
`config.json`. `--beta_prior_sd` skips stage 1 and fixes the sd directly, as `sigma_beta2` given as data in
the Stan model.

The sd is deliberately not learned jointly with `beta`: the joint mode collapses to the spread of the null
coefficients (about 0.1 on the total-imbalance simulations, where nearly every feature is unchanged), which
over-shrinks the real effects and can be bimodal across runs.

### Small sample sizes: empirical Bayes

With few samples per condition, first obtain fitted means from DESeq2, then fit the dispersion model to
them, then fit NBSR with that dispersion model held fixed:

```
Rscript scripts/deseq2.R /path/to/data var1,var2      # writes deseq2_mu.csv
python nbsr/main.py eb /path/to/data var1 var2 [options]
```

`eb` takes the `train` options plus `-f, --mu_file` (default `deseq2_mu.csv`), `--eb_iter` and `--eb_lr` for
the dispersion-model fit, and `--update_dispersion` to keep optimising the dispersion model jointly with
`beta` in the second stage. It writes its outputs directly into `/path/to/data`.

## Inference

```
python nbsr/main.py results /path/to/data var1 level_numerator level_denominator
```

creates `/path/to/data/var1__level_numerator_vs_level_denominator/` containing

- `nbsr_results.h5` with `logRR`, `se`, `stat`, `pvalue`, `padj` (features x samples; identical across samples
  when `var1` is the only covariate),
- `nbsr_results.csv` with `log2FC`, `pvalue`, `padj` per feature when the contrast does not vary by sample,
- `covariance.pth`, the per-sample covariance of `logRR` across features (skip with `--skip_cov`).

Standard errors come from the Cholesky factor of the negative Hessian; `--recompute_hessian` recomputes it
from the checkpoint instead of reading `hessian.npy`. `--absolute_fc` subtracts the mode of the `logRR`
distribution (the compositional shift) before testing.

## Example

```
python nbsr/main.py train data/test trt -i 10000 --trended_dispersion
python nbsr/main.py results data/test trt alt null
```

`data/test` has 200 features and 20 samples with one two-level covariate `trt` (`null` vs `alt`), generated by
`scripts/generate_data.R`. `scripts/de.R` shows how the outputs are read in R for a differential-expression
analysis.

## Feature-wise NB regression

`nbsr.fnb_stats.FeaturewiseNBStats` fits, per feature, `log mu_ij = log s_i + x_i' beta_j` with
`log phi_ij = a_j + b_j log mu_ij + w_i' gamma_j`, all features at once by batched Newton with closed-form
derivatives, and tests contrasts with either the model-based or a sandwich (robust) standard error
(`results(..., robust=True)`). Build a `nbsr.dataset.Dataset` from a counts data frame and metadata with
patsy formulas for the mean and dispersion models, then `FeaturewiseNBStats(dataset).fit()`. It is
experimental and not part of the command line.

## Citation

If you use NBSR, please cite:

Jun S-H, Halushka MK, McCall MN. NBSR: a negative binomial softmax regression model for microRNA-seq data
analysis. *Biostatistics* 27(1): kxag012, 2026. https://doi.org/10.1093/biostatistics/kxag012

Code reproducing the paper's figures: https://github.com/junseonghwan/nbsr-experiments/
