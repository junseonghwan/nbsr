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

## Workflow

Two stages, run by two commands (or `train`, which runs both):

```
python nbsr/main.py prior DATA var1 [var2 ...] [options]    # stage 1 -> DATA/prior/prior.json
python nbsr/main.py fit   DATA var1 [var2 ...] --prior DATA/prior/prior.json [options]
python nbsr/main.py results DATA var1 level_numerator level_denominator [options]
```

**Stage 1, `prior`: empirical prior elicitation.** A diffuse fit (prior sd `--stage1_prior_sd`, default 10, on
every covariate, at `--stage1_iterations`, default half of `-i`) gives near-maximum-likelihood coefficients and
their standard errors; for each covariate the prior sd is then set so that a zero-centred normal's upper tail
matches the precision-weighted `--prior_quantile` (default 0.95) of |beta| across features (the DESeq2
recipe). `prior.json` records the sds, their squares as `sigma_beta2` for the Stan model, the fitted
dispersion-model parameters, and the path of the diffuse fit's checkpoint used as a warm start.
The sd is deliberately not learned jointly with `beta`: the joint mode collapses to the spread of the null
coefficients and over-shrinks the real effects.

With few samples per condition (3 to 5) the joint fit of the dispersion model is unstable. `--dispersion_from
deseq2` fits the dispersion model at the composition implied by PyDESeq2's fitted means (`--eb_iterations`,
`--eb_lr`), saves it as `prior/disp_model.pth`, and marks it in `prior.json` so that `fit` holds it fixed.

**Stage 2, `fit`: inference at fixed prior sds.** Takes the sds (and warm start, and fixed dispersion model
if any) from `--prior prior.json`, or fixed values from `--beta_prior_sd SD [SD ...]` (one value, or one per
covariate with the intercept first; `--beta_prior_sd 1` matches `sigma_beta2 = 1` in the Stan model). Writes
`nbsr_beta.csv`, `nbsr_beta_sd.csv` (the prior sds used), `nbsr_pi.csv`, `nbsr_dispersion.csv`, the
dispersion-model parameters (`nbsr_dispersion_params.csv`, `nbsr_dispersion_bj.csv`, `nbsr_dispersion_bw.csv`),
`checkpoint.pth`, `config.json` and `hessian.npy` into `DATA` (or `--out`). `-r N` runs N initialisations and
keeps the best log posterior.

Options shared by `prior`, `fit` and `train`:

| option | default | meaning |
|---|---|---|
| `-i, --iterations` | 10000 | Adam iterations |
| `-l, --lr` | 0.05 | learning rate (values above 0.2 are rarely stable) |
| `--trended_dispersion` / `--free_dispersion` | trended | dispersion model `b_0 + b_j + b_pi f(pi) + w'b_w`, or one free dispersion per feature |
| `--z_columns NAME` | none | columns of `X.csv` used as external dispersion covariates `W` (repeat for several) |
| `--z_log` | off | log-transform the `--z_columns` first (e.g. library sizes, capture rates) |
| `--z_total_counts` | off | add log total counts per sample as a dispersion covariate |
| `--dispersion_link` | logit | `f`: `logit` (NBSR-HMC) or `log` |
| `--no_feature_offsets` | off | drop the per-feature offsets `b_j` |
| `--b_pi_prior`, `--sigma_bj_prior_sd`, `--sigma_b` | (1, 0.1), 0.5, 1 | priors of the dispersion model |
| `--dispersion_model_file FILE` | none | load a fitted dispersion model and hold it fixed (`--update_dispersion` to optimise it) |
| `--pivot` | off | reference-category parameterisation (last feature's coefficients fixed at zero, as in the Stan model) |

The previous dispersion model `b0 + b1 log pi + b2 log R_i` is the configuration `--dispersion_link log
--no_feature_offsets --z_total_counts --b_pi_prior 0 0.1 --sigma_b 0.1`. A `dispersion.csv` in the data
directory (one value per feature) switches to fixed dispersions.

## Inference

```
python nbsr/main.py results DATA var1 level_numerator level_denominator [--ref_top_frac 0.1] [--laplace_draws 2000]
```

creates `DATA/var1__level_numerator_vs_level_denominator/` containing

- `contrast.csv`, one row per feature: the reference-free CLR contrast `delta_clr` (the coefficient
  difference between the two levels, centred across features, so it does not depend on the pivot) and its
  standard error from the Hessian; the compositional shift, the mode of `delta_clr` over the reference set
  (all features, or the top `--ref_top_frac` by pooled fitted proportion with a floor of `--ref_min`), and
  the absolute-abundance effect `delta_adj = delta_clr - shift` with Wald `z`, `pvalue`, BH `padj` and the
  posterior sign probability `ppos = Phi(z)`; with `--laplace_draws S`, draws of `beta` from the Laplace
  approximation in which the shift is recomputed per draw give `post_mean`, `post_sd`, `q025`, `q975`,
  `ppos_draws` and `lfsr_draws`, so the shift's uncertainty and its covariance with each effect are included.
- `contrast_summary.json`: the shift, the reference-set size, and the shift's spread across draws.
- `nbsr_results.h5` and `nbsr_results.csv`: per-sample log relative-abundance ratios with Wald tests
  (`--absolute_fc` subtracts the shift from them too) and `covariance.pth`, the per-sample covariance of the
  ratios across features (`--skip_cov` to omit).

Standard errors come from the Cholesky factor of the negative Hessian; `--recompute_hessian` recomputes it
from the checkpoint instead of reading `hessian.npy`.

## Example

```
python nbsr/main.py train data/test trt -i 10000 --pivot
python nbsr/main.py results data/test trt alt null --ref_top_frac 0.1 --laplace_draws 2000
```

`data/test` has 200 features and 20 samples with one two-level covariate `trt` (`null` vs `alt`), generated by
`scripts/generate_data.R`. `scripts/de.R` shows how the outputs are read in R for a differential-expression
analysis. Fits from earlier versions of the package (checkpoints, `disp_model.pth`, `config.json`) are still
read by `results`.

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
