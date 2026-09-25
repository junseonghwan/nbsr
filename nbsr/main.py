"""Command line interface.

    nbsr prior   DATA VARS...   stage 1: elicit the prior (diffuse fit, quantile-matched sd per covariate,
                                optionally the dispersion model fitted at DESeq2's composition) -> prior.json
    nbsr fit     DATA VARS...   stage 2: fit at fixed prior sds (from prior.json or --beta_prior_sd)
    nbsr train   DATA VARS...   prior followed by fit
    nbsr results FIT VAR W1 W0  contrasts, standard errors, sign probabilities, per-sample log ratios

DATA holds Y.csv (features x samples) and X.csv (sample metadata); VARS are columns of X.csv.
"""
import os
import shutil
from pathlib import Path

import click
import numpy as np

from nbsr.fit import fit_once
from nbsr.inference import generate_results
from nbsr.nbsr_config import NBSRConfig
from nbsr.prior import apply_prior, elicit_prior, load_prior, prior_filename


@click.group()
def cli():
    pass


def model_options(f):
    """Options shared by prior, fit and train: data, design, dispersion model, optimiser."""
    opts = [
        click.argument("data_path", type=click.Path(exists=True)),
        click.argument("vars", nargs=-1),
        click.option("-i", "--iterations", default=10000, show_default=True, type=int, help="Adam iterations."),
        click.option("-l", "--lr", default=0.05, show_default=True, type=float, help="Adam learning rate."),
        click.option("--out", default=None, type=click.Path(), help="Output directory (default: DATA for fit/train, DATA/prior for prior)."),
        click.option("--trended_dispersion/--free_dispersion", default=True, show_default=True,
                     help="Dispersion model b_0 + b_j + b_pi f(pi) + w'b_w (default) or one free dispersion per feature."),
        click.option("--z_columns", multiple=True, help="Columns of X.csv used as external covariates of the dispersion model."),
        click.option("--z_log", is_flag=True, default=False, help="Log-transform the --z_columns (e.g. library sizes)."),
        click.option("--z_total_counts", is_flag=True, default=False, help="Add log total counts per sample as a dispersion covariate."),
        click.option("--dispersion_link", type=click.Choice(["logit", "log"]), default="logit", show_default=True,
                     help="Transform f of pi in the dispersion model."),
        click.option("--no_feature_offsets", is_flag=True, default=False, help="Drop the per-feature offsets b_j."),
        click.option("--b_pi_prior", nargs=2, type=float, default=(1.0, 0.1), show_default=True, help="Mean and sd of the prior on b_pi."),
        click.option("--sigma_bj_prior_sd", type=float, default=0.5, show_default=True, help="Half-normal scale of the prior on sigma_bj."),
        click.option("--sigma_b", type=float, default=1.0, show_default=True, help="Prior sd of the coefficients on --z_columns."),
        click.option("--dispersion_model_file", default=None, type=str, help="Fitted dispersion model (disp_model.pth) to load and hold fixed."),
        click.option("--update_dispersion", is_flag=True, default=False, help="Optimise a loaded dispersion model jointly with beta."),
        click.option("--estimate_dispersion_sd", is_flag=True, default=False, help="Per-feature sd for the log-normal dispersion prior (free-dispersion model)."),
        click.option("--pivot", is_flag=True, default=False, help="Fix the last feature's coefficients at zero (reference category)."),
    ]
    for opt in reversed(opts):
        f = opt(f)
    return f


def prior_options(f):
    opts = [
        click.option("--stage1_prior_sd", type=float, default=10.0, show_default=True, help="Prior sd of every covariate in the diffuse fit."),
        click.option("--stage1_iterations", type=int, default=None, help="Iterations of the diffuse fit (default: half of -i)."),
        click.option("--prior_quantile", type=float, default=0.95, show_default=True, help="Quantile of |beta| matched to the prior's tail."),
        click.option("--dispersion_from", type=click.Choice(["joint", "deseq2"]), default="joint", show_default=True,
                     help="joint: dispersion model fitted with beta; deseq2: fitted at DESeq2's composition and held fixed (small samples)."),
        click.option("--eb_iterations", type=int, default=3000, show_default=True, help="Iterations for the DESeq2-composition dispersion fit."),
        click.option("--eb_lr", type=float, default=0.05, show_default=True),
    ]
    for opt in reversed(opts):
        f = opt(f)
    return f


def make_config(data_path, vars, out, iterations, lr, trended_dispersion, z_columns, z_log, z_total_counts, dispersion_link,
                no_feature_offsets, b_pi_prior, sigma_bj_prior_sd, sigma_b, dispersion_model_file, update_dispersion,
                estimate_dispersion_sd, pivot, **extra):
    data_path = Path(data_path)
    return NBSRConfig(counts_path=data_path / "Y.csv", coldata_path=data_path / "X.csv",
                      output_path=Path(out) if out else data_path, column_names=list(vars),
                      iterations=iterations, lr=lr, trended_dispersion=trended_dispersion,
                      z_columns=list(z_columns), z_log=z_log, z_total_counts=z_total_counts,
                      dispersion_link=dispersion_link, feature_offsets=not no_feature_offsets,
                      b_pi_prior_mean=b_pi_prior[0], b_pi_prior_sd=b_pi_prior[1], sigma_bj_prior_sd=sigma_bj_prior_sd,
                      sigma_b=sigma_b, dispersion_model_file=dispersion_model_file, update_dispersion=update_dispersion,
                      estimate_dispersion_sd=estimate_dispersion_sd, pivot=pivot, **extra)


@cli.command()
@model_options
@prior_options
def prior(**kw):
    """Stage 1: elicit the empirical prior; writes prior.json (default: DATA/prior/)."""
    out = kw.pop("out") or str(Path(kw["data_path"]) / "prior")
    config = make_config(out=out, **kw)
    config.beta_prior = "empirical"
    elicit_prior(config)


def _fit(config, prior_file, beta_prior_sd, runs):
    if beta_prior_sd:
        config.beta_prior, config.beta_prior_sd = "fixed", list(beta_prior_sd)
    elif prior_file:
        config = apply_prior(config, load_prior(prior_file))
        config.prior_file = str(prior_file)
    else:
        raise click.UsageError("give --prior prior.json (from `nbsr prior`) or --beta_prior_sd")
    data_path = Path(config.output_path)
    losses = []
    for run_no in range(runs):
        run_config = config if runs == 1 else _with_output(config, data_path / f"run{run_no}")
        loss_history, _ = fit_once(run_config)
        losses.append(np.min(loss_history))
    if runs > 1:   # copy the best run's files up to the output directory
        best = data_path / f"run{int(np.argmin(losses))}"
        for f in os.listdir(best):
            if (best / f).is_file():
                shutil.copy2(best / f, data_path / f)


def _with_output(config, path):
    import copy
    c = copy.deepcopy(config)
    c.output_path = path
    return c


@cli.command()
@model_options
@click.option("--prior", "prior_file", type=click.Path(exists=True), default=None, help="prior.json from `nbsr prior`.")
@click.option("--beta_prior_sd", multiple=True, type=float, help="Fixed prior sd of beta: one value, or one per covariate (intercept first).")
@click.option("-r", "--runs", default=1, show_default=True, type=int, help="Independent initialisations; the best log posterior is kept.")
def fit(prior_file, beta_prior_sd, runs, **kw):
    """Stage 2: fit at fixed prior sds, from prior.json or --beta_prior_sd; outputs go to DATA (or --out)."""
    _fit(make_config(**kw), prior_file, beta_prior_sd, runs)


@cli.command()
@model_options
@prior_options
@click.option("-r", "--runs", default=1, show_default=True, type=int, help="Independent initialisations of the main fit.")
def train(runs, **kw):
    """prior followed by fit: elicit the empirical prior into DATA/prior/, then fit with it into DATA (or --out)."""
    out = kw.pop("out")
    prior_kw = {k: kw[k] for k in ["stage1_prior_sd", "stage1_iterations", "prior_quantile", "dispersion_from", "eb_iterations", "eb_lr"]}
    fit_kw = {k: v for k, v in kw.items() if k not in prior_kw}
    base = Path(out) if out else Path(kw["data_path"])
    prior_config = make_config(out=str(base / "prior"), **fit_kw, **prior_kw)
    prior_config.beta_prior = "empirical"
    elicit_prior(prior_config)
    _fit(make_config(out=str(base), **fit_kw), base / "prior" / prior_filename, (), runs)


@cli.command()
@click.argument("fit_path", type=click.Path(exists=True))
@click.argument("var", type=str)
@click.argument("w1", type=str)
@click.argument("w0", type=str)
@click.option("--ref_top_frac", type=float, default=None, help="Estimate the compositional shift from the top fraction of features by pooled abundance (default: all features).")
@click.option("--ref_min", type=int, default=30, show_default=True, help="Minimum number of reference features when --ref_top_frac is set.")
@click.option("--laplace_draws", type=int, default=0, show_default=True, help="Draws from the Laplace approximation to recompute the shift per draw and summarise the absolute effect.")
@click.option("--seed", type=int, default=0, show_default=True)
@click.option("--absolute_fc", is_flag=True, default=False, help="Also subtract the shift from the per-sample logRR outputs.")
@click.option("--recompute_hessian", is_flag=True, default=False)
@click.option("--skip_cov", is_flag=True, default=False, help="Do not compute/save the per-sample covariance of logRR across features.")
def results(fit_path, var, w1, w0, ref_top_frac, ref_min, laplace_draws, seed, absolute_fc, recompute_hessian, skip_cov):
    """Contrast of level W1 against W0 of VAR from a finished fit: contrast.csv (CLR contrast, shift, absolute
    effect, ppos), per-sample nbsr_results.h5/csv."""
    generate_results(fit_path, var, w1, w0, absolute_fc=absolute_fc, recompute_hessian=recompute_hessian,
                     save_cov=not skip_cov, ref_top_frac=ref_top_frac, ref_min=ref_min, laplace_draws=laplace_draws, seed=seed)


if __name__ == "__main__":
    cli()
