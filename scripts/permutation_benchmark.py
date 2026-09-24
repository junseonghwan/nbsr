"""Permutation benchmark on the immune (Pre vs On treatment) data.

Shuffles the treatment labels across samples so that every gene is null, then refits
  * the featurewise NB model in three prior configurations
      full     : b_prior_sd=0.5, gamma_prior_sd=1.0   (mean-dependent + condition-specific dispersion)
      no_gamma : gamma_prior_sd=1e-3                   (condition effect on dispersion switched off)
      constant : b_prior_sd=1e-3, gamma_prior_sd=1e-3 (one dispersion per gene, DESeq2-like)
    each reported with the model-based (inverse Hessian) SE and, as <name>_robust, with the sandwich SE
  * PyDESeq2 (Wald test, default settings)
and records the number of genes called at padj < 0.05 / 0.10 and the fraction of raw p < 0.05.
Permutation 0 is the observed labels.

Usage: python scripts/permutation_benchmark.py [n_perm] [seed] [out_dir]
Outputs to data/immune_data/permutation/: summary.csv (appended per run) and pvalues_<method>.npy.
"""
import sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import anndata as ad
import scipy.stats as ss
from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats
from pydeseq2.default_inference import DefaultInference

from nbsr.dataset import Dataset
from nbsr.fnb_stats import FeaturewiseNBStats

N_PERM = int(sys.argv[1]) if len(sys.argv) > 1 else 10
SEED   = int(sys.argv[2]) if len(sys.argv) > 2 else 1
N_CPUS = 8
OUT = Path(sys.argv[3]) if len(sys.argv) > 3 else Path("data/immune_data/permutation")
OUT.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(N_CPUS)

CONFIGS = {
    "fnb_full":     dict(b_prior_sd=0.5,  gamma_prior_sd=1.0),
    "fnb_no_gamma": dict(b_prior_sd=0.5,  gamma_prior_sd=1e-3),
    "fnb_constant": dict(b_prior_sd=1e-3, gamma_prior_sd=1e-3),
}

adata = ad.read_h5ad("data/immune_data/dataset.h5ad")
counts = pd.DataFrame(np.asarray(adata.X), index=adata.obs_names, columns=adata.var_names)
obs_trt = adata.obs["trt"].astype(str).to_numpy()
G = counts.shape[1]
rng = np.random.default_rng(SEED)

METHODS = [m for c in CONFIGS for m in (c, c + "_robust")] + ["deseq2"]
pvals = {m: np.full((N_PERM + 1, G), np.nan) for m in METHODS}
rows = []

def summarize(method, perm, p, padj, extra=None):
    ok = np.isfinite(p)
    row = dict(perm=perm, method=method, n_tested=int(ok.sum()),
               n_padj05=int(np.nansum(padj < 0.05)), n_padj10=int(np.nansum(padj < 0.10)),
               frac_p05=float(np.mean(p[ok] < 0.05)),
               ks_stat=float(ss.kstest(p[ok], "uniform").statistic))
    row.update(extra or {})
    rows.append(row)
    print(f"  {method:20s} padj<.05: {row['n_padj05']:5d}  padj<.10: {row['n_padj10']:5d}  "
          f"frac p<.05: {row['frac_p05']:.3f}  KS: {row['ks_stat']:.3f}", flush=True)

for perm in range(N_PERM + 1):
    trt = obs_trt if perm == 0 else rng.permutation(obs_trt)
    meta = pd.DataFrame({"trt": pd.Categorical(trt, categories=["Pre", "On"])}, index=counts.index)
    print(f"=== permutation {perm} ({'observed' if perm == 0 else 'shuffled'}) ===", flush=True)

    t0 = time.time()
    ds = Dataset(counts, meta, mean_formula="~trt", disp_formula="~trt", n_cpus=N_CPUS)
    t_ds = time.time() - t0
    for name, cfg in CONFIGS.items():
        t0 = time.time()
        st = FeaturewiseNBStats(ds, **cfg)
        st.fit(method="newton")
        t_fit = round(time.time() - t0, 1)
        for suffix, robust in [("", False), ("_robust", True)]:
            res = st.results("trt", reference_level="Pre", test_level="On", robust=robust)
            pvals[name + suffix][perm] = res["pval"].to_numpy()
            summarize(name + suffix, perm, res["pval"].to_numpy(), res["padj"].to_numpy(),
                      dict(n_converged=int(res["converged"].sum()), seconds=t_fit))

    t0 = time.time()
    dds = DeseqDataSet(counts=counts, metadata=meta, design="~trt",
                       inference=DefaultInference(n_cpus=N_CPUS), quiet=True)
    dds.deseq2()
    stat = DeseqStats(dds, contrast=["trt", "On", "Pre"], inference=DefaultInference(n_cpus=N_CPUS), quiet=True)
    stat.summary()
    df = stat.results_df.reindex(counts.columns)
    pvals["deseq2"][perm] = df["pvalue"].to_numpy()
    summarize("deseq2", perm, df["pvalue"].to_numpy(), df["padj"].to_numpy(),
              dict(n_converged=int(df["pvalue"].notna().sum()), seconds=round(time.time() - t0, 1)))

    pd.DataFrame(rows).to_csv(OUT / "summary.csv", index=False)
    for m, arr in pvals.items():
        np.save(OUT / f"pvalues_{m}.npy", arr)
    print(f"  (dataset build {t_ds:.0f}s)", flush=True)

print("done", flush=True)
