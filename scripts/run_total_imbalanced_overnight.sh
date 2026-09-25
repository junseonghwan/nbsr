#!/bin/bash
# All conditions (HIGH/MED/LOW) x sizes x 20 replicates x the four configurations in scripts/run_simulation.py,
# sequential over conditions, 6 fits in parallel. Resumable: finished fits (score.json) are skipped.
set -u
SRC=~/Dropbox/seong/miRNA/output/total_imbalanced
OUT=~/nbsr_runs/total_imbalanced
PY=~/opt/anaconda3/envs/nbsr/bin/python
cd "$(dirname "$0")/.."
mkdir -p "$OUT"
for cond in HIGH MED LOW; do
  echo "=== $cond $(date)"
  $PY scripts/run_simulation.py --src "$SRC/$cond" --out "$OUT/$cond" --n_jobs 6 --threads 2
done
$PY - <<PYEOF
import pandas as pd, glob
frames = []
for f in glob.glob("$OUT/*/summary.csv"):
    d = pd.read_csv(f); d.insert(0, "condition", f.split("/")[-2]); frames.append(d)
df = pd.concat(frames); df.to_csv("$OUT/summary_all.csv", index=False)
cols = ["coverage", "coverage_perturbed", "bias_log2", "rmse_log2", "mean_se_log2", "TP_shift", "FP_shift", "FN_shift", "TP_zero", "FP_zero", "disp_b_pi", "disp_sigma_bj"]
pd.set_option("display.width", 250)
print("\n=== mean over replicates (all conditions) ===")
print(df.groupby(["condition", "config", "size"])[[c for c in cols if c in df]].mean().round(3).to_string())
print("\nsummary: $OUT/summary_all.csv")
PYEOF
echo "=== finished $(date)"
