"""Run NBSR configurations on every replicate of a simulation grid and score them against fc.csv.

Layout expected under --src: sample_<n>/rep<k>/{X.csv, Y.csv, fc.csv}. Each (config, size, replicate) is
fitted in its own directory under --out (inputs copied there, so the source tree is never written to),
scored, and summarised in <out>/summary.csv. Re-running skips replicates that already have a score.

Example:
    python scripts/run_simulation.py --src ~/Dropbox/seong/miRNA/output/total_imbalanced/HIGH \\
        --out ~/nbsr_runs/total_imbalanced_HIGH --n_jobs 6
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import false_discovery_control, norm

REPO = Path(__file__).resolve().parents[1]
MAIN = REPO / "nbsr" / "main.py"

NEW = ["--trended_dispersion", "--pivot", "--z_columns", "lib_size", "--z_columns", "miRNA_capture", "--z_log"]
PREV = ["--trended_dispersion", "--pivot", "--dispersion_link", "log", "--no_feature_offsets",
        "--z_total_counts", "--b_pi_prior", "0", "0.1", "--sigma_b", "0.1"]
CONFIGS = {
    # New dispersion model, the form of the NBSR-HMC Stan code (log phi = b_0 + b_j + b_pi logit(pi) + b_w' [log lib_size, log capture]) and the
    # previous model (b0 + b1 log pi + b2 log total counts), each with the two-stage empirical beta prior
    # (DESeq2-style quantile matching) or the jointly learned prior. The empirical ones run first.
    "new_emp": NEW + ["--beta_prior", "empirical"],
    "prev_emp": PREV + ["--beta_prior", "empirical"],
    "new_learn": NEW,
    "prev_learn": PREV,
}


def score(run_dir, factor="trt", numerator="alt", denominator="null"):
    Y = pd.read_csv(run_dir / "Y.csv", index_col=0)
    fc = pd.read_csv(run_dir / "fc.csv").set_index("miRNA").loc[Y.index]
    with pd.HDFStore(run_dir / f"{factor}__{numerator}_vs_{denominator}" / "nbsr_results.h5", "r") as store:
        est = store["logRR"].iloc[:, 0].to_numpy()
        se = store["se"].iloc[:, 0].to_numpy()
        padj = store["padj"].iloc[:, 0].to_numpy()
    truth = fc["log_fc"].to_numpy()
    perturbed = ~np.isclose(fc["alpha_null"], fc["alpha_alt"])
    covered = (truth > est - 1.96 * se) & (truth < est + 1.96 * se)
    shift = np.median(truth[~perturbed])
    p_shift = 2 * norm.cdf(-np.abs((est - shift) / se))
    called_shift = false_discovery_control(p_shift, method="bh") < 0.05
    called = padj < 0.05
    def confusion(c):
        return dict(TP=int((c & perturbed).sum()), FP=int((c & ~perturbed).sum()),
                    FN=int((~c & perturbed).sum()), TN=int((~c & ~perturbed).sum()))
    row = dict(n_features=len(est), n_perturbed=int(perturbed.sum()),
               coverage=float(covered.mean()), coverage_perturbed=float(covered[perturbed].mean()),
               bias_log2=float(np.mean((truth - est) / np.log(2))),
               rmse_log2=float(np.sqrt(np.mean(((truth - est) / np.log(2)) ** 2))),
               mean_se_log2=float(np.mean(se / np.log(2))))
    row.update({f"{k}_zero": v for k, v in confusion(called).items()})
    row.update({f"{k}_shift": v for k, v in confusion(called_shift).items()})
    return row


def run_one(src_rep, out_rep, config, iterations, threads):
    score_file = out_rep / "score.json"
    if score_file.exists():
        return json.loads(score_file.read_text())
    out_rep.mkdir(parents=True, exist_ok=True)
    for f in ["X.csv", "Y.csv", "fc.csv"]:
        shutil.copy2(src_rep / f, out_rep / f)
    env = dict(os.environ, OMP_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads))
    t0 = time.time()
    with open(out_rep / "train.log", "w") as log:
        subprocess.run([sys.executable, str(MAIN), "train", str(out_rep), "trt", "-i", str(iterations)] + CONFIGS[config],
                       check=True, stdout=log, stderr=subprocess.STDOUT, env=env)
    with open(out_rep / "results.log", "w") as log:
        subprocess.run([sys.executable, str(MAIN), "results", str(out_rep), "trt", "alt", "null", "--skip_cov"],
                       check=True, stdout=log, stderr=subprocess.STDOUT, env=env)
    row = dict(config=config, size=src_rep.parent.name, rep=src_rep.name, seconds=round(time.time() - t0, 1))
    row.update(score(out_rep))
    params = pd.read_csv(out_rep / "nbsr_dispersion_params.csv").iloc[0].to_dict()
    row.update({f"disp_{k}": float(v) for k, v in params.items()})
    score_file.write_text(json.dumps(row))
    for big in ["run0", "checkpoint.pth", "hessian.npy"]:   # keep the run directories small
        p = out_rep / big
        shutil.rmtree(p) if p.is_dir() else p.unlink(missing_ok=True)
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--configs", nargs="+", default=list(CONFIGS), choices=list(CONFIGS))
    ap.add_argument("--sizes", nargs="*", default=None, help="subset of sample_<n> directories")
    ap.add_argument("--iterations", type=int, default=15000)
    ap.add_argument("--n_jobs", type=int, default=4)
    ap.add_argument("--threads", type=int, default=2, help="torch threads per fit")
    args = ap.parse_args()

    sizes = sorted(p for p in args.src.glob("sample_*") if p.is_dir() and (args.sizes is None or p.name in args.sizes))
    jobs = [(rep, args.out / cfg / size.name / rep.name, cfg)
            for cfg in args.configs for size in sizes for rep in sorted(size.glob("rep*"))]
    print(f"{len(jobs)} fits ({len(args.configs)} configs x {len(sizes)} sizes x replicates), {args.n_jobs} in parallel", flush=True)

    def wrapped(src_rep, out_rep, cfg):
        try:
            row = run_one(src_rep, out_rep, cfg, args.iterations, args.threads)
            print(f"done {cfg} {src_rep.parent.name} {src_rep.name}: coverage {row['coverage']:.3f} "
                  f"TP/FP/FN(shift) {row['TP_shift']}/{row['FP_shift']}/{row['FN_shift']} {row['seconds']}s", flush=True)
            return row
        except Exception as e:  # keep the rest of the grid running; the failure is in the logs
            print(f"FAILED {cfg} {src_rep.parent.name} {src_rep.name}: {e}", flush=True)
            return None

    rows = Parallel(n_jobs=args.n_jobs, backend="loky")(delayed(wrapped)(*j) for j in jobs)
    df = pd.DataFrame([r for r in rows if r is not None])
    args.out.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out / "summary.csv", index=False)
    cols = ["coverage", "coverage_perturbed", "bias_log2", "rmse_log2", "mean_se_log2", "TP_shift", "FP_shift", "FN_shift", "TP_zero", "FP_zero", "seconds"]
    print("\n=== mean over replicates ===")
    print(df.groupby(["config", "size"])[cols].mean().round(3).to_string())
    print(f"\nsummary: {args.out / 'summary.csv'}")


if __name__ == "__main__":
    main()
