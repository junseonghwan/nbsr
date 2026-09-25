from dataclasses import dataclass, asdict, fields
from pathlib import Path
import json

@dataclass
class NBSRConfig():
    counts_path: Path
    coldata_path: Path
    output_path: Path
    column_names: list[str]

    # Optional variables
    z_columns: list[str] | None = None  # external covariates of the dispersion model (columns of X.csv)
    z_log: bool = False                  # log-transform z_columns before use
    # Priors of the dispersion model (nbsr/dispersion.py).
    b_pi_prior_mean: float = 1.0
    b_pi_prior_sd: float = 0.1
    sigma_bj_prior_sd: float = 0.5
    sigma_b: float = 1.0                 # prior sd of the coefficients on z_columns
    dispersion_link: str = "logit"       # "logit" (NBSR-HMC) or "log" (previous NBSR model)
    feature_offsets: bool = True         # per-feature offsets b_j in the dispersion model
    z_total_counts: bool = False         # add log(sum_j Y_ij) as an external dispersion covariate
    lr:    float = 0.05
    iterations: int = 10000
    estimate_dispersion_sd: bool = False
    trended_dispersion:     bool = False
    dispersion_path:        Path | None = None
    dispersion_model_file:  str | None = None # we will look for output_path / dispersion_model_file
    update_dispersion:      bool = False  # optimise a pre-fitted dispersion model jointly with beta.
    pivot:  bool = False
    beta_prior_sd: list[float] | None = None  # prior sd of beta per covariate (intercept first). None = "empirical":
                                         #   a stage-1 fit with a wide prior, then per-covariate sd matched to the
                                         #   upper quantile of |beta| (DESeq2-style), then the main fit with it fixed.
    beta_prior: str = "empirical"        # "empirical" or "fixed" (beta_prior_sd given); set by the CLI, informational
    stage1_iterations: int | None = None # iterations of the stage-1 fit (default: half of iterations)
    stage1_prior_sd: float = 10.0        # prior sd of every covariate in the stage-1 fit
    prior_quantile: float = 0.95         # |beta| quantile matched to the prior's tail
    init_from: Path | None = None        # checkpoint whose beta and dispersion parameters initialise the fit
    use_cuda_if_available: bool = True

    def dump_json(self, file: str | Path):
        """Write the config (incl. derived paths) to disk for provenance."""
        with open(file, "w") as f:
            json.dump(asdict(self), f, indent=4, default=str)

    @classmethod
    def load_json(cls, file: str | Path):
        data = json.load(open(file))
        known = {f.name for f in fields(cls)}
        dropped = sorted(set(data) - known)
        if dropped:
            print(f"config.json: ignoring fields no longer used: {dropped}")
        data = {k: v for k, v in data.items() if k in known}
        data["counts_path"]  = Path(data["counts_path"])
        data["coldata_path"] = Path(data["coldata_path"])
        if data.get("dispersion_path") is not None:
            data["dispersion_path"] = Path(data["dispersion_path"])
        # if data.get("dispersion_model_path") is not None:
        #     data["dispersion_model_path"] = Path(data["dispersion_model_path"])
        return cls(**data)