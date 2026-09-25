from dataclasses import dataclass, asdict
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
    lr:    float = 0.05
    lam:   float = 1.0
    shape: float = 3.0
    scale: float = 2.0
    iterations: int = 10000
    estimate_dispersion_sd: bool = False
    trended_dispersion:     bool = False
    dispersion_path:        Path | None = None
    dispersion_model_file:  str | None = None # we will look for output_path / dispersion_model_file
    update_dispersion:      bool = False  # optimise a pre-fitted dispersion model jointly with beta.
    pivot:  bool = False
    use_cuda_if_available: bool = True

    def dump_json(self, file: str | Path):
        """Write the config (incl. derived paths) to disk for provenance."""
        with open(file, "w") as f:
            json.dump(asdict(self), f, indent=4, default=str)

    @classmethod
    def load_json(cls, file: str | Path):
        data = json.load(open(file))
        data["counts_path"]  = Path(data["counts_path"])
        data["coldata_path"] = Path(data["coldata_path"])
        if data.get("dispersion_path") is not None:
            data["dispersion_path"] = Path(data["dispersion_path"])
        # if data.get("dispersion_model_path") is not None:
        #     data["dispersion_model_path"] = Path(data["dispersion_model_path"])
        return cls(**data)