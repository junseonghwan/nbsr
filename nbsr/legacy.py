"""Compatibility with checkpoints and dispersion-model files written by previous versions of NBSR.

Nothing here affects new fits. `upgrade_state_dict` is applied when a checkpoint.pth is loaded and
`convert_legacy_dispersion_model` when a pickled disp_model.pth is loaded; both translate the previous
parameterisation into the current one so that old runs can be re-analysed with `results`. If old files
are no longer of interest, this module and its two call sites in main.py can be removed together.
"""
import torch

from nbsr.dispersion import DispersionModel


def upgrade_state_dict(sd):
    """Translate checkpoints written before this version of the models.
    - psi (softplus-parameterized learned prior sd) -> fixed beta_prior_sd; lam / hyperprior buffers dropped.
    - library sizes s (now a buffer) recomputed from Y.
    - previous dispersion model b0 + b1 log pi + b2 log R -> DispersionModel(link="log", feature_offsets=False,
      W = log total counts) in the NB2 convention: b_0 = -b0, b_pi = -b1, b_w = -b2, z_bj = 0."""
    sd = dict(sd)
    if "psi" in sd and "beta_prior_sd" not in sd:
        sd["beta_prior_sd"] = torch.nn.functional.softplus(sd.pop("psi"))
        print("checkpoint: converted learned psi to a fixed beta_prior_sd")
    for k in ["lam", "beta_var_shape", "beta_var_scale"]:
        sd.pop(k, None)
    if "s" not in sd and "Y" in sd:
        sd["s"] = sd["Y"].to(torch.float64).sum(1)
    if "disp_model.b0" in sd:
        J = sd["Y"].shape[1]
        sd["disp_model.b_0"] = -sd.pop("disp_model.b0").reshape(1).to(torch.float64)
        sd["disp_model.b_pi"] = -sd.pop("disp_model.b1").reshape(1).to(torch.float64)
        sd["disp_model.b_w"] = -sd.pop("disp_model.b2").reshape(1).to(torch.float64)
        sd["disp_model.z_bj"] = torch.zeros(J, dtype=torch.float64)
        for k in [k for k in sd if k.startswith("disp_model.") and k.split(".", 1)[1] in ("Y", "R", "log_R", "Z", "beta")]:
            sd.pop(k)
        print("checkpoint: converted the previous dispersion model (b0 + b1 log pi + b2 log R) to the NB2 form")
    return sd


def convert_legacy_dispersion_model(old, Y):
    """Previous DispersionModel (b0 + b1 log pi + b2 log R, optional per-feature sd kappa) -> current class."""
    new = DispersionModel(Y.shape[1], W=torch.log(Y.sum(1, keepdim=True)), link="log", feature_offsets=False,
                          estimate_sd=getattr(old, "estimate_sd", False))
    with torch.no_grad():
        new.b_0.copy_(-old.b0.detach().reshape(1)); new.b_pi.copy_(-old.b1.detach().reshape(1)); new.b_w.copy_(-old.b2.detach().reshape(1))
        if new.estimate_sd:
            new.kappa.copy_(old.kappa.detach())
    print("dispersion model: converted the previous form (b0 + b1 log pi + b2 log R) to the NB2 form")
    return new
