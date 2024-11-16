## NBSR implementation notes
There is quite a bit of choices to be made regarding dispersion specification.
Dispersion can be either feature wise or observation wise and it can also be trended where the dispersion is forced on to the mean function or be free to vary. When it is free to vary, typically, we want to impose some penalty in the form of prior elicitation to regulate it from controlling the likelihood and hence the optimization (i.e., rather than optimizing beta, the optimizer chooses to optimize phi to fit the data).
The dispersion can also be modelled in relation to sample specific covariates, leading to observation specific dispersion even if we use trended mean.
After some experimentation, we consider two workflows to yield the best results.

Typically, when we talk about feature wise dispersion parameterization, there is no sample specific covariates in the dispersion model as that necessarily yields observation wise dispersion.

Workflow 1: feature wise (N <= 10).

- Elicit prior by using NBSR EB with feature wise parameterization (either on the mean expression or pi).
- Fit posterior using feature wise and free parameterization.
- This is already implemented via `eb` and then running `train` with `--grbf` option on.

Workflow 2: N <= 10 and allowing for observation wise dispersion.
+ Initial fit of dispersion may be quite large -- so we should cap it at some value or have another prior defined on each $\phi_{ij}$.
+ When eliciting prior, we fit linear model on observation specific dispersion via trended mean (so estimate dispersion parameters b rather than directly maximing \phi_{ij} and then fitting b -- although both should be tried).
+ Then, either fix dispersion values and fit beta for posterior or allow free observation parameters regulated by prior.

Workflow 3: feature wise (N > 10)

- When there are sufficient sample sizes, we don't need to specify the prior.
- The MLE/MAP fit with feature wise dispersion is usually already quite good.
- This is already implemented in the `train` workflow but without `--grbf` specification.

Workflow 4: observation wise using sample specific covariates and trended (N > 10).

- fit observation wise dispersion model by allowing sample specific covariates.
- no prior elicitation is needed.
- use trended mean dispersion.
- This is already implemented in the `train` workflow with `--dispersion_model` on. It has shown to work well for N >= 10 for each experimental condition. But it is sensitive to initialization so we must do multiple runs.

Other workflow to consider:
- Prior elicitation: estimate trended dispersion model to maximize the likelihood -> use it as prior and fit observation wise and free model.

## Dispersion specification

1. Specify a file containing dispersion parameters.
2. Specify path to dispersion model file location containing Empirical Bayes elicited dispersion prior:
	A. Workflow 1: GRBF (`GaussianRBF`, which defaults to feature wise)
	B. Workflow 2: using log linear model (`DispersionModel` with covariates and necessarily observation wise) or 
	C. Workflow 5 (not implemented): Observation wise and free parameterization with prior specification.
3. Estimate it from scratch. In this case, user provides options (flags).
	A. Workflow 3: MAP (feature wise) -- NBSR.
	B. Workflow 4: log linear (observation wise using covariates) -- NBSRDispersion (create a prior model and optimize the params).

Two classes: trended vs free (NBSRTrended vs NBSRFree):
- trended-observation
- free-observation

When the prior model is not available, then by default it fits the MAP model without any prior or fits the trended observation wise model. So free-feature or trended-observation.

So to execute each of the workflows:

1. Generate a file containing dispersion estimates and save it as `dispersion.csv` in the `data_path`. Then, run `python nbsr/main.py train data_path var1 var2`. This will ignore `--dispersion_model_path` and `--observation_wise` flags and just run NBSR with dispersion fixed at the pre-specified values.
2. Run `eb` or `eb2` workflow to generate `disp_model.pth` under `data_path`.

TODO: Update standard NBSR model to utilize the prior distribution for free-observation wise parameterization.
+ It needs to represent phi as NxK or K depending on if feature or observation wise dispersion parameterization is used.

Performance notes:
1. Trended dispersion performs better than free-feature wise parameterization when no prior is specified (i.e., N >= 10).

