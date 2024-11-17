## NBSR implementation notes
There is quite a bit of choices to be made regarding dispersion specification.
Dispersion can be either feature wise or observation wise and it can also be trended where the dispersion is forced on to the mean function or be free to vary. When it is free to vary, typically, we want to impose some penalty in the form of prior elicitation to regulate it from controlling the likelihood and hence the optimization (i.e., rather than optimizing beta, the optimizer chooses to optimize phi to fit the data).
The dispersion can also be modelled in relation to sample specific covariates, leading to observation specific dispersion even if we use trended mean.
After some experimentation, we consider two workflows to yield the best results.

Typically, when we talk about feature wise dispersion parameterization, there is no sample specific covariates in the dispersion model as that necessarily yields observation wise dispersion.

Workflow 1: pre-specify dispersion (from DESeq2 or EdgeR) and run NBSR to see how the results change.

Workflow 2: $N <= 10$ and allowing for observation wise dispersion.
+ Initial fit of dispersion may be quite large -- so we should cap it at some value or have another prior defined on each $\phi_{ij}$?
+ When eliciting prior, we fit linear model on observation specific dispersion via trended mean (so estimate dispersion parameters b rather than directly maximing $\phi_{ij}$ and then fitting b -- although both should be tried). Later the linear model can be generalized to be a local regression on $\log \pi_{ij}$.
+ Then, either fix dispersion values and fit beta for posterior or allow free observation parameters regulated by prior.

Workflow 3: observation wise using sample specific covariates and trended ($N > 10$).

- fit observation wise dispersion model by allowing sample specific covariates.
- no prior elicitation is needed.
- use trended mean dispersion.

Workflow 4: feature wise ($N > 10$)

- When there are sufficient samples, we can just go for MLE/MAP fit with feature wise dispersion.

Other workflow to consider:
- Prior elicitation: estimate dispersion model parameters to maximize the likelihood given $\hat{\mu}_{ij}$ -> use it as prior and fit observation wise and free model.

Two classes for handling trended vs free dispersion parameterization (`NBSRTrended` vs `NegativeBinomialRegressionModel`):

So to execute each of the workflows:

0. Generate a file containing dispersion estimates and save it as `dispersion.csv` in the `data_path`. Then, run 

`python nbsr/main.py train data_path var1 var2 [options]`. 

As long as the program finds `dispersion.csv` in the `data_path`, `--dispersion_model_path` and `--observation_wise` flags will be ignored; NBSR with dispersion fixed at the pre-specified values will be used.

1. Run `eb` workflow to generate `disp_model.pth` under `data_path`.

`python nbsr/main.py eb data_path var1 var2 -f path_to_mu_hat.csv`

2. Fit NBSR trended dispersion with dispersion prior:

`python nbsr/main.py train data_path var1 var2 -d disp_model.pth --trended_dispersion`

To fit NBSR trneded dispersion without pre-elicit dispersion prior with 5 random initialization points:

`python nbsr/main.py train data_path var1 var2 --trended_dispersion -r 5`

If no prior is used, we recommend running it for N >= 10 and also to utilize multiple runs.

3. To fit NBSR with free and feature wise dispersion with prior:

`python nbsr/main.py train data_path var1 var2 -d disp_model.pth`

without prior:

`python nbsr/main.py train data_path var1 var2`
