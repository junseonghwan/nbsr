## NBSR implementation notes
There is quite a bit of choices to be made regarding dispersion specification.
Dispersion can be either feature wise or observation wise and it can also be trended where the dispersion is forced on to the mean function or be free to vary. When it is free to vary, typically, we want to impose some penalty in the form of prior elicitation to regulate it from controlling the likelihood and hence the optimization (i.e., rather than optimizing beta, the optimizer chooses to optimize phi to fit the data).
The dispersion can also be modelled in relation to sample specific covariates, leading to observation specific dispersion even if we use trended mean.
After some experimentation, we consider two workflows to yield the best results.

Typically, when we talk about feature wise dispersion parameterization, there is no sample specific covariates in the dispersion model as that necessarily yields observation wise dispersion.

Two classes for handling trended vs free dispersion parameterization (`NBSRTrended` vs `NegativeBinomialRegressionModel`):


Workflow 1: pre-specify dispersion (from DESeq2 or EdgeR) and run NBSR to see how the results change.

Generate a file containing dispersion estimates and save it as `dispersion.csv` in the `data_path`. Then, run 

`python nbsr/main.py train data_path var1 var2 [options]`. 

As long as the program finds `dispersion.csv` in the `data_path`, `--dispersion_model_path` and `--observation_wise` flags will be ignored; NBSR with dispersion fixed at the pre-specified values will be used.

This will use `NegativeBinomialRegressionModel`.

Workflow 2: observation wise dispersion using initial fit of $\hat{\mu}_{ij}$.

+ When eliciting prior, we fit linear model on observation specific dispersion via trended mean. TODO: the linear model can be generalized to local regression on $\log \pi_{ij}$.

`python nbsr/main.py eb data_path var1 var2 -f path_to_mu_hat.csv`

This will first estimate prior model parameters $b$ and then, estimate NBSR parameters using trended dispersion.

This will use `NBSRTrended`.

Workflow 3: observation wise using sample specific covariates and trended ($N > 10$).

- fit observation wise dispersion model by allowing sample specific covariates.
- no prior elicitation is needed.
- use trended mean dispersion.

`python nbsr/main.py train data_path var1 var2 -d disp_model.pth --trended_dispersion -r 5`

Run it with different initializations with option `-r 5`.

This will use `NBSRTrended`.

Workflow 4: feature wise ($N > 10$)

- When there are sufficient samples, we can just go for MLE/MAP fit with feature wise dispersion.

`python nbsr/main.py train data_path var1 var2 -r 5`

This will use `NegativeBinomialRegressionModel`.