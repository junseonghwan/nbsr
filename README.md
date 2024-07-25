# Python code for microRNA analysis

Negative Binomial Softmax Regression (NBSR) model for the counts data arising from miRNA-seq experiments.

## Installation

```bash
git clone https://github.com/junseonghwan/nbsr.git
```

```
cd /path/to/nbsr/
python -m pip install .
```

For development mode `-e` option:

```
cd /path/to/nbsr/
python -m pip install -e .
```

Run tests: 

```
python -m unittest tests/
```

or 

```
pytest tests/
```

## Execution

The required arguments are (1) path to directory containing `X.csv` and `Y.csv` and (2) a list of explatory variable names (covariates) separated by space matching the column names in `X.csv`.

```
python nbsr/main.py train path/to/data var1 var2 var3
```

Optional arguments:

```
-i: number of optimization iterations. Default: 10,000.
-l: learning rate for the optimzer. Default: 0.05. Recommended values less than 0.2.
-r: number of optimization runs. Default: 1.
--lam: scaling parameter of standar deviation on beta. Default: 1.
--shape: shape parameter for Beta prior distribution on standard deviation on beta. Default: 3.
--shape: shape parameter for Beta prior distribution on standard deviation on beta. Default: 2.
--dispersion_model: flag indicating where log dispersion modeling should be used. Default: False. 
--feature_specific_intercept: flag indicating feature specific intercept is to be used for the log dispersion model.
--grbf: flag indicating Gaussian Radial Basis Function is to be used as a prior on dispersion. 
```

We recommend to use ``--dispersion_model` when the number of samples for each experimental condtion `>= 10` and to enable `--feature_specific_intercept` flag. To use `--grbf` prior on the dispersion when fitting NBSR, we first running DESeq2 and NBSR with Empirical Bayes first. To run NBSR with Empirical Bayes dispersion, we require DESeq2 to be run first, producing a file `deseq2_mu.csv` in the path containing the data files `X.csv` and `Y.csv`. For this purpose, first run the script `deseq2.R`:

```
Rscript deseq2.R /path/to/data var1,var2,var3
```

This requires [R installation](https://www.r-project.org/) with [DESeq2 package installed](https://bioconductor.org/packages/release/bioc/html/DESeq2.html).

Then, run NBSR EB:

```
python nbsr/main.py eb path/to/data var1 var2 var3
```

with optional arguments:

```bash
--nbsr_iter: The number of iterations to use for obtaining the maximum likelihood estimates for the dipsersion parameter given the DESeq2 mean expression levels. Default: 5,000.
--nbsr_lr: The learning rate for obtaining the maximum likelihood estimates for the dipsersion parameter. Default: 0.05.
--grbf_iter: The number of iterations for optimizing GRBF model parameters. Default: 5,000.
--grbf_lr: The learning rate for optimizing GRBF model parameters. Default: 0.05.
--knot_count: The number of centers or knots for GRBF. Default: 10.
--grbf_sd: The scale parameter for the GRBF kernel. Default: 0.5.
```

Finally, we can fit NBSR with the prior on dispersion specified using GRBF by setting the `--grbf` flag: 

```
python nbsr/main.py train path/to/data var1 var2 var3 --grbf
```

Note that if `--dispersion_model` flag is set, then `--grbf` flag will be ignored. 
