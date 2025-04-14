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

The required arguments are (1) path to directory containing `X.csv` and `Y.csv` and (2) a list of explanatory variable names (covariates) separated by space matching the column names in `X.csv`.

```
python nbsr/main.py train /path/to/data var1 var2 var3
```

Optional arguments:

```
-i: number of optimization iterations. Default: 10,000.
-l: learning rate for the optimzer. Default: 0.05. Recommend trying values <= 0.2.
-r: number of optimization runs. Default: 1.
--lam: scaling parameter of standar deviation on beta. Default: 1.
--shape: shape parameter for Beta prior distribution on standard deviation on beta. Default: 3.
--scale: shape parameter for Beta prior distribution on standard deviation on beta. Default: 2.
---trended_dispersion: use trended median dispersion. Default: False.
```

When the number of samples for each experimental condtion is `>= 10`, we recommend to try `--trended_dispersion`.
When the number of samples is small, we recommend to first run DESeq2 to obtain mean expression levels for each sample and feature and then to run NBSR with Empirical Bayes. NBSR EB will estimate the dispersion model parameters, which will be used as a prior in estimating the feature-wise dispersion parameters. 
To run DESeq2, use `deseq2.R`:

```
Rscript scripts/deseq2.R /path/to/data var1,var2,var3
```

This requires [R installation](https://www.r-project.org/) with [DESeq2 package installed](https://bioconductor.org/packages/release/bioc/html/DESeq2.html). `/path/to/data` should match the path containing the data files `X.csv` and `Y.csv`. Running `deseq2.R` will produce a file `/path/to/data/deseq2_mu.csv`.

Then, run NBSR EB:

```
python nbsr/main.py eb /path/to/data var1 var2 var3
```

with optional arguments:

```
--nbsr_iter: The number of iterations to use for obtaining the maximum likelihood estimates for the dipsersion parameter given the DESeq2 mean expression levels. Default: 5,000.
--nbsr_lr: The learning rate for obtaining the maximum likelihood estimates for the dipsersion parameter. Default: 0.05.
--grbf_iter: The number of iterations for optimizing GRBF model parameters. Default: 5,000.
--grbf_lr: The learning rate for optimizing GRBF model parameters. Default: 0.05.
--knot_count: The number of centers or knots for GRBF. Default: 10.
--grbf_sd: The scale parameter for the GRBF kernel. Default: 0.5.
```

Finally, we can fit NBSR with the prior on dispersion specified using GRBF by setting the `--grbf` flag: 

```
python nbsr/main.py train /path/to/data var1 var2 var3 --grbf
```

Note that if `--dispersion_model` flag is set, then `--grbf` flag will be ignored. 

Refer to script `run_nbsr_eb.sh`, which streamlines the process of running NBSR with GRBF dispersion prior.

Running NBSR's `train` will generate a checkpoint file in the same path where the data resides (i.e., `/path/to/data`).

Finally, to perform inference to compare two experimental conditions say on `var1` with `level1` (numerator) and `level2` (denominator):

```
python nbsr/main.py results /path/to/data var1 level1 level2
```

The above command will create a directory `/path/to/data/level1_level2` and generate result files.

## Example

A test dataset can be found in `data/test`. This dataset contains one covariate `trt` (treatment) with levels `null` and `alt` with `n=10` for each of the two conditions. The code for generating this test data can be found in `scripts/generate_data.R`. 

We will run NBSR on the test data via command:

```
python nbsr/main.py train data/test/ trt -i 20000 --dispersion_model --feature_specific_intercept
```

Then, to compare two treatment levels with `alt` in the numerator:

```
python nbsr/main.py results data/test trt alt null
```


## Analysis

While we are working on developing an R package to interface with Python code, we suggest to run NBSR on command line and load the results in R for analysis. An example script for performing differential expression analysis is given in `scripts/de.R`.

Further information will be provided once we develop an R package.

The figures from the paper can be reproduced by following the scripts provided [here](https://github.com/junseonghwan/nbsr-experiments/).

## Citation

If you use our package for analysis, please cite our paper doi: https://doi.org/10.1101/2024.05.07.592964.

