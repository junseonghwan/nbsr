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

## Analysis

To analyze individual dataset using Jupyter-lab, refer to `demo.ipynb`.

## Batch execution to replicate simulation results

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
---estimate_dispersion_sd: use trended mean dispersion if flag is on. Default: False.
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

Finally, to perform inference to compare two experimental conditions say on `var1` with `level1` (numerator) and `level2` (denominator):

```
python nbsr/main.py results /path/to/data var1 level1 level2
```

The above command will create a directory `/path/to/data/level1_level2` and generate result files.

## Example

A test dataset can be found in `data/test`. This dataset contains one covariate `trt` (treatment) with levels `null` and `alt` with `n=10` for each of the two conditions. The code for generating this test data can be found in `scripts/generate_data.R`. 

We will run NBSR on the test data via command:

```
python nbsr/main.py train data/test/ trt -i 10000 --dispersion_model --feature_specific_intercept
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

