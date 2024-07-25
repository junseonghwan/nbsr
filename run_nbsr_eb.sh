#!/bin/bash

echo $(date)
echo which python
echo which Rscript


# Arguments.
# 1: Path to X.csv, Y.csv.
# 2: List of covariates.
# 3: Training iterations for main NBSR run.
# 4: Learning rate for main NBSR run.
IFS=',' read -r -a covariates <<< "$2"
Rscript deseq2.R $1 $2
echo "${covariates[@]}"
# Edit below line to specify EB and GRBF iterations and learning rate.
python nbsr/main.py eb $1 "${covariates[@]}"
# Fit NBSR.
python nbsr/main.py train $1 "${covariates[@]}" -i $3 -l $4

echo $(date)
