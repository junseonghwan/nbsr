#!/bin/bash

echo $(date)
echo which python


# Arguments.
# 1: Path to X.csv, Y.csv.
# 2: List of covariates.
# 3: Training iterations.
# 4: Learning rate.
IFS=',' read -r -a covariates <<< "$2"
Rscript deseq2.R $1 $2
python nbsr/main.py eb $1 "${covariates[@]}"
# Fit NBSR.
python nbsr/main.py train $1 "${covariates[@]}" -i $3 -l $4

echo $(date)
