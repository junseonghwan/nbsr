#!/usr/bin/env Rscript
args = commandArgs(trailingOnly=TRUE)
print(args)

library(DESeq2)

rep_path <- args[1]
covariates <- strsplit(args[2], ",")[[1]] 
Y <- read.csv(paste0(rep_path, "/Y.csv"))
X <- read.csv(paste0(rep_path, "/X.csv"))
feature_count <- dim(Y)[1]
sample_count <- dim(Y)[2]

Y <- as.matrix(Y)
se <- SummarizedExperiment(assays = list(counts = Y), colData = X)
model_formula <- as.formula(paste("~", paste(covariates, collapse="+")))
dds2 <- DESeqDataSet(se, model_formula)
dds2 <- DESeq(dds2)
#res2 <- results(dds2, contrast = c(var_name, w1, w0))
deseq2_mu <- assays(dds2)[["mu"]]
write.table(deseq2_mu, file = paste0(rep_path, "/deseq2_mu.csv"), row.names = T, quote =F, col.names = T, sep=",")

