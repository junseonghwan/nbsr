rm(list=ls())
set.seed(1)

library(data.table)
library(DESeq2)
library(edgeR)
library(ggplot2)

source("scripts/functions.R")

data_path <- "data/test/"
X <- fread(paste0(data_path, "/X.csv"))
Y <- as.matrix(read.csv(paste0(data_path, "/Y.csv")))
fc <- fread(paste0(data_path, "/fc.csv"))

level1 <- "alt"
level2 <- "null"

thresholds <- c(0.01, 0.05, 0.1)

# Run DESeq2.
se <- SummarizedExperiment(assays = list(counts = Y), colData = X)
dds2 <- DESeqDataSet(se, ~ trt)
dds2 <- DESeq(dds2)
res2 <- results(dds2, contrast = c("trt", level1, level2))

# Run EdgeR.
d <- edgeR::DGEList(counts=Y)
d <- calcNormFactors(d)
design_mat <- model.matrix(~ trt, X)
d <- edgeR::estimateDisp(d, design = design_mat)
fit <- glmFit(d, design_mat)
lrt <- glmLRT(fit, contrast=c(0, -1))
edgeR_results <- topTags(lrt, n=Inf, sort.by="none")

# Run NBSR on terminal and then run below code.
R <- colSums(Y)
nbsr_pi <- read.csv(paste0(data_path, "/nbsr_pi.csv"), header = F)
nbsr_phi <- as.matrix(read.csv(paste0(data_path, "/nbsr_dispersion.csv"), header = F))
dim(nbsr_phi)
nbsr_mu <- sweep(nbsr_pi, 2, R, "*")

rownames(nbsr_pi) <- rownames(nbsr_mu) <- rownames(Y)
colnames(nbsr_pi) <- colnames(nbsr_mu) <- colnames(Y)

nbsr_pi <- as.matrix(nbsr_pi)
nbsr_mu <- as.matrix(nbsr_mu)

# Compute log2 fold change for NBSR.
nbsr_log2_fc <- log2(nbsr_pi[,which(X$trt == "alt")[1]]) - log2(nbsr_pi[,which(X$trt == "null")[1]])

deseq2_err <- fc$log2_fc - res2$log2FoldChange
edgeR_err <- fc$log2_fc - edgeR_results$table$logFC
nbsr_err <- fc$log2_fc - nbsr_log2_fc

plot(fc$log2_fc, res2$log2FoldChange)
points(fc$log2_fc, edgeR_results$table$logFC, col='red')
points(fc$log2_fc, nbsr_log2_fc, col='blue')

print(paste0("DESeq2 RMSE:",  sqrt(mean(deseq2_err^2))))
print(paste0("EdgeR RMSE:",  sqrt(mean(edgeR_err^2))))
print(paste0("NBSR RMSE:",  sqrt(mean(nbsr_err^2))))

ii <- which(abs(fc$alpha_null - fc$alpha_alt) > 0)
mean((deseq2_err[ii])^2)
mean((edgeR_err[ii])^2)
mean((nbsr_err[ii])^2)

# Results table.

## DESeq2
deseq2_lb <- res2$log2FoldChange - 1.96*res2$lfcSE
deseq2_ub <- res2$log2FoldChange + 1.96*res2$lfcSE
deseq2_coverage <- (fc$log2_fc > deseq2_lb) & (fc$log2_fc < deseq2_ub)
mean(deseq2_coverage)

## NBSR
results_path <- paste0(data_path, "/", level1, "_", level2)
nbsr_logRR <- as.matrix(read.csv(paste0(results_path, "/nbsr_logRR.csv"), header = F))
nbsr_logRR_sd <- as.matrix(read.csv(paste0(results_path, "/nbsr_logRR_sd.csv"), header = F))

nbsr_log_fc <- nbsr_logRR[1,]
nbsr_log_fc_sd <- nbsr_logRR_sd[1,]
nbsr_lb <- (nbsr_log_fc - 1.96*nbsr_log_fc_sd)
nbsr_ub <- (nbsr_log_fc + 1.96*nbsr_log_fc_sd)
nbsr_coverage <- (fc$log_fc > nbsr_lb) & (fc$log_fc < nbsr_ub)
mean(nbsr_coverage)

# Or, we can convert to log2.
nbsr_log2_fc <- log2(exp(nbsr_logRR[1,]))
nbsr_log2_fc_sd <- log2(exp(nbsr_logRR_sd[1,]))
nbsr_lb <- (nbsr_log2_fc - 1.96*nbsr_log2_fc_sd)
nbsr_ub <- (nbsr_log2_fc + 1.96*nbsr_log2_fc_sd)
nbsr_coverage <- (fc$log2_fc > nbsr_lb) & (fc$log2_fc < nbsr_ub)
mean(nbsr_coverage)

nbsr_stats <- nbsr_log_fc/nbsr_log_fc_sd
nbsr_pvals <- 2*pnorm(abs(nbsr_stats), lower.tail = FALSE)
nbsr_adj_pvals <- p.adjust(nbsr_pvals, method = "BH")

# Construct result tables for all three methods.
deseq2_dt <- data.table(log2_fc=res2$log2FoldChange, lb=deseq2_lb, ub=deseq2_ub, pval=res2$padj, method="DESeq2")
edgeR_dt <- data.table(log2_fc=edgeR_results$table$logFC, lb=NA, ub=NA, pval=edgeR_results$table$FDR, method="EdgeR")
nbsr_dt <- data.table(log2_fc=nbsr_log2_fc, lb=nbsr_lb, ub=nbsr_ub, pval=nbsr_adj_pvals, method="NBSR")

# Quantify performance at various threshold values.
gt_sig_idx <- which(fc$log2_fc != 0)
gt_not_sig_idx <- which(fc$log2_fc == 0)

deseq2_ret <- evaluate(deseq2_dt$pval, gt_sig_idx, gt_not_sig_idx, thresholds)
edgeR_ret <- evaluate(edgeR_dt$pval, gt_sig_idx, gt_not_sig_idx, thresholds)
nbsr_ret <- evaluate(nbsr_dt$pval, gt_sig_idx, gt_not_sig_idx, thresholds)


