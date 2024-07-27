rm(list=ls())
set.seed(1)

library(data.table)
library(DESeq2)
library(edgeR)
library(ggplot2)

data_path <- "data/test/"
X <- fread(paste0(data_path, "/X.csv"))
Y <- as.matrix(read.csv(paste0(data_path, "/Y.csv")))
fc <- fread(paste0(data_path, "/fc.csv"))

# Run DESeq2.
se <- SummarizedExperiment(assays = list(counts = Y), colData = X)
dds2 <- DESeqDataSet(se, ~ trt)
dds2 <- DESeq(dds2)
res2 <- results(dds2, contrast = c("trt", "alt", "null"))

# Run EdgeR.
d <- edgeR::DGEList(counts=Y)
d <- calcNormFactors(d)
design_mat <- model.matrix(~ trt, X)
d <- edgeR::estimateDisp(d, design = design_mat)
fit <- glmFit(d, design_mat)
lrt <- glmLRT(fit, contrast=c(0, -1))
edgeR_results <- topTags(lrt, n=Inf, sort.by="none")

R <- colSums(Y)
nbsr_pi <- read.csv(paste0(output_path, "/nbsr_pi.csv"), header = F)
nbsr_phi <- as.matrix(read.csv(paste0(output_path, "/nbsr_dispersion.csv"), header = F))
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

