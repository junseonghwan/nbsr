rm(list=ls())
set.seed(1)
library(data.table)
library(ggplot2)
library(MCMCpack)
library(seqinr)
library(stringr)

source("scripts/functions.R")

output_path <- "data/test"
if (!dir.exists(output_path)) {
  dir.create(output_path, recursive = T)
}

sample_count <- 10
K <- 200
swap_count <- 40
seq_depth <- 10^(6:8)

miRNA_polya_est <- fread(file = "data/microRNAome_dir_params.csv")
dat <- miRNA_polya_est[cell_type == "t_lymphocyte_cd4"]

# Select top K features.
setorder(dat, -alpha_hat)
dat[,sum(alpha_hat)]
dat[1:K,sum(alpha_hat)]

sub_dat <- dat[1:K]

alpha_null <- copy(sub_dat)
alpha_alt <- copy(sub_dat)

low_idxs <- which(alpha_null$alpha_hat <= 1)
high_idxs <- which(alpha_null$alpha_hat > 1)
length(high_idxs)

# Sample pairs of indices
idx1 <-  sample(x = low_idxs, size = swap_count, replace = FALSE)
idx2 <-  sample(x = high_idxs, size = swap_count, replace = FALSE)
swap_idx_pairs <- cbind(idx1, idx2)

alpha_alt$alpha_hat[idx1] <- alpha_null$alpha_hat[idx2]
alpha_alt$alpha_hat[idx2] <- alpha_null$alpha_hat[idx1]

alpha_null[,alpha_bar := normalize(alpha_hat)]
alpha_alt[,alpha_bar := normalize(alpha_hat)]

alpha_alt[,alpha := alpha_hat]
alpha_null[,alpha := alpha_hat]

ret <- generate_data(alpha_alt, alpha_null, sample_count, sample_count, reads_min_max = c(seq_depth[1], tail(seq_depth,1)), max_percent_miRNA = 0.5)
row_idxs <- rowSums(ret$Y > 0) > 1
sum(row_idxs)
X <- ret$X
Y <- ret$Y[row_idxs,]
fc <- ret$log_fc[row_idxs]
write.table(X, file = paste0(output_path, "/X.csv"), quote = F, col.names = T, row.names = F, sep=",")
write.table(Y, file = paste0(output_path, "/Y.csv"), quote = F, col.names = T, row.names = T, sep=",")
fwrite(fc, file = paste0(output_path, "/fc.csv"))
