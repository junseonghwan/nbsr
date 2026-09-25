library(DESeq2)
library(ggplot2)

immune_data <- readRDS("data/immune_data/immune_data.rds")

rep_path <- "data/immune_data/"
covariates <- c("trt")
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

# Also Extract the size factors and the dispersion trend parameters.
size_factors <- sizeFactors(dds2)
fitted_dispersions <- mcols(dds2)$dispFit
disp_function <- dispersionFunction(dds2)
coefs <- attr(disp_function, "coefficients")
coefs_df <- as.data.frame(t(coefs))
write.table(size_factors, file = paste0(rep_path, "/deseq2_size_factors.csv"), row.names = F, quote =F, col.names = F, sep=",")
write.table(fitted_dispersions, file = paste0(rep_path, "/deseq2_fitted_dispersions.csv"), row.names = F, quote =F, col.names = F, sep=",")
write.table(coefs_df, file = paste0(rep_path, "/deseq2_dispersion_trend_coefs.csv"), row.names = F, quote =F, col.names = T, sep=",")

normalized_counts <- counts(dds2, normalized = TRUE)
mu_bar <- rowMeans(normalized_counts)
write.table(mu_bar, file = "data/immune_data/deseq2_mu_bar.csv", row.names = F, quote =F, col.names = F, sep=",")
phi_trend <- coefs[2] / mu_bar + coefs[1]
trend_df <- data.frame(
  mu  = mu_bar,
  phi = phi_trend
)
ggplot(trend_df, aes(x = mu, y = phi)) +
  geom_point(alpha = 0.2, size = 0.5) +
  geom_line(data = trend_df, aes(x = mu, y = phi),
            color = "red", linewidth = 1) +
  scale_x_log10() +
  scale_y_log10() +
  labs(x = expression(bar(mu)[j]),
       y = expression(hat(phi)[j]),
       title = "Mean-Dispersion Trend") +
  theme_bw()

idx <- 501
plot(deseq2_mu[idx,], Y[idx,])
mean(abs(Y[idx,] - deseq2_mu[idx,]))
phi <- fitted_dispersions[idx]

beta_deseq2 <- coef(dds2) 

sum(Y[,74])
csums <- colSums(Y)
hist(csums)
csums[1]
csums[74]
which.max(csums)

mu <- deseq2_mu[idx,]
sd_deseq2 <- sqrt(mu + phi * (mu^2))
mean(sd_deseq2[X$trt == "Pre"])
mean(sd_deseq2[X$trt == "On"])
