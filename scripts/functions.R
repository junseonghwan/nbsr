decorate_figure <- function(pl, xlab_text, ylab_text, xtext_rotate_angle=0, vjust=0, hjust=0)
{
  pl <- pl + ylab(str_wrap(ylab_text, width=28)) +
             xlab(str_wrap(xlab_text, width=20)) +
             theme(plot.tag = element_text(size = 24, 
                   margin = margin(r = -20), 
                   face = "bold")) +
             theme(axis.title.x = element_text(size = 16)) + 
             theme(axis.title.y = element_text(size = 16)) +
             theme(axis.text = element_text(size=14))
  if (abs(xtext_rotate_angle) > 0) {
    pl <- pl + theme(axis.text.x = element_text(angle = xtext_rotate_angle, vjust = vjust, hjust = hjust))
  }
  
  return(pl)
}
polish_cell_type <- function(x) {
  polished_cell_type <- tools::toTitleCase(x)
  polished_cell_type <- gsub(pattern = "_", replacement = " ", x = polished_cell_type)
  polished_cell_type <- gsub(pattern = "cd", replacement = "CD", x = polished_cell_type)
  return(polished_cell_type)
}
find_cutoff <- function(x, p)
{
  s <- sum(x)
  sorted_props <- sort(x/s, decreasing = T)
  cumprops <- cumsum(sorted_props)
  return(head(which(cumprops >= p), 1))
}
cutoff_reads <- function(x, p)
{
  s <- sum(x)
  sorted_counts <- sort(x, decreasing = T)
  sorted_props <- sort(x/s, decreasing = T)
  cumprops <- cumsum(sorted_props)
  i <- head(which(cumprops >= p), 1)
  return(cumsum(sorted_counts)[i])
}
get_sorted_props <- function(x, p=1)
{
  s <- sum(x)
  oidx <- order(x, decreasing = T)
  cumprops <- cumsum(x[oidx])/s
  i <- head(which(cumprops >= p), 1)
  ff <- x[oidx][1:i]/s
  return(ff)
}
get_sorted_props2 <- function(x, p=1)
{
  s <- sum(x)
  oidx <- order(x, decreasing = T)
  cumprops <- cumsum(x[oidx])/s
  ff <- x[oidx]/s
  return(ff)
}
entropy <- function(x) {
  y <- x[x != 0]
  return(-sum(y * log(y)))
}

kl_uniform <- function(x) {
  y <- x[x != 0]
  N <- length(x)
  return(sum(y * log(N*y)))
}
# Normalized the parameters.
normalize <- function(alpha) {
  alpha_bar <- alpha / sum(alpha)
  return(alpha_bar)
}
compute_max_entropy <- function(feature_count) {
  return(entropy(rep(1/feature_count, feature_count)))
}
compute_iqr <- function(x) {
  y <- quantile(x, c(0.25, 0.75))
  return(y[2] - y[1])
}
rmse <- function(x, y)
{
  return(mean(sqrt((x - y)**2)))
}

binning <- function(alpha, x0=1, x1=100)
{
  bin <- rep(0, length(alpha))
  bin[alpha < x0] <- 1
  bin[alpha >= x0 & alpha < x1] <- 2
  bin[alpha >= x1] <- 3
  return(bin)
}

evaluate_simul <- function(results_path, case_count, rep_count)
{
  results <- list()
  results_by_bin <- list()
  for (j in 1:case_count)
  {  
    case_path <- paste0(results_path, "/case", j)
    for (i in 1:rep_count)
    {
      rep_path <- paste0(case_path, "/rep", i)
      nbsr_results <- fread(paste0(rep_path, "/nbsr_results.csv"))
      nbsr_log_rr <- read.csv(paste0(rep_path, "/nbsr_logRR.csv"), header = F)
      nbsr_log_rr_sd <- read.csv(paste0(rep_path, "/nbsr_logRR_sd.csv"), header = F)
      deseq2_results <- fread(paste0(rep_path, "/deseq2_results.csv"))
      fc <- fread(paste0(rep_path, "/fc.csv"))
      
      log2_rr <- as.numeric(nbsr_log_rr[1,])/log(2)
      log2_rr_sd <- as.numeric(nbsr_log_rr_sd[1,])/log(2)
      
      rr.df <- data.table()
      rr.df$bias <- mean(fc$log2_fc - log2_rr)
      rr.df$rmse <- sqrt(mean((fc$log2_fc - log2_rr)^2))
      rr.df$mae <- mean(abs(fc$log2_fc - log2_rr))
      lb_rr <- log2_rr - 1.96 * log2_rr_sd
      ub_rr <- log2_rr + 1.96 * log2_rr_sd
      rr.df$coverage <- mean(fc$log2_fc > lb_rr & fc$log2_fc < ub_rr)
      rr.df$method <- "NBSR"
      
      deseq2.df <- data.table()
      deseq2.df$bias <- mean(fc$log2_fc - deseq2_results$log2FoldChange)
      deseq2.df$rmse <- sqrt(mean((fc$log2_fc - deseq2_results$log2FoldChange)^2))
      deseq2.df$mae <- mean(abs(fc$log2_fc - deseq2_results$log2FoldChange))
      lb_deseq2 <- deseq2_results$log2FoldChange - 1.96 * deseq2_results$lfcSE
      ub_deseq2 <- deseq2_results$log2FoldChange + 1.96 * deseq2_results$lfcSE
      deseq2.df$coverage <- mean(fc$log2_fc > lb_deseq2 & fc$log2_fc < ub_deseq2)
      deseq2.df$method <- "DESeq2"
      
      results[[j*rep_count + i]] <- rbind(rr.df, deseq2.df)
      results[[j*rep_count + i]]$rep <- i
      results[[j*rep_count + i]]$case <- j
      
      # We are interested in classifying the error based on the bin.
      fc[,bin := binning(alpha_null)]
      fc[,log2fc_rr := log2_rr]
      fc[,log2fc_rr_sd := log2_rr_sd]
      fc[,":="(log2fc_rr_lb = lb_rr, log2fc_rr_ub = ub_rr)]
      fc[,":="(log2fc_deseq2_lb = lb_deseq2, log2fc_deseq2_ub = ub_deseq2)]
      fc[,log2fc_deseq2 := deseq2_results$log2FoldChange]
      fc[,log2fc_deseq2_sd := deseq2_results$lfcSE]
      ret_rr <- fc[,
                   .(rmse = rmse(log2_fc, log2fc_rr), 
                     coverage = mean(log2_fc >= log2fc_rr_lb & log2_fc <= log2fc_rr_ub),
                     sd = mean(log2fc_rr_sd),
                     method = "NBSR"),
                   by=.(bin)]
      
      ret_deseq2 <- fc[,
                       .(rmse = rmse(log2_fc, log2fc_deseq2),
                         coverage = mean(log2_fc >= log2fc_deseq2_lb & log2_fc <= log2fc_deseq2_ub),
                         sd = mean(log2fc_deseq2_sd),
                         method = "DESeq2"),
                       by=.(bin)]
      ret <- rbind(ret_rr, ret_deseq2)
      ret$rep <- i
      ret$case <- j
      results_by_bin[[j*rep_count + i]] <- ret
    }
  }
  ret.dt <- rbindlist(results)
  results_by_bin.dt <- rbindlist(results_by_bin)
  return(list(results_by_bin.dt, ret.dt))
}
sample_dirichlet_mult <- function(alpha, n, reads_min_max)
{
  sample_names <- paste0("Sample", 1:n)
  probs <- rdirichlet(n, alpha)

  Y <- apply(probs, 1, function(pp) {
    size_factor <- runif(1, min = reads_min_max[1], max = reads_min_max[2])
    rmultinom(1, size = size_factor, prob = pp)
  })

  return(list(Y=Y, pi=t(probs)))
}
prec_recall <- function(predicted_idx, gt_idx)
{
  fp <- sum(!(predicted_idx %in% gt_idx))
  tp <- sum((gt_idx %in% predicted_idx))
  fn <- sum(!(gt_idx %in% predicted_idx))
  precision <- ifelse(tp + fp == 0, 0, tp / (tp + fp))
  recall <- tp / (tp + fn)
  return(list(prec=precision, recall=recall))
}
prec_recall2 <- function(predicted_idx, gt_idx)
{
  tp <- length(intersect(predicted_idx, gt_idx))
  fp <- length(setdiff(predicted_idx, gt_idx))
  fn <- length(setdiff(gt_idx, predicted_idx))
  precision <- ifelse(tp + fp == 0, 0, tp / (tp + fp))
  recall <- tp / (tp + fn)
  return(list(prec=precision, recall=recall))
}
performance_metrics <- function(pred_sig_idx, pred_not_sig_idx, gt_sig_idx, gt_not_sig_idx)
{
  # Let's measure a bunch of stuff here.
  # 1. accuracy
  # 2. precision and recall/sensitivity (tpr)
  # 3. specificity (tnr)
  # 4. fpr and fnr
  tp <- sum(pred_sig_idx %in% gt_sig_idx)
  tn <- sum(pred_not_sig_idx %in% gt_not_sig_idx)
  fp <- sum(pred_sig_idx %in% gt_not_sig_idx)
  fn <- sum(pred_not_sig_idx %in% gt_sig_idx)
  total_count <- length(gt_sig_idx) + length(gt_not_sig_idx)
  print((tp + tn + fp + fn) == total_count)
  accuracy <- (tp + tn) / total_count
  ret <- prec_recall2(pred_sig_idx, gt_sig_idx)
  prec <- ret$prec
  recall <- ret$recall
  tpr <- tp/(tp + fn)
  tnr <- tn/(tn + fp)
  fnr <- 1 - tpr
  fpr <- 1 - tnr
  return(data.table(acc=accuracy, prec=prec, tpr=recall, tnr=tnr, fpr=fpr, fnr=fnr))
}
generate_data <- function(unperturbed_dat, perturbed_dat, n_A, n_B, reads_min_max=c(10^6,10^8), max_percent_miRNA=0.5)
{
  nn <- n_A + n_B
  sample_names <- paste0("Sample", 1:nn)
  pi_null <- rdirichlet(n_A, unperturbed_dat[,alpha])
  pi_alt <- rdirichlet(n_B, perturbed_dat[,alpha])
  
  foo <- function(pi) {
    n <- dim(pi)[1]
    lib_size <- rep(0, n)
    size_factor <- rep(0, n)
    Y <- matrix(0, ncol = n, nrow = dim(pi)[2])
    for (i in 1:n)
    {
      lib_size[i] <- runif(1, min = reads_min_max[1], max = reads_min_max[2])
      percent_miRNA <- runif(1, 0, max_percent_miRNA)
      size_factor[i] <- ceiling(lib_size[i] * percent_miRNA)
      y <- rmultinom(1, size = size_factor[i], prob = pi[i,])
      Y[,i] <- y
    }
    return(list(Y=Y, lib_size=lib_size, miRNA_reads=size_factor))
  }
  ret_null <- foo(pi_null)
  ret_alt <- foo(pi_alt)

  Y <- cbind(ret_null$Y, ret_alt$Y)
  pi <- t(rbind(pi_null, pi_alt))
  
  X <- data.frame(sample=sample_names, 
                  trt = c(rep("null", n_A), rep("alt", n_B)),
                  lib_size = c(ret_null$lib_size, ret_alt$lib_size),
                  miRNA_reads = c(ret_null$miRNA_reads, ret_alt$miRNA_reads))
  
  log_fc.dt <- data.table(miRNA = unperturbed_dat$miRNA,
                          alpha_null = unperturbed_dat[,alpha], 
                          alpha_alt = perturbed_dat[,alpha],
                          alpha_null_bar = unperturbed_dat[,alpha_bar], 
                          alpha_alt_bar = perturbed_dat[,alpha_bar])
  log_fc.dt[,log2_fc := log2(alpha_alt_bar) - log2(alpha_null_bar)]
  log_fc.dt[,log_fc := log(alpha_alt_bar) - log(alpha_null_bar)]
  
  rownames(Y) <- unperturbed_dat$miRNA
  colnames(Y) <- sample_names
  return(list(Y=Y, X=X, pi=pi, log_fc=log_fc.dt))
}
compute_expected_zeros_NB <- function(mu, phi)
{
  size <- 1/phi
  pzero <- dnbinom(0, size, mu = mu)
  return(if (is.vector(mu)) sum(pzero) else colSums(pzero))
}
softplus <- function(x, beta = 1)
{
  y <- (1/beta) * log(1 + exp(beta * x))
  return(y)
}
inv_softplus <- function(y, beta = 1)
{
  x <- log(exp(beta*y) - 1) / beta
  return(x)
}
sigmoid <- function(x) {
  pi <- exp(x) / (1 + exp(x))
  return(pi)
}
logistic <- function(x, L=1, k=1, x0=0) {
  L / (1 + exp(-k * (x - x0)))
}
compute_zero_prob <- function(mu, phi)
{
  size <- 1/phi
  pzero <- dnbinom(0, size, mu = mu)
  return(pzero)
}
plot_sparsity <- function(feature_sparsity, dispersion) {
  daf <- data.frame(feature_sparsity, dispersion)
  pl <- ggplot(daf, aes(feature_sparsity, dispersion)) + geom_point() + theme_bw()
  pl <- pl + xlab("Proportion of zeros (sparsity)") + ylab("Estimated dispersion")
  return(pl)
}
fit_deseq2 <- function(se, design_mat, fit_type = "local")
{
  dds <- DESeqDataSet(se, design = design_mat)
  dds <- DESeq(dds, fitType = fit_type)
  rdat <- rowData(dds)
  sparsity <- rowMeans(assay(se) == 0)
  return(list(fit=dds, pl=plot_sparsity(sparsity, rdat$dispersion)))
}
fit_edgeR <- function(Y, design_mat)
{
  d <- edgeR::DGEList(counts=Y)
  d <- calcNormFactors(d)
  d <- edgeR::estimateDisp(d, design = design_mat)
  fit <- glmFit(d, design_mat)
  sparsity <- rowMeans(Y == 0)
  return(list(fit=fit, pl=plot_sparsity(sparsity, fit$dispersion)))
}
bias_plots <- function(subplt.dt, title, fig_lables="")
{
  pl1 <- ggplot(subplt.dt, aes(percent_miRNAs, observed - predicted, size=filtered_miRNA_reads)) + geom_point() + theme_bw()
  #pl1 <- pl1 + ggtitle(title)
  pl1 <- pl1 + xlab("% miRNAs") + ylab("Observed - Predicted") + labs(size = "miRNA Reads")
  pl1 <- pl1 + geom_hline(yintercept = 0)
  
  pl2 <- ggplot(subplt.dt, aes(filtered_miRNA_reads, observed - predicted, size=percent_miRNAs)) + geom_point() + theme_bw()
  pl2 <- pl2 + geom_hline(yintercept = 0)
  pl2 <- pl2 + scale_x_log10()
  pl2 <- pl2 + labs(size = "% miRNAs") + ylab("Observed - Predicted") + xlab("miRNA Reads (log10 scale)")
  
  pl <- ggarrange(pl1, pl2, 
                     ncol = 1, nrow = 2, align = "hv",
                     labels = fig_lables)
  pl <- annotate_figure(pl, top = text_grob(title, size = 14, face = "bold"))
  return(pl)
}
# Compute precision-recall as the p-value threshold is adjusted.
evaluate <- function(pvals, gt_sig_idx, gt_not_sig_idx, thresholds) {
  return_list <- list()
  for (i in 1:length(thresholds))
  {
    threshold <- thresholds[i]
    return_list[[i]] <- performance_metrics(which(pvals <= threshold), 
                                            which(pvals > threshold),
                                            gt_sig_idx,
                                            gt_not_sig_idx)
  }
  ret <- rbindlist(return_list)
  ret$threshold <- thresholds
  return(ret)
}
compute_cv <- function(dat_dt) {
  dat_dt[,variance:=alpha_bar * (1 - alpha_bar) / (sum(alpha) + 1)]
  dat_dt[,cv := sqrt(variance) / alpha_bar]
  dat_dt[,phi := cv^2]
}
