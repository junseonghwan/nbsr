import numpy as np
import torch
import scipy

def summarize_(X, mu, beta, size_factor, pivot, reshape=False):
    rna_count = len(mu)
    covariate_count = X.shape[1]
    beta = beta.reshape((covariate_count, rna_count))
    log_unnorm_expr = mu + np.matmul(X, beta)
    if pivot:
        sample_count = X.shape[0]
        log_unnorm_expr = np.column_stack((np.zeros(sample_count), log_unnorm_expr))
    log_norm = scipy.special.logsumexp(log_unnorm_expr, 1)
    pi_fitted = np.exp(log_unnorm_expr - log_norm[:,None])
    Y_fitted = size_factor[:,None] * pi_fitted
    return((Y_fitted, pi_fitted))

def summarize(model, X, s, pivot, reshape_beta):
    return(summarize_(X, model.mu.data.numpy(), model.beta.data.numpy(), s, pivot, reshape_beta))
