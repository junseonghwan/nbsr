import numpy as np
import torch
from scipy.special import logsumexp
import pandas as pd

def get_beta(model):
    return(model.beta.reshape((model.X.shape[1], model.dim)))

def summarize_(X, mu, beta, size_factor, pivot):
    print(mu.shape, X.shape, beta.shape)
    log_unnorm_expr = mu + np.matmul(X, beta)
    if pivot:
        sample_count = X.shape[0]
        log_unnorm_expr = np.column_stack((log_unnorm_expr, np.zeros(sample_count)))
    log_norm = logsumexp(log_unnorm_expr, 1)
    pi_fitted = np.exp(log_unnorm_expr - log_norm[:,None])
    Y_fitted = size_factor[:,None] * pi_fitted
    return((Y_fitted, pi_fitted))

def summarize(model, X, s, pivot, reshape_beta):
    return(summarize_(X, model.mu.data.numpy(), get_beta(model).data.numpy() if reshape_beta else model.beta.data.numpy(), s, pivot))

def FisherInformation(model):
    H = torch.autograd.functional.hessian(model.log_likelihood, (model.mu, model.beta))
    #H_mu = hessian[0][0]
    #H_beta = hessian[1][1]
    #H_mu = torch.func.hessian(model.log_likelihood, 0)(model.mu.data, model.beta.data)
    #H_beta = torch.func.hessian(model.log_likelihood, 1)(model.mu.data, model.beta.data)
    I = -torch.row_stack((torch.column_stack((H[0][0], H[0][1])),
                         torch.column_stack((H[1][0], H[1][1]))))
    return(I)

# Computes log P(Y = k | z, w_1) / log P(Y = k | z, w_0).
def logRR(model, var, w0, w1):
    # Get a copy of the design matrix.
    print(var, w0, w1)
    X_df = pd.get_dummies(model.X_df, drop_first=True, dtype=int)
    Z = X_df.to_numpy()
    mu = model.mu.data.numpy()
    beta_reshaped = get_beta(model).data.numpy()
    colnames = X_df.columns.values

    sample_count = model.sample_count
    K = model.rna_count
    P = model.covariate_count
    dim = (K-1) * P
    zeros = np.zeros(sample_count)

    var_level0 = "{varname}_{levelname}".format(varname=var, levelname=w0)
    var_level1 = "{varname}_{levelname}".format(varname=var, levelname=w1)
    col_idx0 = np.where(colnames == var_level0)[0]
    col_idx1 = np.where(colnames == var_level1)[0]
    # Zero out z corresponding to var.
    for i,colname in enumerate(colnames):
        if var in colname:
            Z[:,i] = 0

    Xbeta = mu + np.matmul(Z, beta_reshaped)
    A = np.zeros((K, dim))

    if len(col_idx0) > 0:
        Xbeta0 = Xbeta + beta_reshaped[col_idx0,:]
        beta_kprime0 = np.column_stack((beta_reshaped[col_idx0,:], 0))
        A[np.arange(K-1), np.arange(col_idx0, dim, P)] = -1
    else:
        beta_kprime0 = np.zeros((1, K))
        Xbeta0 = Xbeta

    if len(col_idx1) > 0:
        Xbeta1 = Xbeta + beta_reshaped[col_idx1,:]
        beta_kprime1 = np.column_stack((beta_reshaped[col_idx1,:], 0))
        A[np.arange(K-1), np.arange(col_idx1, dim, P)] = 1
    else:
        beta_kprime1 = np.zeros((1, K))
        Xbeta1 = Xbeta

    Xbeta0 = np.column_stack((Xbeta0, zeros))
    Xbeta1 = np.column_stack((Xbeta1, zeros))
    log_norm0 = logsumexp(Xbeta0, 1)
    log_norm1 = logsumexp(Xbeta1, 1)

    offset = (beta_kprime1 - beta_kprime0).transpose()
    logRRi = offset + log_norm0 - log_norm1
    logRRi = logRRi.transpose()
    #log2RRi = np.log2(np.exp(logRRi))

    pi0_hat = np.exp(Xbeta0 - log_norm0[:,None])
    pi1_hat = np.exp(Xbeta1 - log_norm1[:,None])
    pi_diff = pi0_hat - pi1_hat
    
    pi0_hat_ = pi0_hat[:,np.arange(K-1)]
    pi1_hat_ = pi1_hat[:,np.arange(K-1)]
    pi_diff_ = pi_diff[:,np.arange(K-1)]

    B = pi_diff_[:,:,np.newaxis] * Z[:,np.newaxis,:]
    if len(col_idx0) > 0:
        B[:,:,col_idx0] = pi0_hat_[:,:,np.newaxis]
    if len(col_idx1) > 0:
        B[:,:,col_idx1] = -pi1_hat_[:,:,np.newaxis]
        
    B = np.reshape(B, (sample_count, (K-1)*P, -1))
    B_ = np.tile(B, K)
    # Transpose it so that we have the Jacobian in the desired dimension.
    B_ = np.transpose(B_, (0, 2, 1))
    
    A = np.zeros((sample_count, K, (K-1) * P))
    if len(col_idx0) > 0:
        A[np.arange(sample_count)[:,None], np.arange(K-1), np.arange(col_idx0, dim, P)] = -1
    if len(col_idx1) > 0:
        A[np.arange(sample_count)[:,None], np.arange(K-1), np.arange(col_idx1, dim, P)] = 1

    Jbeta = A + B_
    Jmu = np.tile(pi_diff_[:,:,np.newaxis], K)
    Jmu = np.transpose(Jmu, (0, 2, 1))
    J = np.concatenate((Jmu, Jbeta), axis = 2)
    
    I = FisherInformation(model)
    Sigma = torch.inverse(I)

    cov_mat = np.matmul(J, Sigma.data.numpy()) @ J.transpose(0, 2, 1)
    sd_est = np.sqrt(np.diagonal(cov_mat, axis1=1, axis2=2))

    return(logRRi, sd_est, pi0_hat, pi1_hat)
