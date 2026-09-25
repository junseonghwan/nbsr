import numpy as np
import torch
import pandas as pd
#from numba import njit

import os
import sys

def kron_hessian(X, pi, A, B, dim):
    """Hessian of a softmax-mean NB log-likelihood w.r.t. the flat coefficient vector, covariate-major.

    For the NBSR models every per-sample Hessian is a Kronecker product x_i x_i' (x) M_i, where the
    dim x dim matrix M_i collects, over features j, the terms

        A_ij (1[j=k] - pi_ik)(1[j=k'] - pi_ik')  +  B_ij pi_ik (1[k=k'] - pi_ik').

    Summing over j in closed form gives
        M_i = diag(A_i) - A_i pi_i' - pi_i A_i' + (sum_j A_ij) pi_i pi_i' + (sum_j B_ij)(diag(pi_i) - pi_i pi_i'),
    restricted to the first `dim` features (the pivot feature, if any, has no coefficients), while the sums
    over j run over all J features. The full Hessian is sum_i x_i x_i' (x) M_i, returned as a
    (P*dim, P*dim) matrix with index d*dim + k for covariate d and feature k.

    X: (N, P); pi: (N, J); A, B: (N, J). All computations stay on the device of pi.
    """
    dtype = pi.dtype
    X = X.to(dtype)
    p = pi[:, :dim]
    A_d = A[:, :dim]
    sumA = A.sum(1)[:, None, None]
    sumB = B.sum(1)[:, None, None]
    pp = p.unsqueeze(2) * p.unsqueeze(1)                                      # (N, dim, dim)
    M = (torch.diag_embed(A_d)
         - A_d.unsqueeze(2) * p.unsqueeze(1) - p.unsqueeze(2) * A_d.unsqueeze(1)
         + sumA * pp
         + sumB * (torch.diag_embed(p) - pp))
    H = torch.einsum("nd,ne,nkl->dkel", X, X, M)
    return H.reshape(X.shape[1] * dim, X.shape[1] * dim)


def construct_tensor_from_coldata(coldata_pd, column_names, sample_count, include_intercept=True):
    X_intercept = torch.ones(sample_count, 1, dtype=torch.float64)
    # column data does not exist -> fit a model with just the intercept.
    if coldata_pd is None or len(column_names) == 0:
        if include_intercept:
            return (X_intercept, {})
        else:
            return None

    # column data exists -> check that the column names specified in the config exists in the column data.
    # if no, exit with error.
    # if yes, retrieve the relevant column data and convert to dummy variables and return a tensor with intercept term prepended.
    X_df_names = coldata_pd.columns.to_list()
    for column_name in column_names:
        exists = column_name in X_df_names
        print("Column name " + column_name + " exists? " + str(exists))
        if not exists:
            print(column_name + " does not exist in the data frame.")
            sys.exit(1)
    X_df = coldata_pd[column_names]
    X_design = pd.get_dummies(X_df, drop_first=True, dtype=int)
    X_tensor = torch.tensor(X_design.to_numpy(), dtype=torch.float64)
    if include_intercept:
        X_tensor = torch.cat([X_intercept, X_tensor], dim = 1)
    variable_map = {}
    for idx, column in enumerate(X_design.columns):
        variable_map[column] = idx
    return (X_tensor, variable_map)

def read_file_if_exists(file_path):
    if file_path is None:
        return None
    if os.path.exists(file_path):
        return np.loadtxt(file_path)
    return None

def create_directory(path):
    if not os.path.exists(path):
    	os.makedirs(path)
