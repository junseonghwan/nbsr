import numpy as np
import torch
from scipy.special import logsumexp
import pandas as pd
#from numba import njit

import os
import sys

# @njit(cache=True) 
def hessian_trended_nbsr(X, Y, pi, p, r, aa, cc, b_1, pivot=True):
    N, P = X.shape
    J    = Y.shape[1]
    dim = J - 1 if pivot else J
    JP   = dim * P

    # allocate outputs
    #g = np.zeros(JP, dtype=np.float64)
    H = np.zeros((JP, JP), dtype=np.float64)

    # loop #1: over samples i
    for i in range(N):
        x_i   = X[i]     # (P,)
        y_i   = Y[i]     # (J,)
        pi_i = pi[i]    # (J,)
        p_i = p[i]
        r_i = r[i]
        a_i = aa[i]
        c_i = cc[i]

        # accumulate gradient
        # w1 = r_i * (1 - p_i) * (1 + b_1)
        # w2 = y_i * p_i * (1 + b_1)
        # w3 = r_i * a_i * b_1
        # grad_w = (-w1 + w2 - w3)
        # g[idx_d_k] += grad_w[j] * x_i[d] * delta_j_k

        # accumulate Hessian
        for j in range(J):
            for r_idx in range(JP):
                k, d = divmod(r_idx, P)
                idx_k = d*dim + k
                delta_j_k = (1.0 if j == k else 0.0) - pi_i[k]
                for c_idx in range(JP):
                    kp, dp = divmod(c_idx, P)
                    idx_kp = dp*dim + kp
                    delta_j_kp = (1.0 if j == kp else 0.0) - pi_i[kp]
                    delta_k_kp = (1.0 if k == kp else 0.0) - pi_i[kp]

                    xx = x_i[d] * x_i[dp]
                    delta_j_k_kp = delta_j_k * delta_j_kp

                    term1_1 = b_1 * delta_j_k_kp
                    term1_2 = -p_i[j] * (1 + b_1) * delta_j_k_kp
                    term1_3 = pi_i[k] * delta_k_kp
                    term1 = r_i[j] * (1 - p_i[j]) * (1 + b_1) * xx * (term1_1 + term1_2 + term1_3)

                    term2_1 = (1 - p_i[j]) * (1 + b_1) * delta_j_k_kp
                    term2_2 = pi_i[k] * delta_k_kp
                    term2 = -y_i[j] * p_i[j] * (1 + b_1) * xx * (term2_1 + term2_2)
                    
                    term3_1 = a_i[j] * b_1 * delta_j_k_kp
                    term3_2 = (r_i[j] * b_1 * c_i[j] + (1 - p_i[j]) * (1 + b_1)) * delta_j_k_kp
                    term3_3 = a_i[j] * pi_i[k] * delta_k_kp
                    term3 = r_i[j] * b_1 * xx * (term3_1 + term3_2 + term3_3)

                    H[idx_k, idx_kp] += (term1 + term2 + term3)

    return H

# @njit(cache=True) 
def log_lik_gradients2(X, Y, pi, mu, phi, pivot=True):
    N, P = X.shape
    J    = Y.shape[1]
    dim = J - 1 if pivot else J
    JP   = dim * P

    # allocate outputs -- grad will be returned as 3d-matrix (J, J, P) with the axis=0 to be collapsed.
    # g[d,j,k] = \nabla_{k,d}
    g = np.zeros((J, J, P), dtype=np.float64)
    H = np.zeros((JP, JP), dtype=np.float64)
    #H = np.zeros((J, JP, JP), dtype=np.float64)
    
    # reciprocal dispersions
    r = 1.0 / phi
    var = mu + phi * (mu ** 2)
    D   = phi * (mu ** 2) / var

    I = np.eye(J)

    # loop #1: over samples i
    for i in range(N):
        x_i   = X[i]     # (P,)
        y_i   = Y[i]     # (J,)
        pi_i = pi[i]    # (J,)
        D_i = D[i]

        w1 = r * D_i
        w2 = y_i * (1.0 - D_i)
        grad_w = (w1 - w2)
        hess_w = w1 * (1.0 - D_i) + w2 * D_i

        # accumulate gradient g
        delta_j_k = I - pi_i[:,None].T # delta_j_k[j,k] = 1[j = k] - pi_i[k]
        ret = (-grad_w[:, None, None]         # shape (J,1,1)
                * delta_j_k[:, :, None]         # shape (J,J,1)
                * x_i[None, None, :]            # shape (1,1,P)
                ) # shape (J, J, P)
        g += ret

        # accumulate hessian H.
        for dim1 in range(JP):
            k, d = divmod(dim1, P)
            idx_k = d * dim + k
            for dim2 in range(JP):
                kp, dp = divmod(dim2, P)
                idx_kp = dp * dim + kp
                delta_k_kp = (1.0 if k == kp  else 0) - pi_i[kp]
                
                # This is slower than doing a loop over j.
                #H[:, idx_k, idx_kp] += (grad_w * (x_i[d] * x_i[dp]) * pi_i[k] * delta_k_kp - hess_w * (x_i[d] * x_i[dp]) * delta_j_k[:,k] * delta_j_k[:,kp])

                t1 = 0.0
                t2 = 0.0
                for j in range(J):
                    t1 += grad_w[j] * pi_i[k] * delta_k_kp
                    t2 += hess_w[j] * delta_j_k[j,k] * delta_j_k[j,kp]

                H[idx_k, idx_kp] += (t1 - t2) * x_i[d] * x_i[dp]
    return (g, H)

# @njit(cache=True) 
def hessian_nbsr(X, Y, pi, mu, phi, pivot=True):
    N, P = X.shape
    J    = Y.shape[1]
    dim = J - 1 if pivot else J
    JP   = dim * P

    # allocate outputs
    #g = np.zeros(JP, dtype=np.float64)
    H = np.zeros((JP, JP), dtype=np.float64)

    # reciprocal dispersions
    r = 1.0 / phi
    var = mu + phi * (mu ** 2)
    D   = phi * (mu ** 2) / var

    # loop #1: over samples i
    for i in range(N):
        x_i   = X[i]     # (P,)
        y_i   = Y[i]     # (J,)
        pi_i = pi[i]    # (J,)
        D_i = D[i]

        w1 = r * D_i
        w2 = y_i * (1.0 - D_i)
        grad_w = (w1 - w2)
        hess_w = w1 * (1.0 - D_i) + w2 * D_i

        # accumulate gradient g[d*dim + k] = sum_{i,j} grad_w[j]*(1[j=k] - pi_i[k]) * xi[d].
        # accumulate Hessian H[d*dim + k, d*dim + kp] = sum_{i,j} 
        for j in range(J):
            for k in range(dim):
                pi_ik = pi_i[k]
                ind_j_k = 1.0 if j == k else 0.0
                for d in range(P):
                    idx_k = d * dim + k
                    #g[idx_k] += -grad_w[j] * x_i[d] * (ind_j_k - pi_ik)
                    for kp in range(dim):
                        pi_ikp = pi_i[kp]
                        ind_k_kp = 1.0 if k == kp else 0.0
                        ind_j_kp = 1.0 if j == kp else 0.0
                        for dp in range(P):
                            idx_kp = dp * dim + kp
                            term1 = grad_w[j] * x_i[d] * x_i[dp] * pi_ik * (ind_k_kp - pi_ikp)
                            term2 = hess_w[j] * x_i[d] * x_i[dp] * (ind_j_k - pi_ik) * (ind_j_kp - pi_ikp)
                            H[idx_k, idx_kp] += (term1 - term2)
    return H

def construct_tensor_from_coldata(coldata_pd, column_names, sample_count, include_intercept=True):
    X_intercept = torch.ones(sample_count, 1)
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

def torch_rbf(x, a, b, c):
    # Ensuring x is a tensor
    if not torch.is_tensor(x):
        x = torch.tensor(x, dtype=torch.float32)

    # Reshape x to have three dimensions if it's not already
    if x.dim() == 1:
        x = x.unsqueeze(1)  # For a vector, make it N x 1
    x = x.unsqueeze(-1)  # Add an extra dimension for broadcasting: N x K x 1

    # a and c are vectors of length M. Reshape for broadcasting
    a = a.reshape(1, 1, -1)  # 1 x 1 x M
    c = c.reshape(1, 1, -1)  # 1 x 1 x M

    # Perform the operation
    # Broadcasting will align dimensions automatically
    result = torch.sum(a * torch.exp(-b * (x - c)**2), dim=-1)  # Sum over the last dimension

    return result

def read_file_if_exists(file_path):
    if file_path is None:
        return None
    if os.path.exists(file_path):
        return np.loadtxt(file_path)
    return None

def create_directory(path):
    if not os.path.exists(path):
    	os.makedirs(path)

def softplus_np(x):
    return np.log1p(np.exp(x))

def reshape(model, params):
    return(params.reshape((model.X.shape[1], model.dim)))

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

# Source: https://github.com/scipy/scipy/blob/v1.11.3/scipy/stats/_morestats.py#L4863-L5059
def false_discovery_control(ps, *, axis=0, method='bh'):
    """Adjust p-values to control the false discovery rate.

    The false discovery rate (FDR) is the expected proportion of rejected null
    hypotheses that are actually true.
    If the null hypothesis is rejected when the *adjusted* p-value falls below
    a specified level, the false discovery rate is controlled at that level.

    Parameters
    ----------
    ps : 1D array_like
        The p-values to adjust. Elements must be real numbers between 0 and 1.
    axis : int
        The axis along which to perform the adjustment. The adjustment is
        performed independently along each axis-slice. If `axis` is None, `ps`
        is raveled before performing the adjustment.
    method : {'bh', 'by'}
        The false discovery rate control procedure to apply: ``'bh'`` is for
        Benjamini-Hochberg [1]_ (Eq. 1), ``'by'`` is for Benjaminini-Yekutieli
        [2]_ (Theorem 1.3). The latter is more conservative, but it is
        guaranteed to control the FDR even when the p-values are not from
        independent tests.

    Returns
    -------
    ps_adusted : array_like
        The adjusted p-values. If the null hypothesis is rejected where these
        fall below a specified level, the false discovery rate is controlled
        at that level.

    See Also
    --------
    combine_pvalues
    statsmodels.stats.multitest.multipletests

    Notes
    -----
    In multiple hypothesis testing, false discovery control procedures tend to
    offer higher power than familywise error rate control procedures (e.g.
    Bonferroni correction [1]_).

    If the p-values correspond with independent tests (or tests with
    "positive regression dependencies" [2]_), rejecting null hypotheses
    corresponding with Benjamini-Hochberg-adjusted p-values below :math:`q`
    controls the false discovery rate at a level less than or equal to
    :math:`q m_0 / m`, where :math:`m_0` is the number of true null hypotheses
    and :math:`m` is the total number of null hypotheses tested. The same is
    true even for dependent tests when the p-values are adjusted accorded to
    the more conservative Benjaminini-Yekutieli procedure.

    The adjusted p-values produced by this function are comparable to those
    produced by the R function ``p.adjust`` and the statsmodels function
    `statsmodels.stats.multitest.multipletests`. Please consider the latter
    for more advanced methods of multiple comparison correction.

    References
    ----------
    .. [1] Benjamini, Yoav, and Yosef Hochberg. "Controlling the false
           discovery rate: a practical and powerful approach to multiple
           testing." Journal of the Royal statistical society: series B
           (Methodological) 57.1 (1995): 289-300.

    .. [2] Benjamini, Yoav, and Daniel Yekutieli. "The control of the false
           discovery rate in multiple testing under dependency." Annals of
           statistics (2001): 1165-1188.

    .. [3] TileStats. FDR - Benjamini-Hochberg explained - Youtube.
           https://www.youtube.com/watch?v=rZKa4tW2NKs.

    .. [4] Neuhaus, Karl-Ludwig, et al. "Improved thrombolysis in acute
           myocardial infarction with front-loaded administration of alteplase:
           results of the rt-PA-APSAC patency study (TAPS)." Journal of the
           American College of Cardiology 19.5 (1992): 885-891.

    Examples
    --------
    We follow the example from [1]_.

        Thrombolysis with recombinant tissue-type plasminogen activator (rt-PA)
        and anisoylated plasminogen streptokinase activator (APSAC) in
        myocardial infarction has been proved to reduce mortality. [4]_
        investigated the effects of a new front-loaded administration of rt-PA
        versus those obtained with a standard regimen of APSAC, in a randomized
        multicentre trial in 421 patients with acute myocardial infarction.

    There were four families of hypotheses tested in the study, the last of
    which was "cardiac and other events after the start of thrombolitic
    treatment". FDR control may be desired in this family of hypotheses
    because it would not be appropriate to conclude that the front-loaded
    treatment is better if it is merely equivalent to the previous treatment.

    The p-values corresponding with the 15 hypotheses in this family were

    >>> ps = [0.0001, 0.0004, 0.0019, 0.0095, 0.0201, 0.0278, 0.0298, 0.0344,
    ...       0.0459, 0.3240, 0.4262, 0.5719, 0.6528, 0.7590, 1.000]

    If the chosen significance level is 0.05, we may be tempted to reject the
    null hypotheses for the tests corresponding with the first nine p-values,
    as the first nine p-values fall below the chosen significance level.
    However, this would ignore the problem of "multiplicity": if we fail to
    correct for the fact that multiple comparisons are being performed, we
    are more likely to incorrectly reject true null hypotheses.

    One approach to the multiplicity problem is to control the family-wise
    error rate (FWER), that is, the rate at which the null hypothesis is
    rejected when it is actually true. A common procedure of this kind is the
    Bonferroni correction [1]_.  We begin by multiplying the p-values by the
    number of hypotheses tested.

    >>> import numpy as np
    >>> np.array(ps) * len(ps)
    array([1.5000e-03, 6.0000e-03, 2.8500e-02, 1.4250e-01, 3.0150e-01,
           4.1700e-01, 4.4700e-01, 5.1600e-01, 6.8850e-01, 4.8600e+00,
           6.3930e+00, 8.5785e+00, 9.7920e+00, 1.1385e+01, 1.5000e+01])

    To control the FWER at 5%, we reject only the hypotheses corresponding
    with adjusted p-values less than 0.05. In this case, only the hypotheses
    corresponding with the first three p-values can be rejected. According to
    [1]_, these three hypotheses concerned "allergic reaction" and "two
    different aspects of bleeding."

    An alternative approach is to control the false discovery rate: the
    expected fraction of rejected null hypotheses that are actually true. The
    advantage of this approach is that it typically affords greater power: an
    increased rate of rejecting the null hypothesis when it is indeed false. To
    control the false discovery rate at 5%, we apply the Benjamini-Hochberg
    p-value adjustment.

    >>> from scipy import stats
    >>> stats.false_discovery_control(ps)
    array([0.0015    , 0.003     , 0.0095    , 0.035625  , 0.0603    ,
           0.06385714, 0.06385714, 0.0645    , 0.0765    , 0.486     ,
           0.58118182, 0.714875  , 0.75323077, 0.81321429, 1.        ])

    Now, the first *four* adjusted p-values fall below 0.05, so we would reject
    the null hypotheses corresponding with these *four* p-values. Rejection
    of the fourth null hypothesis was particularly important to the original
    study as it led to the conclusion that the new treatment had a
    "substantially lower in-hospital mortality rate."

    """
    # Input Validation and Special Cases
    ps = np.asarray(ps)

    ps_in_range = (np.issubdtype(ps.dtype, np.number)
                   and np.all(ps == np.clip(ps, 0, 1)))
    if not ps_in_range:
        raise ValueError("`ps` must include only numbers between 0 and 1.")

    methods = {'bh', 'by'}
    if method.lower() not in methods:
        raise ValueError(f"Unrecognized `method` '{method}'."
                         f"Method must be one of {methods}.")
    method = method.lower()

    if axis is None:
        axis = 0
        ps = ps.ravel()

    axis = np.asarray(axis)[()]
    if not np.issubdtype(axis.dtype, np.integer) or axis.size != 1:
        raise ValueError("`axis` must be an integer or `None`")

    if ps.size <= 1 or ps.shape[axis] <= 1:
        return ps[()]

    ps = np.moveaxis(ps, axis, -1)
    m = ps.shape[-1]

    # Main Algorithm
    # Equivalent to the ideas of [1] and [2], except that this adjusts the
    # p-values as described in [3]. The results are similar to those produced
    # by R's p.adjust.

    # "Let [ps] be the ordered observed p-values..."
    order = np.argsort(ps, axis=-1)
    ps = np.take_along_axis(ps, order, axis=-1)  # this copies ps

    # Equation 1 of [1] rearranged to reject when p is less than specified q
    i = np.arange(1, m+1)
    ps *= m / i

    # Theorem 1.3 of [2]
    if method == 'by':
        ps *= np.sum(1 / i)

    # accounts for rejecting all null hypotheses i for i < k, where k is
    # defined in Eq. 1 of either [1] or [2]. See [3]. Starting with the index j
    # of the second to last element, we replace element j with element j+1 if
    # the latter is smaller.
    np.minimum.accumulate(ps[..., ::-1], out=ps[..., ::-1], axis=-1)

    # Restore original order of axes and data
    np.put_along_axis(ps, order, values=ps.copy(), axis=-1)
    ps = np.moveaxis(ps, -1, axis)

    return np.clip(ps, 0, 1)
