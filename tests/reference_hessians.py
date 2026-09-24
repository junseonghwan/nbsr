"""Reference (loop) implementations of the NBSR log-likelihood Hessians, kept only to test the
vectorized nbsr.utils.kron_hessian path. O(N J dim^2 P^2) Python loops: use tiny problems."""
import numpy as np


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

