"""Reference (loop) implementation of the fixed-dispersion NBSR log-likelihood Hessian, kept only to test the
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
