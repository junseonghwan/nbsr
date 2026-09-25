import torch

def log_negbinomial(x, mu, phi):
    """Compute the log pdf of x,
    under a negative binomial distribution with mean mu and dispersion phi."""
    sigma2 = mu + phi * (mu ** 2)
    p = mu / sigma2 # equivalent to r / (mu + r).
    r = 1 / phi
    a = torch.lgamma(x + r) - torch.lgamma(x+1) - torch.lgamma(r)
    b = r * torch.log(p) + x * torch.log(1 - p)
    return a + b

def log_multinomial(x, pi):
    """Compute the log pdf of x,
    under a multinomial distribution with parameter pi."""
    n = torch.sum(x)
    log_pdf = torch.lgamma(n + 1) - torch.sum(torch.lgamma(x + 1)) + torch.sum(x * torch.log(pi))
    return log_pdf

def log_gamma(x, shape, scale):
    log_pdf = (shape - 1) * torch.log(x) - x / scale - torch.lgamma(shape) - shape * torch.log(scale)
    return(log_pdf)

def log_invgamma(x, alpha, beta):
    log_term1 = alpha * torch.log(beta)
    log_term2 = - torch.lgamma(alpha)
    log_term3 = - (alpha + 1) * torch.log(x)
    log_term4 = - beta / x

    log_prob = log_term1 + log_term2 + log_term3 + log_term4
    return log_prob

def log_normal(x, mu, std):
    return -torch.log(torch.sqrt(2*torch.pi*(std**2))) - (0.5 * ((x-mu)/std)**2)

def log_lognormal(x, mu, std):
    return -torch.log(x * torch.sqrt(2*torch.pi*(std**2))) - (0.5*((torch.log(x) - mu)/std)**2)

def softplus_inv(y):
    return y + y.neg().expm1().neg().log()

def softplus(y):
    return torch.nn.Softplus()(y)


def nb_log_density_derivatives(y, mu, phi, second=True):
    """Derivatives of log NB(y; mu, phi) with respect to u = log mu and v = log phi, elementwise.

    Returns (l_u, l_v, l_uu, l_uv, l_vv); the second-order terms are None when second=False.
    Written with q = 1 / (1 + phi mu) and r = 1 / phi so that it stays finite for very small phi.
    """
    r = 1.0 / phi
    q = 1.0 / (1.0 + phi * mu)
    l_u = (y - mu) * q
    dl_dr = (torch.special.digamma(y + r) - torch.special.digamma(r) + 1.0
             - torch.log1p(phi * mu) - (y + r) * phi * q)
    l_v = -r * dl_dr
    if not second:
        return l_u, l_v, None, None, None
    l_uu = -mu * (1.0 + phi * y) * q ** 2
    l_uv = -(y - mu) * mu * phi * q ** 2
    d2l_dr2 = (torch.special.polygamma(1, y + r) - torch.special.polygamma(1, r)
               + phi - 2.0 * phi * q + (y + r) * (phi * q) ** 2)
    l_vv = r * dl_dr + r ** 2 * d2l_dr2
    return l_u, l_v, l_uu, l_uv, l_vv
