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

# Let's optimize for beta.
def log_normal(x, mu, std):
    """Compute the log pdf of x,
    under a lognormal distribution with mean mu and standard deviation std."""
    return -0.5 * torch.log(2*torch.pi*(std**2)) - (0.5 * (1/(std**2)) * (x-mu)**2)

def softplus_inv(y):
    return y + y.neg().expm1().neg().log()