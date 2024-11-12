import torch

from nbsr.distributions import log_negbinomial, log_normal, log_half_normal, log_lognormal, log_invgamma, softplus, softplus_inv

# LogNormal model for the dispersion parameters of the NBSRDispersion.
class DispersionModel(torch.nn.Module):
    def __init__(self, Y, Z = None, feature_specific_intercept = False, b0j=None, optimize_kappa=True):
        super().__init__()
        self.softplus = torch.nn.Softplus()
        self.Z = Z # NxP design matrix for covariates to use in modeling the dispersion.

        self.feature_specific_intercept = feature_specific_intercept

        self.R = Y.sum(1)
        self.log_R = torch.log(self.R)

        # Formulate design matrix: dimension is N x K x (P+3).
        # For each sample i=1,...,N, the KxP matrix contains 
        # a column of 1's (intercept_j), 
        # a column over j log \pi_{i,j}, 
        # a column of log R_i
        # and remaining P variables as containined in Z.
        # In total we need P+3 parameters.
        self.sample_count = Y.shape[0]
        self.feature_count = Y.shape[1]
        if b0j is not None:
            self.b0 = torch.nn.Parameter(b0j, requires_grad=True)
            self.feature_specific_intercept = True
        elif feature_specific_intercept:
            self.b0 = torch.nn.Parameter(torch.randn(self.feature_count, dtype=torch.float64), requires_grad=True)
        else:
            self.b0 = torch.nn.Parameter(torch.randn(1, dtype=torch.float64), requires_grad=True)
        self.b1 = torch.nn.Parameter(-torch.abs(torch.randn(1, dtype=torch.float64)), requires_grad=True)
        #self.b2 = torch.nn.Parameter(-torch.abs(torch.randn(1, dtype=torch.float64)), requires_grad=True)
        if self.Z is None:
            self.beta = None
            self.covariate_count = 0
        else:
            self.covariate_count = Z.shape[1]
            self.beta = torch.nn.Parameter(torch.randn(self.covariate_count, dtype=torch.float64), requires_grad=True)

        if optimize_kappa:
            self.kappa = torch.nn.Parameter(torch.randn(1, dtype=torch.float64), requires_grad=True)
        else:
            kappa = torch.tensor(0.01, dtype=torch.float64)
            self.kappa = torch.nn.Parameter(kappa, requires_grad=False)

    def get_sd(self):
        return self.softplus(self.kappa)

    # Compute the mean of the distribution.
    def forward(self, pi):
        # log_pi has shape (self.sample_count, self.feature_count)
        #assert(log_pi.shape[0] == self.sample_count)
        #assert(log_pi.shape[1] == self.feature_count)
        # log \phi_{ij} = b_{0,j} + b_1 log \pi_{i,j} + b_2 log R_i + \beta' z_i + \epsilon_j.
        if self.feature_specific_intercept:
            val0 = self.b0.unsqueeze(-1).transpose(0,1).expand(self.sample_count, self.feature_count)
        else:
            val0 = self.b0
        val1 = self.b1 * torch.log(pi) 
        #val2 = self.b2 * self.log_R.unsqueeze(-1).expand(-1, self.feature_count)
        if self.Z is None:
            val3 = 0
        else:
            val3 = torch.mm(self.Z, self.beta.unsqueeze(-1)).expand(-1, self.feature_count)
        #log_phi_mean = val0 + val1 + val2 + val3
        log_phi_mean = val0 + val1 + val3
        return(log_phi_mean)

    def log_prior(self):
        log_prior0 = log_normal(self.b0, torch.zeros_like(self.b0), torch.tensor(1.)).sum()
        log_prior1 = log_normal(self.b1, torch.tensor([0.]), torch.tensor(0.1))
        #log_prior2 = log_normal(self.b2, torch.tensor([0.]), torch.tensor(0.1))
        log_prior_kappa = log_invgamma(self.get_sd(), torch.tensor(3.), torch.tensor(2.)).sum()
        #log_prior = log_prior0 + log_prior1 + log_prior2 + log_prior_kappa
        log_prior = log_prior0 + log_prior1 + log_prior_kappa
        return log_prior

    # LogNormal evaluated at phi | pi and current model parameter.
    def log_density(self, phi, pi):
        log_pi = torch.log(pi)

        log_phi_mean = self.forward(log_pi)
        log_lik_vals = log_lognormal(phi, log_phi_mean, self.get_sd().unsqueeze(0))
        return(log_lik_vals)

class DispersionGRBF(torch.nn.Module):
    def __init__(self, min_value, max_value, Z=None, sd=None, knot_count = 10, width=1.2):
        super().__init__()
        self.Z = Z # NxP design matrix of covariates.
        # the number of parameters is knot_count + 2 + Z.shape[1] (+2 for intercept and log pi and other covariates in Z).
        self.dim = knot_count + 2
        if Z is not None:
            self.dim += Z.shape[1]

        #self.beta = torch.nn.Parameter(torch.randn(self.dim, dtype=torch.float64), requires_grad=True)
        self.beta = torch.nn.Parameter(torch.zeros(self.dim, dtype=torch.float64), requires_grad=True)
        # kappa parameter is in the log scale.
        if sd is None:
            self.kappa = torch.nn.Parameter(softplus_inv(torch.tensor(0.5)), requires_grad=True)
        else:
            self.kappa = softplus_inv(torch.tensor(sd, requires_grad=False))
        self.width = width
        self.knot_count = knot_count
        if knot_count > 0:
            self.centers = torch.linspace(min_value, max_value, knot_count)
            delta = torch.diff(self.centers)
            self.h = torch.tensor([delta[0] * self.width]).repeat(knot_count)
        else:
            self.centers = None
            self.h = None

    def forward(self, pi):
        log_phi_mean, _ = self.evaluate_mean(pi, self.beta, self.centers, self.h)
        return(log_phi_mean)

    def get_sd(self):
        return torch.exp(self.kappa)

    # phi ~ LogNormal(f(log pi), sd).
    def log_density(self, phi, pi):
        assert(phi.shape == pi.shape)
        # phi: log of dispersion values of dimension NxK -- N=1 if using feature wise dispersion.
        # pi: pi values of dimension NxK -- N=1 if using feature wise dispersion.
        log_phi_mean = self.forward(pi)
        sd = self.get_sd()
        log_lik = log_lognormal(phi, log_phi_mean, sd)
        return(log_lik)

    def log_prior(self):
        log_prior_beta = log_normal(self.beta, torch.zeros_like(self.beta), torch.tensor(1.)).sum()
        log_prior_kappa = log_invgamma(self.get_sd(), torch.tensor(10.), torch.tensor(3.)).sum()
        return (log_prior_beta + log_prior_kappa)

    # pi: not in log scale (will take the log in the function)
    # centers: in log scale.
    # Returns log phi mean (trend).
    def evaluate_mean(self, pi, beta, log_centers=None, scales=None):
        # pi is a tensor of dimension NxK.
        log_pi = torch.log(pi)
        N, K = log_pi.shape
        intercept = torch.ones(N, K, 1)
        log_pi_expanded = log_pi.unsqueeze(2)
        if log_centers is not None:
            L = log_centers.shape[0]
            # g is N x K x L
            g = self.torch_gaussian_rbf(log_pi_expanded, log_centers, scales)
            # concatenate intercept and log_mu
            X = torch.concat([intercept, log_pi_expanded, g], dim = 2)
        else:
            X = torch.concat([intercept, log_pi], dim = 2)
        if self.Z is not None:
            Z = self.Z.unsqueeze(1).expand(-1, K, -1)
            X = torch.cat((Z,X), dim=2)
        f = torch.matmul(X, beta)
        return(f, X)

    def torch_gaussian_rbf(self, x, center, scale):
        if not torch.is_tensor(x):
            x = torch.tensor(x, dtype=torch.float64)

        # center is a vector of length L. Reshape for broadcasting
        center = center.reshape(1, 1, center.shape[0])  # 1 x 1 x L
        # scale is either a vector of length L or a scalar.
        scale = scale.reshape(1, 1, scale.shape[0])  # 1 x 1 x L or 1 x 1 x 1
        
        # Broadcasting will align dimensions automatically.
        val = (x - center) / scale
        val = torch.exp(-0.5 * val**2)
        return(val)
