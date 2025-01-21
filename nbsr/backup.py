def log_likelihood_samples(pi, model, disp_model, mc_samples, observation_wise=False):
	# We approximate the Q function by sampling the dispersion.
	K = pi.shape[1]
	epsilon = torch.randn(mc_samples, K) # MxK

	if observation_wise:
		phi = torch.exp(disp_model.forward(pi).unsqueeze(0) + epsilon * disp_model.get_sd())
	else:
		pi_bar = torch.mean(pi, 0, keepdim=True) # 1xK
		phi = torch.exp(disp_model.forward(pi_bar) + epsilon * disp_model.get_sd())

	# Compute the log likelihood of Y.
	log_obs_lik = torch.stack([model.log_likelihood2(phi_m) for phi_m in phi])
	log_dispersion_lik = torch.stack([disp_model.log_density(phi_m, pi) for phi_m in phi])
	return (log_obs_lik + log_dispersion_lik, phi)

@click.command()
@click.argument('data_path', type=click.Path(exists=True))
@click.argument('vars', nargs=-1)
@click.option('-f', '--mu_hat_file', default="deseq2_mu.csv", type=str, help="File name containing initial estimates of mu_hat.")
@click.option('-e', '--max_em_iter', default=100, type=int, help="Number of EM algorithm iterations.")
@click.option('-n', '--mc_samples', default=20, type=int, help="Number of Monte Carlo samples to approximate the E-step.")
@click.option('-m', '--m_max_iter', default=500, type=int, help="Number of iterations for the M-step.")
@click.option('--lr', default=0.05, type=float, help="Learning rate for the M-step.")
@click.option('--knot_count', default=10, type=int, help="GRBF model number of knots.")
@click.option('--grbf_sd', default=0.5, type=float, help="GRBF model standard deviation parameter.")
@click.option('--observation_wise', is_flag=True, show_default=False, default=False, type=bool, help="Fit observation wise dispersion model.")
def mcem(data_path, vars, mu_hat_file, max_em_iter, mc_samples, m_max_iter, lr, knot_count, grbf_sd, observation_wise):
	# Read in the mean expression.
	# Optimize GRBF prior via EM algorithm.
	# Marginalize out phi by sampling.
	# Obtain the mean dispersion and use it for fitting NBSR and output it to file.
	print("Performing Empirical Bayes estimation of dispersion.")
	column_names = list(vars)
	mu_hat = pd.read_csv(os.path.join(data_path, mu_hat_file))
	mu_hat = torch.tensor(mu_hat.transpose().to_numpy())
	pi_hat = mu_hat / mu_hat.sum(dim=1, keepdim=True)

	if observation_wise == False: # feature wise dispersion
		pi_bar = torch.mean(pi_hat, 0, keepdim=True) # Take the average over samples.
	else:
		pi_bar = pi_hat

	log_pi_bar = torch.log(pi_bar)
	min_val = torch.min(log_pi_bar.data)
	max_val = torch.max(log_pi_bar.data)

	counts_pd = pd.read_csv(os.path.join(data_path, "Y.csv"))
	if os.path.exists(os.path.join(data_path, "X.csv")):
		coldata_pd = pd.read_csv(os.path.join(data_path, "X.csv"), na_filter=False, skipinitialspace=True)
	else:
		coldata_pd = None
	
	Y = torch.tensor(counts_pd.transpose().to_numpy(), dtype=torch.float64)
	X,_ = construct_tensor_from_coldata(coldata_pd, column_names, counts_pd.shape[1], include_intercept=False)

	grbf_disp_model = DispersionGRBF(min_val, max_val, sd=grbf_sd, knot_count=knot_count)
	model = NBSREmpiricalBayes(Y, mu_hat)
	#model.specify_beta_prior(1., 3., 2.)

	optimizer = torch.optim.Adam(grbf_disp_model.parameters(),lr=lr)
	print("Optimizing GRBF parameters by maximizing the posterior conditional on mu hat.")

	# Use MC-EM to estimate GRBF prior.
	for em_iter in range(max_em_iter):
		print(f"EM iteration {em_iter}.")

		curr_beta = grbf_disp_model.beta.detach().clone()
		# Compute the weights.
		# 1. Draw phi_ij^m | b^{(t)} samples for m = 1, ..., mc_samples, t = 1, ..., max_em_iter.
		# 2. Compute the log likelihood log P(Y_ij | mu_hat_ij, phi_ij^m).
		# 3. Prior and proposal cancels.
		# Update log_likelihood_samples to generate phi (K dimensional rather than NxK dimensional).
		log_weights, phi_samples = log_likelihood_samples(pi_hat, model, grbf_disp_model, mc_samples, observation_wise)
		# Sample \phi is sampled and weights are computed given b^{(t)} -> these are not to be optimized.
		phi_samples = phi_samples.detach()
		log_weights = log_weights.detach()
		log_weights_sum = torch.logsumexp(log_weights, dim=0, keepdim=True)  # Shape: 1 x N x K
		normalized_log_weights = log_weights - log_weights_sum  # Shape: M x N x K
		normalized_weights = torch.exp(normalized_log_weights).data
		#err = torch.sum(normalized_weights.sum(0) - 1., dtype=torch.float32)
		#self.assertTrue(torch.allclose(err, torch.tensor([0.]), atol=1e-6))

		for iter in range(m_max_iter):
			curr_beta_iter = grbf_disp_model.beta.detach().clone()
			# Construct the Q-function.
			log_obs_lik = torch.stack([model.log_likelihood2(phi_m) for phi_m in phi_samples])
			log_phi_lik = torch.stack([grbf_disp_model.log_density(phi_m, pi_hat) for phi_m in phi_samples])
			# Compute the log of prior for the GRBF parameters. Normal prior.
			log_prior = log_normal(grbf_disp_model.beta, torch.zeros_like(grbf_disp_model.beta), torch.tensor(1.)).sum()
			objective_q = (normalized_weights * (log_obs_lik + log_phi_lik)).sum() + log_prior

			# Optimize the posterior
			loss = -objective_q
			optimizer.zero_grad()
			loss.backward(retain_graph=True)
			optimizer.step()

			if iter % 10 == 0:
				print(f"Iter: {iter}. Loss: {loss.data}")

			mean_sq_change = torch.mean(torch.abs(curr_beta_iter - grbf_disp_model.beta.detach()))
			if mean_sq_change < 1e-6:
				print(f"Parameter change: {mean_sq_change}")
				print("M step converged.")
				break

		print(f"Log obs likelihood: {log_obs_lik.detach().data.sum()}")
		print(f"Log phi likelihood: {log_phi_lik.detach().data.sum()}")
		# Check for convergence: if the parameters have not moved much.
		mean_sq_change = torch.mean((curr_beta - grbf_disp_model.beta.detach())**2)
		print(f"Parameter change: {mean_sq_change}")
		if mean_sq_change < 1e-6:
			print("Converged.")
			break

	log_phi_mean = grbf_disp_model.evaluate_mean(pi_bar, grbf_disp_model.beta, grbf_disp_model.centers, grbf_disp_model.h)[0]
	trended_dispersion = torch.exp(log_phi_mean)
	np.savetxt(os.path.join(data_path, "dispersion_grbf.csv"), trended_dispersion.data.numpy().transpose(), delimiter=',')
	
	torch.save(grbf_disp_model, os.path.join(data_path, 'grbf_model.pth'))
	torch.save({
		'model_state': grbf_disp_model.state_dict()
        }, os.path.join(data_path, 'grbf_model.pth'))

