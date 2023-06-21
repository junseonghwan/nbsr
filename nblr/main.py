import click
import pandas as pd
import numpy as np
import torch
from nblr.negbinomial_model import NegativeBinomialRegressionModel
from nblr.utils import *
import os

torch.set_default_dtype(torch.float64)
torch.set_printoptions(precision=9)

@click.group()
def cli():
	pass

def moving_average(arr, window):
	return np.convolve(arr, np.ones(window), 'valid') / window

def RunInference(counts, coldata, output, column_names, max_iter, learning_rate, s0, shape, scale, tol, window_size):
	click.echo(counts)
	click.echo(coldata)
	counts_pd = pd.read_csv(counts)
	coldata_pd = pd.read_csv(coldata, na_filter=False)
	Y = counts_pd.transpose().to_numpy()
	print("Y: ", Y.shape)
	X = coldata_pd[column_names]
	print("X: ", X.shape)

	print(torch.get_default_dtype())
	model = NegativeBinomialRegressionModel(X, Y, s0=s0, shape=shape, scale=scale)
	print(model.X.dtype, model.Y.dtype)
	# Fit the model.
	optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)
	loss_history = []
	for i in range(max_iter):
		#loss = -model.log_posterior(model.mu, model.beta)
		loss = -model.log_likelihood(model.mu, model.beta)
		optimizer.zero_grad()
		loss.backward(retain_graph=True)
		optimizer.step()
		loss_history.append(loss.data.numpy()[0])
		if i % 100 == 0:
			print("Iter:", i)
			print(loss.data)
		if len(loss_history) > window_size:
			moving_avg = moving_average(np.array(loss_history), window_size)
			diffs = np.abs(np.diff(moving_avg))
			#print(diffs[-5:])
			if np.all(diffs[-5:] < tol):
				print(f"Convergence is reached at iter {i} with loss {loss_history[-1]}")
				break

	# Optimization using L-BFGS:
	# optimizer = torch.optim.LBFGS(model.parameters(),lr=learning_rate)
	# def closure():
	# 	optimizer.zero_grad()
	# 	loss = -model.log_posterior(model.mu, model.beta)
	# 	loss.backward()
	# 	return loss
	
	# curr_loss = torch.inf
	# for i in range(iters):
	# 	optimizer.step(closure)
	# 	loss = -model.log_posterior(model.mu, model.beta)
	# 	if i % 10 == 0:
	# 		print("Iter:", i)
	# 		print(loss.data)
	# 	if torch.abs(curr_loss - loss) < tol:
	# 		break
	return model

@click.command()
@click.argument('counts', type=click.Path(exists=True))
@click.argument('coldata', type=click.Path(exists=True))
@click.argument('output', type=click.Path(exists=True))
@click.argument('var', nargs=-1)
@click.option('-i', '--max_iter', default=1000, type=int)
@click.option('-l', '--learning_rate', default=0.1, type=float)
@click.option('--w0', default="null", type=str)
@click.option('--w1', default="alt", type=str)
@click.option('--s0', default=2, type=float)
@click.option('--shape', default=2, type=float)
@click.option('--scale', default=1, type=float)
@click.option('--tol', default=0.01, type=float)
@click.option('--window', default=10, type=int)
def Run(counts, coldata, output, var, w0, w1, max_iter, learning_rate, s0, shape, scale, tol, window):
	cols = list(var)
	assert(len(var) == 1)
	model = RunInference(counts, coldata, output, cols, max_iter, learning_rate, s0, shape, scale, tol, window)
	logRRi, sd_est = logRR(model, cols[0], w0, w1)
	beta = get_beta(model).data.numpy()
	torch.save(model, os.path.join(output, "nblr_model"))
	np.savetxt(os.path.join(output, "nblr_mu.csv"), model.mu.data.numpy(), delimiter=',')
	np.savetxt(os.path.join(output, "nblr_beta.csv"), beta, delimiter=',')
	np.savetxt(os.path.join(output, "nblr_phi.csv"), model.phi.data.numpy(), delimiter=',')
	np.savetxt(os.path.join(output, "nblr_logRR.csv"), logRRi.transpose(), delimiter=',')
	np.savetxt(os.path.join(output, "nblr_logRR_sd.csv"), sd_est.transpose(), delimiter=',')

cli.add_command(Run)
# cli.add_command(RunInference)

if __name__ == '__main__':
    cli()
