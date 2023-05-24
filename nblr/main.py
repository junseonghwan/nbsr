import click
import pandas as pd
import numpy as np
import torch
from negbinomial_model import NegativeBinomialRegressionModel
from utils import *
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
	coldata_pd = pd.read_csv(coldata)
	Y = counts_pd.transpose().to_numpy()
	print("Y: ", Y.shape)
	X = []
	for col_name in column_names:
		X.append(pd.get_dummies(coldata_pd[col_name]).to_numpy())
	X = np.column_stack(X)
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

	s = np.sum(Y, 1)
	Y_fit, pi_fit = summarize(model, X, s, False, True)

	# Compute logRR_i and Var(logRR_i) for each sample i.
	# Estimate logRR and Var(logRR).
	# Construct p-value for testing.

	return (Y_fit, pi_fit, model.mu.data.numpy(), model.beta.data.numpy().reshape(model.covariate_count, model.dim), model.phi.data.numpy())

@click.command()
@click.argument('path', type=click.Path(exists=True))
@click.argument('experiment_index', type=str)
@click.argument('column_names', nargs=-1)
@click.option('-i', '--max_iter', default=1000, type=int)
@click.option('-l', '--learning_rate', default=0.01, type=float)
@click.option('--s0', default=2, type=float)
@click.option('--shape', default=2, type=float)
@click.option('--scale', default=1, type=float)
@click.option('--tol', default=0.1, type=float)
@click.option('--window', default=10, type=int)
def BatchRun(path, experiment_index, column_names, max_iter, learning_rate, s0, shape, scale, tol, window):
	begin, end = experiment_index.split(",")
	for i in range(int(begin), int(end)+1):
		output_path = os.path.join(path, "p" + str(i))
		coldata_path = os.path.join(output_path, "X.csv")
		counts_path = os.path.join(output_path, "Y.csv")
		(_, pi, mu, beta, phi) = RunInference(counts_path, coldata_path, output_path, column_names, max_iter, learning_rate, s0, shape, scale, tol, window)
		np.savetxt(os.path.join(output_path, "mu.csv"), mu, delimiter=',')
		np.savetxt(os.path.join(output_path, "beta.csv"), beta, delimiter=',')
		np.savetxt(os.path.join(output_path, "phi.csv"), phi, delimiter=',')
		np.savetxt(os.path.join(output_path, "pi.csv"), pi, delimiter=',')

@click.command()
@click.argument('counts', type=click.Path(exists=True))
@click.argument('coldata', type=click.Path(exists=True))
@click.argument('output', type=click.Path(exists=True))
@click.argument('column_names', nargs=-1)
@click.option('-i', '--max_iter', default=1000, type=int)
@click.option('-l', '--learning_rate', default=0.1, type=float)
@click.option('--s0', default=2, type=float)
@click.option('--shape', default=2, type=float)
@click.option('--scale', default=1, type=float)
@click.option('--tol', default=0.1, type=float)
@click.option('--window', default=10, type=int)
def Run(counts, coldata, output, column_names, max_iter, learning_rate, s0, shape, scale, tol, window):
	(_, pi, mu, beta, phi) = RunInference(counts, coldata, output, column_names, max_iter, learning_rate, s0, shape, scale, tol, window)
	np.savetxt(os.path.join(output, "mu.csv"), mu, delimiter=',')
	np.savetxt(os.path.join(output, "beta.csv"), beta, delimiter=',')
	np.savetxt(os.path.join(output, "phi.csv"), phi, delimiter=',')
	np.savetxt(os.path.join(output, "pi.csv"), pi, delimiter=',')

cli.add_command(BatchRun)
cli.add_command(Run)
# cli.add_command(RunInference)

if __name__ == '__main__':
    cli()
