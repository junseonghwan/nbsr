import click
import pandas as pd
import numpy as np
import torch
from nblr.negbinomial_model import NegativeBinomialRegressionModel
from nblr.utils import *
import os
import copy

torch.set_default_dtype(torch.float64)
torch.set_printoptions(precision=9)

checkpoint_filename = "checkpoint.pth"

@click.group()
def cli():
	pass

def assess_convergence(loss_history, tol, lookback_iterations):
	diffs = np.abs(np.diff(loss_history))
	if np.all(diffs[-lookback_iterations:] < tol):
		print(f"Convergence reached")
		return True
	else:
		print(f"Not converged")
		return False	

def fit(model, optimizer, iterations, tol, lookback_iterations):
	# Fit the model.
	loss_history = []
	# We will store the best solution.
	best_model_state = None
	best_loss = torch.inf
	for i in range(iterations):
		#loss = -model.log_posterior(model.mu, model.beta)
		loss = -model.log_likelihood(model.mu, model.beta)
		optimizer.zero_grad()
		loss.backward(retain_graph=True)
		optimizer.step()
		loss_history.append(loss.data.numpy()[0])
		if loss.data < best_loss:
			best_model_state = copy.deepcopy(model.state_dict())
			best_loss = loss.data

		if i % 100 == 0:
			print("Iter:", i)
			print(loss.data)

	return (loss_history, best_model_state, best_loss)

def ConstructModel(counts, coldata, column_names, s0, shape, scale):
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
	return model

@click.command()
@click.argument('model_path', type=click.Path(exists=True))
@click.argument('output', type=click.Path(exists=True))
@click.argument('var', type=str)
@click.argument('w0', type=str)
@click.argument('w1', type=str)
def Results(model_path, output, var, w0, w1):
	checkpoint = torch.load(os.path.join(model_path, checkpoint_filename))
	counts_path = checkpoint['counts_path']
	coldata_path = checkpoint['coldata_path']
	cols = checkpoint['cols']
	s0 = checkpoint['s0']
	shape = checkpoint['shape']
	scale = checkpoint['scale']
	model = ConstructModel(counts_path, coldata_path, cols, s0, shape, scale)
	model.load_state_dict(checkpoint['best_model_state_dict'])

	logRRi, sd_est = logRR(model, var, w0, w1)
	np.savetxt(os.path.join(output, "nblr_logRR.csv"), logRRi.transpose(), delimiter=',')
	np.savetxt(os.path.join(output, "nblr_logRR_sd.csv"), sd_est.transpose(), delimiter=',')

@click.command()
@click.argument('counts_path', type=click.Path(exists=True))
@click.argument('coldata_path', type=click.Path(exists=True))
@click.argument('output_path', type=click.Path(exists=True))
@click.argument('vars', nargs=-1)
@click.option('-i', '--iterations', default=1000, type=int)
@click.option('-l', '--learning_rate', default=0.1, type=float)
@click.option('--s0', default=2, type=float)
@click.option('--shape', default=2, type=float)
@click.option('--scale', default=1, type=float)
@click.option('--tol', default=0.001, type=float)
@click.option('--lookback_iterations', default=100, type=int)
def Train(counts_path, coldata_path, output_path, vars, iterations, learning_rate, s0, shape, scale, tol, lookback_iterations):
	assert(len(vars) > 0)
	cols = list(vars)
	model = ConstructModel(counts_path, coldata_path, cols, s0, shape, scale)
	optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)
	loss_history, best_model_state, best_loss = fit(model, optimizer, iterations, tol, lookback_iterations)
	converged = assess_convergence(loss_history, tol, lookback_iterations)
	torch.save({
        'model_state_dict': model.state_dict(),
        'best_model_state_dict': best_model_state,
        'optimizer_state_dict': optimizer.state_dict(),
        's0': s0,
        'shape': shape,
        'scale': scale,
        'counts_path': counts_path,
        'coldata_path': coldata_path,
        'output_path': output_path,
        'loss': loss_history,
        'best_loss': best_loss,
        'cols': cols,
        'converged': converged
        }, os.path.join(output_path, 'checkpoint.pth'))

@click.command()
@click.argument('checkpoint_path', type=click.Path(exists=True))
@click.option('-i', '--iterations', default=1000, type=int)
@click.option('-r', '--repeats', default=10, type=int)
@click.option('--tol', default=0.001, type=float)
@click.option('--lookback_iterations', default=100, type=int)
def Resume(checkpoint_path, iterations, repeats, tol, lookback_iterations):
	checkpoint = torch.load(os.path.join(checkpoint_path, checkpoint_filename))
	counts_path = checkpoint['counts_path']
	coldata_path = checkpoint['coldata_path']
	output_path = checkpoint['output_path']
	cols = checkpoint['cols']
	s0 = checkpoint['s0']
	shape = checkpoint['shape']
	scale = checkpoint['scale']
	model = ConstructModel(counts_path, coldata_path, cols, s0, shape, scale)
	model.load_state_dict(checkpoint['model_state_dict'])
	optimizer = torch.optim.Adam(model.parameters())
	optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
	loss_history = checkpoint['loss']
	curr_best_loss = checkpoint['best_loss']
	curr_best_state = checkpoint['best_model_state_dict']
	for i in range(repeats):
		loss, best_model_state, best_loss = fit(model, optimizer, iterations, tol, lookback_iterations)
		loss_history.extend(loss)
		converged = assess_convergence(loss_history, tol, lookback_iterations)
		if best_loss < curr_best_loss:
			curr_best_loss = curr_best_loss
			curr_best_state = copy.deepcopy(best_model_state)

		if converged:
			break

	torch.save({
        'model_state_dict': model.state_dict(),
        'best_model_state_dict': best_model_state,
        'optimizer_state_dict': optimizer.state_dict(),
        's0': s0,
        'shape': shape,
        'scale': scale,
        'counts_path': counts_path,
        'coldata_path': coldata_path,
        'output_path': output_path,
        'loss': loss_history,
        'best_loss': best_loss,
        'cols': cols,
        'converged': converged
        }, os.path.join(output_path, 'checkpoint.pth'))

cli.add_command(Train)
cli.add_command(Resume)
cli.add_command(Results)

if __name__ == '__main__':
    cli()
