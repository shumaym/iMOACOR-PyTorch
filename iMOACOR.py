#!/usr/bin/env python3
import os
os.chdir(os.path.dirname(os.path.realpath(__file__)))

import time
from datetime import datetime
import itertools
import signal
import numpy as np
import torch
import sys
import config

version = '1.1'

# Gather all configuration parameters from config.py, according to the command-line arguments.
(obj_function_name, n, k, M, Rmin, Rmax, q, xi,
	Gmax, N, H, EPSILON, MAX_RECORD_SIZE, var_threshold,
	tol_threshold, num_runs, scalarizer, flag_create_snapshots) = config.main(version, sys.argv[1:])

if torch.__version__[0] < '1':
	print('\nUnsupported PyTorch version. Please upgrade to PyTorch v1.0 or greater.')
	quit()

flag_jit_enabled = True if (torch.__version__[0] >= '1') else False
torch.set_num_threads(4)
dtype = torch.float32
dtype_int = torch.int32

# Used during calculation of standard deviations
std_dev_coefficient = torch.as_tensor(xi / (N-1), dtype=dtype)
# Used in the VADS scalarizer function.
p = torch.as_tensor(2.)

###
# Used for reference points required in Hypervolume calculation.
# Relevant values are printed to a .ref file in the output subfolder.
###
reference_points_dict = {
	'DTLZ1': ' '.join([str(1) for _ in range(k)]),
	'DTLZ2': ' '.join([str(2) for _ in range(k)]),
	'DTLZ3': ' '.join([str(7) for _ in range(k)]),
	'DTLZ4': ' '.join([str(2) for _ in range(k)]),
	'DTLZ5': ' '.join([str(4) for _ in range(k)]),
	'DTLZ6': ' '.join([str(11) for _ in range(k)]),
	'DTLZ7': ' '.join([str(1) for _ in range(k-1)] + [str(21)])
}

if obj_function_name[:4] == 'DTLZ':
	reference_point = reference_points_dict[obj_function_name]
	obj_function = getattr(__import__('objective_functions', fromlist=[obj_function_name]), obj_function_name)
else:
	print('\n\nFatal error: obj_function_name not valid!\n')
	quit()


### Creation of output folder. ###
try:
	os.makedirs('./output')
except PermissionError:
	print('\n\nFatal error: You do not have the required permissions to create the output folder. Exiting.')
	quit()
except FileExistsError:
	pass

### Creation of output subfolder. ###
executionStart = ''.join([str(datetime.now().date())[5:], '_', str(datetime.now().time())[:8]]).replace(':', '-')
output_subfolder = ''.join(['./output/', obj_function.__name__, '_m', str(k), '_', executionStart])
try:
	os.makedirs(output_subfolder)
except PermissionError:
	print('\n\nFatal error: You do not have the required permissions to create the output subfolder. Exiting.')
	quit()
except FileExistsError:
	pass

if flag_create_snapshots:
	### Creation of snapshot folder. ###
	try:
		os.makedirs('./snapshots')
	except PermissionError:
		print('\n\nFatal error: You do not have the required permissions to create the snapshots folder. Exiting.')
		quit()
	except FileExistsError:
		pass

	### Creation of snapshot subfolder. ###
	snapshot_subfolder = ''.join(['./snapshots/', obj_function.__name__, '_m', str(k), '_', executionStart])
	try:
		os.makedirs(snapshot_subfolder)
		with open(''.join([snapshot_subfolder, '/reference_point.ref']), 'w') as f:
			f.write(reference_point)
	except PermissionError:
		print('\n\nFatal error: You do not have the required permissions to create the snapshots subfolder. Exiting.')
		quit()
	except FileExistsError:
		pass

### Creation of file containing the appropriate reference point for hypervolume calculations. ###
try:
	with open(''.join([output_subfolder, '/reference_point.ref']), 'w') as f:
		f.write(reference_point)
except PermissionError:
	print('\n\nFatal error: You do not have the required permissions to create the file \'{0}\'\n'
		+ 'Exiting.'.format(''.join([output_subfolder, '/reference_point.ref'])))
	quit()
except FileExistsError:
	pass

def exit_gracefully(signum, frame):
	"""
	More elegant handling of keyboard interrupts.
	Enter 'Ctrl+C', then 'y' to terminate early.
	"""
	signal.signal(signal.SIGINT, original_sigint)
	try:
		input_text = input('\n\nReally Quit? (y/n)> ').lower().lstrip()
		if input_text != '' and input_text[0] == 'y':
			print('Exiting.')
			quit()
	except KeyboardInterrupt:
		print('\nExiting.')
		quit()
	signal.signal(signal.SIGINT, exit_gracefully)


original_sigint = signal.getsignal(signal.SIGINT)
signal.signal(signal.SIGINT, exit_gracefully)


def save_pareto_set(execNum):
	""" Writes t_archive's x values to .pos file """
	with open(''.join([output_subfolder, '/Run', str(execNum) , '.pos']), 'w') as f:
		f.write('# k={0}\n'.format(k))
		f.write('# n={0}\n'.format(n))
		f.write('# N={0}\n'.format(N))
		f.write('# q={0}\n'.format(q))
		f.write('# xi={0}\n'.format(xi))
		for row in t_archive.numpy():
			for idx,col in enumerate(row[:n]):
				f.write(str(col))
				if idx != n - 1:
					f.write('\t')
			f.write('\n')


def save_pareto_front(execNum):
	""" Writes t_archive's Fx values to .pof file """
	with open(''.join([output_subfolder, '/Run', str(execNum) , '.pof']), 'w') as f:
		f.write('# k={0}\n'.format(k))
		f.write('# n={0}\n'.format(n))
		f.write('# N={0}\n'.format(N))
		f.write('# q={0}\n'.format(q))
		f.write('# xi={0}\n'.format(xi))
		for row in t_archive.numpy():
			for idx,col in enumerate(row[n:]):
				f.write(str(col))
				if idx != k - 1:
					f.write('\t')
			f.write('\n')


def save_pareto_set_snapshot(execNum, output_subfolder, gen):
	""" Writes t_archive's x values of the current generation to a .pos file """
	num_digits = len(str(Gmax))
	gen_padded = str(gen).zfill(num_digits)

	with open(''.join([output_subfolder, '/Run', str(execNum) , '_gen', gen_padded,'.pos']), 'w') as f:
		f.write('# Lang=PyTorch\n')
		f.write('# Function={0}\n'.format(obj_function_name))
		f.write('# Scalarizer={0}\n'.format(scalarizer))
		f.write('# k={0}\n'.format(k))
		f.write('# n={0}\n'.format(n))
		f.write('# N={0}\n'.format(N))
		f.write('# q={0}\n'.format(q))
		f.write('# xi={0}\n'.format(xi))
		f.write('# Rmin={0}\n'.format(Rmin))
		f.write('# Rmax={0}\n'.format(Rmax))
		ranks_py = ranks.numpy()
		for row,sol in enumerate(t_archive.numpy()):
			for col,x in enumerate(sol[:n]):
				f.write(str(x))
				f.write('\t')
			f.write(str(ranks_py[row]))
			f.write('\n')


def save_pareto_front_snapshot(execNum, output_subfolder, gen):
	""" Writes t_archive's Fx values of the current generation to a .pos file """
	num_digits = len(str(Gmax))
	gen_padded = str(gen).zfill(num_digits)

	with open(''.join([output_subfolder, '/Run', str(execNum) , '_gen', gen_padded,'.pof']), 'w') as f:
		f.write('# Lang=PyTorch\n')
		f.write('# Function={0}\n'.format(obj_function_name))
		f.write('# Scalarizer={0}\n'.format(scalarizer))
		f.write('# k={0}\n'.format(k))
		f.write('# n={0}\n'.format(n))
		f.write('# N={0}\n'.format(N))
		f.write('# q={0}\n'.format(q))
		f.write('# xi={0}\n'.format(xi))
		ranks_py = ranks.numpy()
		for row,sol in enumerate(t_archive.numpy()):
			for col,Fx in enumerate(sol[n:]):
				f.write(str(Fx))
				f.write('\t')
			f.write('\n')


def update_archive(
		t_archive, ants_x, ants_Fx,
		ranks, us, norms):
	"""
	Combines the t_archive and ant solutions,
	sorts by (rank, magnitude, u) of each solution,
	then stores the top N solutions back in t_archive.
	"""
	union_solutions = torch.cat((t_archive, torch.cat((ants_x, ants_Fx), dim=1)), dim=0)

	ranks_max = torch.max(ranks)
	norms_max = torch.max(norms)
	us_max = torch.max(us)

	order_values = ranks * ranks_max * norms_max * us_max
	order_values.add_(norms * norms_max * us_max)
	order_values.add_(us)

	order = torch.argsort(order_values)

	t_archive = union_solutions[order[:N]]
	ranks = ranks[order]
	us = us[order]

	return t_archive, ranks, us


def compute_gauss_probs(ranks):
	"""
	Computes probability of selecting each solution of t_archive within roulette_wheel_selection function.
	"""
	# torch.ones(1) at the end prevents a TracerWarning.
	gauss_weights = torch.exp(-torch.pow(ranks[:N], 2) / (2.*q*q*N*N)) / (q*N*torch.sqrt(2.*np.pi*torch.ones(1)))
	gauss_probs = torch.cumsum(gauss_weights, dim=0)
	return gauss_probs / gauss_probs[-1]


def roulette_wheel_selection():
	"""
	Selects solutions to use as the Gaussian means based on their positional ranks within t_archive.
	PyTorch does not currently have a numpy.searchsorted equivalent.
	"""
	ants_random_values.uniform_(0, 1)
	g = gauss_probs_normalized.numpy()
	return torch.from_numpy(np.searchsorted(g, ants_random_values.numpy(), side='right').reshape(M,n))


def compute_means(t_archive, indices):
	"""
	Lookup Gaussian means in t_archive from given indices.
	Using Numpy for this operation runs roughly 30x faster than in PyTorch.
	"""
	return torch.as_tensor(np.diagonal(t_archive.numpy()[indices.numpy(), :n], axis1=1, axis2=2))


def compute_std_devs():
	"""
	Computes 'standard deviation' for each ant, using a non-standard formula.
	"""
	torch.mul(
		torch.sum(
			torch.abs(torch.add(ants_means.unsqueeze(dim=1), - t_archive[:, :n])),
			dim=1),
		std_dev_coefficient,
		out=std_devs)


def sample():
	"""
	Creates and samples Gaussian distributions for each ant's (mean, std_dev) in each positional dimension.
	"""
	torch.clamp(
		torch.normal(
			mean=ants_means,
			std=std_devs),
		min=Rmin,
		max=Rmax,
		out=ants_x)


def init_weight_vectors(H, k):
	"""
	Creates Weight Vectors, which are Simple Lattices used in the alpha/indicator calculations.
	The EPSILON value is added to all elements after the fact (instead of only 0 values)
	for the sake of simplicity and to reduce runtime.
	"""
	combinations = torch.tensor(list(itertools.product([x for x in range(H+1)], repeat=k)), dtype=dtype)
	return combinations[torch.sum(combinations, dim=1) == H]/H + EPSILON


def init_ref_points():
	"""
	Initialize z_ideal and z_nad with appropriate values from ants_Fx.
	"""
	global z_ideal
	global z_nad
	z_ideal = torch.min(t_archive[:,n:], dim=0)[0]
	z_nad = torch.max(t_archive[:,n:], dim=0)[0]


def update_ref_points(gen):
	"""
	Updates the RECORD, z_min, z_ideal, z_max, and z_nad.
	"""
	global z_min
	global z_ideal
	global z_max
	global z_nad

	z_min = torch.min(ants_Fx, dim=0)[0]
	z_ideal = torch.min(z_min, other=z_ideal)
	z_max = torch.max(ants_Fx, dim=0)[0]

	# Save z_max to RECORD
	RECORD[gen % MAX_RECORD_SIZE].copy_(z_max)

	if gen >= MAX_RECORD_SIZE - 1:
		mus = np.mean(RECORD.numpy(), axis=0)
		vs = np.sum(np.power(RECORD.numpy() - mus, 2), axis=0) / np.float32(MAX_RECORD_SIZE)
		for component in range(k):
			mu = mus[component]
			v = vs[component]
			if v > var_threshold:
				z_nad.fill_(max(z_max))
				break
			else:
				if torch.abs(z_nad[component] - z_ideal[component]) < tol_threshold:
					z_nad[component] = max(z_nad)
					mark[component] = MAX_RECORD_SIZE

				elif z_max[component] > z_nad[component]:
					z_nad[component] = z_max[component] + np.abs(z_max[component] - z_nad[component])
					mark[component] = MAX_RECORD_SIZE

				elif v == 0 and (z_max[component] <= mu or (z_max[component] - mu) > tol_threshold) and mark[component] == 0:
					v = max([sublist[component] for sublist in RECORD])
					z_nad[component] = (z_nad[component] + v)/2.
					mark[component] = MAX_RECORD_SIZE

			if mark[component] > 0:
				mark[component] -= 1


def normalize_obj_function(t_archive, ants_Fx, z_nad, z_ideal):
	"""
	Creates a normalized version of both t_archive's Fx and the ant's Fx.
	If z_nad[i] - z_ideal[i] == 0, then nFx == Fx for dimension i.
	"""
	vs = z_nad - z_ideal
	Fx = torch.cat((t_archive[:, n:], ants_Fx), dim=0)
	nFx = (Fx - z_ideal)/vs
	return torch.where(vs != 0., nFx, Fx)


def asf(union_nFx, WV):
	"""
	Achievement Scalarizing Function.
	Produces alpha values to be used in the R2 Ranking algorithm.
	"""
	return torch.max(torch.abs(union_nFx / WV.unsqueeze(dim=1)), dim=2)[0]


def vads(union_nFx, WV, WV_magnitudes):
	"""
	Vector Angle Distance Scaling.
	Produces alpha values to be used in the R2 Ranking algorithm.
	"""
	nFx_magnitudes = torch.sqrt(torch.sum(torch.pow(union_nFx, 2), 1))
	obj_vector_deviations = torch.pow(
		torch.sum((WV/WV_magnitudes).unsqueeze(1) *	(union_nFx/nFx_magnitudes.unsqueeze(1)), dim=2),
		exponent=p)
	return nFx_magnitudes / obj_vector_deviations


def r2_ranking():
	"""
	Implentation of the R2 indicator, which evaluates the desired aspects of a Pareto front approximation.
	Implemented using Numpy rather than PyTorch due to its seemingly vastly shorter runtime in indexing operations.
	"""
	global norms

	union_solutions = torch.cat((t_archive, torch.cat((ants_x, ants_Fx), dim=1)), dim=0)

	norms = torch.sqrt(torch.sum(torch.pow(union_solutions[:, n:n+k], 2), dim=1))
	alphas_py = alphas.numpy()

	order_values = alphas_py * np.max(alphas_py) + np.max(norms.numpy()) + norms.numpy()

	ranks_py = ranks.fill_(np.inf).numpy()
	comp_ranks_py = np.arange(1, N+M+1, dtype=np.float32)
	greater_than_list = np.empty([N+M], dtype=np.bool)

	alpha_indices_py = np.arange(N+M)
	indices_lists = np.argsort(order_values, axis=1)
	us_py = us.numpy()
	us_py[N:] += np.inf

	time_inner_start = time.time()
	for i in range(N):
		np.greater(
			ranks_py[indices_lists[i]],
			comp_ranks_py,
			out=greater_than_list)
		np.put(
			ranks_py,
			indices_lists[i][greater_than_list],
			comp_ranks_py)
		np.put(
			us_py,
			indices_lists[i][greater_than_list],
			alphas_py[i][alpha_indices_py[indices_lists[i]]])


def ant_solution_construction():
	""" Manages the construction of ant solutions. """
	global gauss_probs_normalized
	global ants_means
	global ants_Fx

	gauss_probs_normalized = compute_gauss_probs(ranks)
	indices = roulette_wheel_selection()
	ants_means = compute_means(t_archive, indices)
	compute_std_devs()
	sample()
	ants_Fx = obj_function(ants_x)


ants_x = torch.empty([M, n], dtype=dtype)
ants_Fx = torch.empty([M, k], dtype=dtype)
ants_means = torch.empty([M, n], dtype=dtype)
ants_random_values = torch.empty([M, n, 1], dtype=dtype)
std_devs = torch.empty([M, n], dtype=dtype)

t_archive = torch.empty([N, n+k], dtype=dtype)

RECORD = torch.empty([MAX_RECORD_SIZE, k], dtype=dtype)
z_min = torch.empty([k], dtype=dtype)
z_ideal = torch.empty([k], dtype=dtype)
z_nad = torch.empty([k], dtype=dtype)
z_max = torch.empty([k], dtype=dtype)
mark = torch.empty([k], dtype=dtype_int)

union_solutions = torch.empty([N+M, k+n], dtype=dtype)
union_nFx = torch.empty([N+M, k], dtype=dtype)
ranks = torch.empty([N+M], dtype=dtype)
us = torch.empty([N+M], dtype=dtype)
norms = torch.empty([N+M], dtype=dtype)
alphas = torch.empty([N+M, N], dtype=dtype)

WV = init_weight_vectors(H, k)
WV_magnitudes = torch.sqrt(torch.sum(torch.pow(WV, 2), dim=1)).unsqueeze_(dim=1)

flag_jit_traced = False

for run in range(num_runs):
	print('\nStarting run {0}/{1}'.format(run, num_runs - 1))
	time_run_start = time.time()

	t_archive.fill_(np.inf)
	RECORD.fill_(0.)
	mark.fill_(0.)
	ranks.fill_(np.inf)
	us.fill_(0.)

	# Initializes t_archive with random solutions
	for _ in range((np.ceil(N / M)).astype(np.int32)):
		ants_x.uniform_(Rmin,Rmax)
		ants_Fx = obj_function(ants_x)
		t_archive = torch.cat(
			(torch.cat((ants_x, ants_Fx), dim=1),
			t_archive),
			dim=0)[:N]

	init_ref_points()
	RECORD[0].copy_(z_nad)
	union_nFx = normalize_obj_function(t_archive, ants_Fx, z_nad, z_ideal)

	if scalarizer == 'ASF':
		alphas = asf(union_nFx, WV)
	elif scalarizer == 'VADS':
		alphas = vads(union_nFx, WV, WV_magnitudes)

	r2_ranking()

	# If PyTorch version >= 1.0.0, replace specific functions with JIT-compiled versions.
	if flag_jit_enabled and not flag_jit_traced:
		compute_gauss_probs = torch.jit.trace(compute_gauss_probs, (ranks), check_trace=False)
		obj_function = torch.jit.trace(obj_function, (ants_x), check_trace=False)
		if scalarizer == 'ASF':
			asf = torch.jit.trace(asf, (union_nFx, WV), check_trace=False)
		elif scalarizer == 'VADS':
			vads = torch.jit.trace(vads, (union_nFx, WV, WV_magnitudes), check_trace=False)
		update_archive = torch.jit.trace(update_archive, (t_archive, ants_x, ants_Fx, ranks, us, norms), check_trace=False)
		normalize_obj_function = torch.jit.trace(normalize_obj_function, (t_archive, ants_Fx, z_nad, z_ideal), check_trace=False)
		flag_jit_traced = True

	for gen in range(Gmax):
		ant_solution_construction()

		update_ref_points(gen)

		union_nFx = normalize_obj_function(t_archive, ants_Fx, z_nad, z_ideal)

		if scalarizer == 'ASF':
			alphas = asf(union_nFx, WV)
		elif scalarizer == 'VADS':
			alphas = vads(union_nFx, WV, WV_magnitudes)

		r2_ranking()

		t_archive, ranks, us = update_archive(t_archive, ants_x, ants_Fx, ranks, us, norms)

		if flag_create_snapshots:
			save_pareto_set_snapshot(run, snapshot_subfolder, gen)
			save_pareto_front_snapshot(run, snapshot_subfolder, gen)

	time_run_end = time.time()
	print('Seconds taken in run: {0}'.format(time_run_end-time_run_start))

	save_pareto_front(run)
	save_pareto_set(run)