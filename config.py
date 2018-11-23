#!/usr/bin/env python3
"""
This script provides all parameters for the main program.
"""

from configparser import ConfigParser
from numpy import math as np_math

scalarizers_list = ('ASF', 'VADS')

# n:	Number of search space dimensions
# k:	Number of objective functions, and so the number of results per solution
# Rmin:	Lower domain bound for all search space dimensions
# Rmax:	Upper domain bound for all search space dimensions
# q:	Controls diversification process of the search
# xi:	Controls convergence rate (used in standard deviation calculations)
# Gmax:	Number of generations
# M:	Number of ants
# N:	Size of t_archive, where the best solutions are stored.
# 		Each row of t_archive holds a solution's [(n) x values, (k) Fx values]
# H:	Proportional parameter utilized for the construction of N convex weight vectors.


def main(*args):
	global n
	global k
	global M

	try:
		config_path = args[0][0]
	except IndexError:
		print('Fatal error: Please specify the path of the configuration file as the first argument.\nExiting')
		quit()
	# Number of independent runs per execution.
	num_runs = args[0][1] if len(args[0]) >= 2 else 1
	try:
		num_runs = int(num_runs)
	except ValueError:
		print('\nInvalid number of runs provided. Defaulting to 1.')
		num_runs = 1

	# Method of scalarization for the R2 Ranking algorithm.
	scalarizer = str(args[0][2]).upper() if len(args[0]) >= 3 else 'ASF'
	if scalarizer not in scalarizers_list:
		print('\nError: Provided scalarizer name not implemented.')
		print('Available scalarization functions: {0}'.format(', '.join(scalarizers_list)))
		print('Defaulting to {0}.\n'.format(scalarizers_list[0]))
		scalarizer = scalarizers_list[0]
		quit()

	try:
		# Searches for the appropriate configuration file.
		config = ConfigParser()
		config.read(config_path)
		chosenConfig = config['Parameters']
		
		# Reads from configuration file.
		obj_function_name = chosenConfig['Function']
		n = int(chosenConfig['n'])
		k = int(chosenConfig['k'])
		Rmin = float(chosenConfig['Rmin'])
		Rmax = float(chosenConfig['Rmax'])
		q = float(chosenConfig['q'])
		xi = float(chosenConfig['xi'])
		Gmax = int(chosenConfig['Gmax'])
		MAX_RECORD_SIZE = int(chosenConfig['MAX_RECORD_SIZE'])
		EPSILON = 1e-4
		var_threshold = float(chosenConfig['varThreshold'])
		tol_threshold = float(chosenConfig['tolThreshold'])
		H = int(chosenConfig['H'])
		M = N = np_math.factorial(H + k - 1) // np_math.factorial(k - 1) // np_math.factorial(H)
	except KeyError:
		print('Fatal error: Unable to read configuration file.\n'
			+ 'Ensure that the provided path is correct.')
		quit()

	print('\nParameters:')
	print('Objective Function:\t{0}'.format(obj_function_name))
	print('\tScalarizer:\t{0}'.format(scalarizer))
	print('\tn:\t{0:5}'.format(n))
	print('\tk:\t{0:5}'.format(k))
	print('\tRmin:\t{0:5}'.format(Rmin))
	print('\tRmax:\t{0:5}'.format(Rmax))
	print('\tq:\t{0:5}'.format(q))
	print('\txi:\t{0:5}'.format(xi))
	print('\tGmax:\t{0:5}'.format(Gmax))
	print('\tM:\t{0:5}'.format(M))
	print('\tH:\t{0:5}'.format(H))
	print('\tMAX_ARCHIVE_SIZE:\t{0}\n'.format(N))

	return (obj_function_name, n, k, M, Rmin, Rmax, q, xi,
			Gmax, N, H, EPSILON, MAX_RECORD_SIZE, var_threshold,
			tol_threshold, num_runs, scalarizer)

if __name__ == '__main__':
	main()