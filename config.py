#!/usr/bin/env python3
"""
This script provides all parameters for the main program.
"""

from configparser import ConfigParser
from numpy import math as np_math

# Number of independent runs per execution.
num_runs = 30

# Change the following 3 lines to specify the configuration file.
# Possible functions: ['DTLZ']
function_family = 'DTLZ'

# Possible function numbers: ['1' ... '7']
# Note that DTLZ1 and DTLZ3 do not converge for this approach.
function_number = '6'

# Possible function dimensions: ['3' ... '10']
function_dimension = '7'

# Possible scalarizing: ['ASF', 'VADS']
scalarizing = 'ASF'

# Searches for the appropriate configuration file.
config = ConfigParser()
config.read('./config/' + function_family + '/'
	+ function_dimension + 'D/'
	+ function_family + function_number + '_' + function_dimension + 'D.config')

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

print("Parameters:")
print("\nObjective Function:\t{0}".format(obj_function_name))
print("\tScalarizing:\t{0}".format(scalarizing))
print("\tn:\t{0:5}".format(n))
print("\tk:\t{0:5}".format(k))
print("\tRmin:\t{0:5}".format(Rmin))
print("\tRmax:\t{0:5}".format(Rmax))
print("\tq:\t{0:5}".format(q))
print("\txi:\t{0:5}".format(xi))
print("\tGmax:\t{0:5}".format(Gmax))
print("\tM:\t{0:5}".format(M))
print("\tH:\t{0:5}".format(H))
print("\tMAX_ARCHIVE_SIZE:\t{0}\n".format(N))