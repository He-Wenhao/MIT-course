from utils import *
from thermal_fin_soln import thermal_fin
import numpy as np
from scipy.sparse import lil_matrix, csc_matrix

"""
 -------------------------------------------------------------------------

   ReducedBasisOffline.m

 -------------------------------------------------------------------------
   Reduced basis off-line: computes the reduced basis matrices and source
   vector from a sample.
 -------------------------------------------------------------------------
"""

# Load coarse mesh and sample data
grids = load('grids')
mesh = grids['coarse']
sn = load('sn')['sn']

# Solve the sample cases and construct Z
N = sn.shape[0]
Z = np.zeros((mesh['n_nodes'], N))

# Construct Z - fill me in!
# ...

# Initialization of reduced-base matrix and vector
F = lil_matrix((mesh['n_nodes'], 1))
ANq = []    # Using a list data structure to hold the six ANq matrices

# Calculate ANq and FN - fill me in!
# ...

# Save the reduced matrix and vector
with shelve.open('rb_offline_data') as shelf:
    shelf['ANq'] = ANq
    shelf['FN'] = FN
