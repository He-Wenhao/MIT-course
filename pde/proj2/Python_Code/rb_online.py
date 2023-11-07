from utils import *
import numpy as np

def reduced_basis_online(mu, ANq, FN):
    """
    -------------------------------------------------------------------------

    ReducedBasisOnline.m

    -------------------------------------------------------------------------
    Reduced basis on-line: computes the temperature distribution and root
    temperature using the reduced basis computed by the off-line part.

    The reduced basis data ReducedBasis.mat should have been previously
    loaded using the comand "load ReducedBasis".
    -------------------------------------------------------------------------

    INPUT   mu      thermal conductivities of sections and Biot number 1x5
            N       dimension of the reduced basis
            ANq     reduced matrix
            FN      reduced source

    OUTPUT  uN      temperature discretization in the fin from reduced basis
            TrootN  root temperature from reduced basis

    -------------------------------------------------------------------------
    """

    # Parameter setup
    sigma = np.ones(6)
    sigma[:4] = mu[:4]
    sigma[5] = mu[4]       # Bi

    # Construct A_N from AN_qs
    # Fill me in!

    # Solve the system
    uN = np.linalg.solve(AN, FN)

    # Compute the temperature at the root
    # Fill me in!

    return uN, TrootN