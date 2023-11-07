import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve as solve


def thermal_fin(mesh, mu):

    """
    -------------------------------------------------------------------------

    ThermalFin.py

    -------------------------------------------------------------------------
    Computes the temperature distribution and root temperature for a fin
    using the Finite Element Method.
    -------------------------------------------------------------------------

    INPUT   mesh    grid label (coarse, medium, or fine)
            mu      thermal conductivities of sections and Biot number 1x5

    OUTPUT  u       temperature discretization in the fin
            Troot   root temperature

    -------------------------------------------------------------------------
    """

    # Parameters setup: rearrange the values in mu
    # kappa = [k1, k2, k3, k4, k0, Bi]

    kappa = np.ones(6)
    kappa[:4] = mu[:4]
    kappa[5] = mu[4]       # Bi

    A = lil_matrix((mesh['n_nodes'], mesh['n_nodes']))
    F = lil_matrix((mesh['n_nodes'], 1))

    # Domain Interior
    for i in range(5):
        for n in range(len(mesh['theta'][i])):
            phi = mesh['theta'][i][n,:]
            """
            Fill me in!
            """
            A[phi[:, None], phi] += A_local

    # Boundaries (not root)
    i=5
    for n in range(len(mesh['theta'][i])):
        phi = mesh['theta'][i][n,:]
        """
        Fill me in!
        """
        A[phi[:, None], phi] += A_local

    # Root boundary
    i=6
    for n in range(len(mesh['theta'][i])):
        phi = mesh['theta'][i][n,:]
        """
        Fill me in!
        """
        F[phi] = F[phi] + F_local[:, None]


    # Solve for the temperature distribution
    u = solve(A.asformat('csc'), F.asformat('csc'))

    # Compute the temperature at the root
    # Fill me in!

    return Troot, u