from utils import *
from thermal_fin import thermal_fin
from rb_online import reduced_basis_online

# Example for computing FEM solution
loaded_vars = load('grids')
mesh = loaded_vars['coarse']

mu = [0.4, 0.6, 0.8, 1.2, 0.1]

Troot, u = thermal_fin(mesh, mu)
print('Troot =', Troot)
plot(mesh, u)

# Example for running RB online
rb_offline_data = load('rb_offline_data')
ANq = rb_offline_data['ANq']
FN = rb_offline_data['FN']

mu1 = [0.4, 0.6, 0.8, 1.2, 0.1]
mu2 = [1.8, 4.2, 5.7, 2.9, 0.3]

uN, TrootN = reduced_basis_online(mu1, ANq, FN)

print(TrootN)