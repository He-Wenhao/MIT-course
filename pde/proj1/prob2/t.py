import numpy as np
# test activated or not
def is_acti(acti_lst,cord):
    flag = 0
    for block in acti_lst:
        ac_y_cor = (block-1)//4
        ac_x_cor = (block-1)%4
        x_l = 1/6*(1+ac_x_cor)
        x_h = 1/6*(2+ac_x_cor)
        y_l = 1/6*(1+ac_y_cor)
        y_h = 1/6*(2+ac_y_cor)
        if cord[0] >= x_l and cord[0] <= x_h and cord[1] >= y_l and cord[1] <= y_h:
            flag = 1
    return flag
import numpy as np
# test activated or not
def is_acti(acti_lst,cord):
    flag = 0
    for block in acti_lst:
        ac_y_cor = (block-1)//4
        ac_x_cor = (block-1)%4
        x_l = 1/6*(1+ac_x_cor)
        x_h = 1/6*(2+ac_x_cor)
        y_l = 1/6*(1+ac_y_cor)
        y_h = 1/6*(2+ac_y_cor)
        if cord[0] >= x_l and cord[0] <= x_h and cord[1] >= y_l and cord[1] <= y_h:
            flag = 1
    return flag
# Define the grid size N

import numpy as np
import matplotlib.pyplot as plt

def relax(u, f,  nu, h):
    N = u.shape[0]
    u_new = np.copy(u)
    for _ in range(nu):
        for i in range(1, N - 1):
            for j in range(1, N - 1):
                u_new[i, j] = 0.25 * (u[i - 1, j] + u[i + 1, j] + u[i, j - 1] + u[i, j + 1] - h * h * f[i, j])
        u, u_new = u_new, u
    return u
        
        
def restrict(r):
    
    N = r.shape[0]
    Nc = N  // 2
    rc = np.zeros((Nc, Nc))
    for i in range(1, Nc - 1):
        for j in range(1, Nc - 1):
            #rc[i, j] = 0.25 * (r[2 * i, 2 * j] + r[2 * i + 1, 2 * j] + r[2 * i, 2 * j + 1] + r[2 * i + 1, 2 * j + 1])
            rc[i,j] = r[2*i,2*j]
    
    return rc

def calc_res(f,u,h):
    N = f.shape[0]
    u_res = np.zeros((N+1, N+1))
    for i in range(1, N - 1):
        for j in range(1, N - 1):
                u_res[i,j] = f[i,j] + (4*u[i, j] -  (u[i - 1, j] + u[i + 1, j] + u[i, j - 1] + u[i, j + 1] ) )/(h**2)
    return u_res

    
def interpolate(e):
    Nc = e.shape[0]
    N = 2 * Nc 
    ef = np.zeros((N+1, N+1))
    for i in range(1, N ):
        for j in range(1, N ):
            ef[i, j] = e[i // 2, j // 2]
    return ef

def two_grid_cycle( u, f ,A , h, nu1, nu2, nuc, iteration,N,A_res):
    u_temp = np.copy(u)
    for iteri in range(iteration):
        # Relaxation with initial guess
        u1_3 = relax(u_temp, f,  nu1, h)
         

        # Compute residual and restrict
        #r = f - (A.reshape(((N+1)**2,(N+1)**2)) .dot( u1_3.reshape((N+1)**2))).reshape((N+1,N+1))
        r = calc_res(f,u1_3,h)
        rc = restrict(r)

        #print('r',np.linalg.norm((r).reshape((r.shape[0])**2), ord=2))
        #print(rc)
        # Coarse grid correction
        #eh = np.linalg.solve(A_res.reshape(((N//2)**2,(N//2)**2)), rc.reshape((N//2)**2)).reshape((N//2,N//2))
        eh = relax(np.zeros(((N ) // 2,(N ) // 2)), rc,  nuc, 2*h)
        # Prolongate and correct
        e = interpolate(eh)
        u2_3 = u1_3 + e  ##############

        # Relaxation with corrected guess
        u_temp = relax(u2_3, f,  nu2, h)
       
        
    return u_temp

# Set up your grid, boundary conditions, and f (right-hand side)
N = 25 - 1
dx = 1.0 / N
u = np.zeros((N+1, N+1))
f = np.zeros((N+1, N+1))
act_lst = [1,7,14,16]
for i in range(0,N+1):
    for j in range(0,N+1):
        if is_acti(act_lst,(i*dx,j*dx)):
        #if i<2:
            f[i,j] = -1

A = np.zeros((N + 1, N + 1, N + 1, N + 1))
for i in range(0, N+1):
    for j in range(0, N+1):
        A[i,j,i,j] = 1
        if i in range(1, N) and j in range(1, N):
            A[i,j,i+1,j] = -0.25
            A[i,j,i-1,j] = -0.25
            A[i,j,i,j+1] = -0.25
            A[i,j,i,j-1] = -0.25
            
A_res = np.zeros((N //2, N //2, N //2, N //2))
for i in range(0, N//2):
    for j in range(0, N//2):
        A_res[i,j,i,j] = 1
        if i in range(1, N//2-1) and j in range(1, N//2-1):
            A_res[i,j,i+1,j] = -0.25
            A_res[i,j,i-1,j] = -0.25
            A_res[i,j,i,j+1] = -0.25
            A_res[i,j,i,j-1] = -0.25

# Number of iterations for relaxation
nu1 = 2
nu2 = 2
nuc = 100
iteration = 50

# Perform a two-grid cycle
u = two_grid_cycle(u, f, A, dx,nu1, nu2, nuc, iteration,N,A_res)




x = np.linspace(0, 1, N + 1)  # x-coordinates
y = np.linspace(0, 1, N + 1)  # y-coordinates
# Plot the solution
X, Y = np.meshgrid(x, y)
plt.contourf(X, Y, u, levels=20, cmap='viridis')
plt.colorbar()
plt.title("Solution of Poisson's Equation with Relaxation (Jacobi Method)")
plt.xlabel("j-axis")
plt.ylabel("i-axis")
plt.show()

