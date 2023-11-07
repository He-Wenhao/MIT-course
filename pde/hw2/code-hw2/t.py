import numpy as np
import matplotlib.pyplot as plt


def exact_solution(x, epsilon, s):
    return 1 - s*x + (s-1) * (1 - np.exp(-x/epsilon)) / (1 - np.exp(-1/epsilon))


def solve_convection_diffusion(N, epsilon, s, scheme):
    dx = 1.0 / N
    A = np.zeros((N+1, N+1))
    b = np.zeros(N+1)
    
    A[0, 0] = 1
    b[0] = 1
    A[N, N] = 1
    b[N] = 0
    
    for j in range(1, N):
        if scheme == 'A':
            A[j, j-1] = -epsilon / dx**2 + 1 / (2*dx)
            A[j, j] = 2 * epsilon / dx**2
            A[j, j+1] = -epsilon / dx**2 - 1 / (2*dx)
        
        elif scheme == 'B':
            A[j, j-1] = -epsilon / dx**2 + 1 / dx
            A[j, j] = 2 * epsilon / dx**2 - 1 / dx
            A[j, j+1] = -epsilon / dx**2
        
        elif scheme == 'C':
            if j < N-1:
                A[j, j-1] = -epsilon / dx**2
                A[j, j] = 2 * epsilon / dx**2 + 3 / (2*dx)
                A[j, j+1] = -epsilon / dx**2 - 4 / (2*dx)
                A[j, j+2] = 1 / (2*dx)
            else:
                A[j, j-1] = -epsilon / dx**2
                A[j, j] = 2 * epsilon / dx**2 + 2 / (2*dx)
                A[j, j+1] = -epsilon / dx**2 - 4 / (2*dx)
        
        b[j] = s
    
    u = np.linalg.solve(A, b)
    return u

epsilon = 1/20  # Given value for epsilon
s = 10  # Source term
N0 = 20
ord_lst = list(range(1,7))
err_A = []
err_B = []
err_C = []

for ord in ord_lst:
    N = N0*(2**ord)
    x = np.linspace(0, 1, N+1)
    # Solve for each scheme
    uA = solve_convection_diffusion(N, epsilon, s, 'A')
    uB = solve_convection_diffusion(N, epsilon, s, 'B')
    uC = solve_convection_diffusion(N, epsilon, s, 'C')
    u_exact = exact_solution(x, epsilon, s)
    
    err_A.append(np.linalg.norm(uA-u_exact))
    err_B.append(np.linalg.norm(uB-u_exact))
    err_C.append(np.linalg.norm(uC-u_exact))
    

    
# Plot
plt.figure(figsize=(10,6))
plt.plot(ord_lst, err_A, label="Central Difference (A)")
plt.plot(ord_lst, err_B, label="Two-Point Upwind (B)")
plt.plot(ord_lst, err_C, label="Three-Point Second Order Upwind (C)")

plt.xlabel("order")
plt.ylabel("error")
plt.title("Convection-Diffusion Solutions Errors for Different Schemes")
plt.legend()
plt.grid(True)
plt.show()