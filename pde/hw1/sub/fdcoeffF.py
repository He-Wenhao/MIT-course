import matplotlib

from pylab import *
from scipy.special import factorial
import numpy as np


"""
def fdcoeffF(k, xbar, x):

Compute coefficients for finite difference approximation for the
derivative of order k at xbar based on grid values at points in x.

This function returns a row vector c of dimension 1 by n, where n=length(x),
containing coefficients to approximate u^{(k)}(xbar), 
the k'th derivative of u evaluated at xbar,  based on n values
of u at x(1), x(2), ... x(n). 

If U is a column vector containing u(x) at these n points, then 
c*U will give the approximation to u^{(k)}(xbar).

Note for k=0 this can be used to evaluate the interpolating polynomial 
itself.

Requires length(x) > k.  
Usually the elements x(i) are monotonically increasing
and x(1) <= xbar <= x(n), but neither condition is required.
The x values need not be equally spaced but must be distinct.  

This program should give the same results as fdcoeffV.m, but for large
values of n is much more stable numerically.

 Based on the program "weights" in 
   B. Fornberg, "Calculation of weights in finite difference formulas",
   SIAM Review 40 (1998), pp. 685-691.

Note: FornfdcoeffF(berg's algorithm can be used to simultaneously compute the
coefficients for derivatives of order 0, 1, ..., m where m <= n-1.
this gives a coefficient matrix C(1:n,1:m) whose k'th column gives
the coefficients for the k'th derivative.

In this version we set m=k and only compute the coefficients for
derivatives of order up to order k, and then return only the k'th column
of the resulting C matrix (converted to a row vector).  
This routine is then compatible with fdcoeffV.   
It can be easily modified to return the whole array if desired.

From  http://www.amath.washington.edu/~rjl/fdmbook/  (2007)
"""

def fdcoeffF(k, xbar, x):
    n = len(x) - 1
    if k > n:
        raise ValueError('*** len(x) must be larger than k')

    m = k  # for consistency with Fornberg's notation
    c1 = 1.
    c4 = x[0] - xbar
    C = zeros((n+1,m+1))
    C[0,0] = 1.
    for i in range(1,n+1):
        mn = min(i,m)
        c2 = 1.
        c5 = c4
        c4 = x[i] - xbar
        for j in range(i):
            c3 = x[i] - x[j]
            c2 = c2*c3
            if j==i-1:
                for s in range(mn,0,-1):
                    C[i,s] = c1*(s*C[i-1,s-1] - c5*C[i-1,s])/c2
                C[i,0] = -c1*c5*C[i-1,0]/c2
            for s in range(mn,0,-1):
                C[j,s] = (c4*C[j,s] - s*C[j,s-1])/c3
            C[j,0] = c4*C[j,0]/c3
        c1 = c2

    c = C[:,-1] # last column of C
    return c

# Initialize empty lists to store the data
x_list = []
f_list = []

# Specify the path to your text file
file_path = 'data_points_p4.txt'

# Open the file for reading
with open(file_path, 'r') as file:
    # Iterate through each line in the file
    for line in file:
        # Split each line into columns using space as the delimiter
        columns = line.strip().split()
        
        # Check if the line contains at least two columns
        if len(columns) >= 2:
            # Add the first column to x_list and the second column to f_list
            x_list.append(float(columns[0]))
            f_list.append(float(columns[1]))

# Print the extracted data
#print("x_list:", len(x_list))
#print("f_list:", f_list)

    
A = []
for i in range(21):
    A.append(fdcoeffF(2,x_list[i],x_list))

A = np.matrix(A)
#A = A.transpose()
for i in range(21):
    A[0,i] = 0
    A[20,i] = 0
A[0,0] = 1
A[20,20] = 1
#print(A)
f_list = np.array(f_list)
f_list[0] = 0
f_list[20] = 0
print(f_list)
print(len(f_list))

x_res = -np.linalg.inv(A).dot(f_list)

print(x_res)


############# plot
import numpy as np
import matplotlib.pyplot as plt

# Define the range of x values from 0 to 2*pi
x = np.linspace(0, 2 * np.pi, 1000)

# Define the function
y = np.sin(x) + 1/4 * np.cos(2*x) - 1/4

# Create the plot
plt.plot(x, y, label=r'$\sin(x) + \frac{1}{4}\cos(2x) - \frac{1}{4}$')
plt.plot(x_list, np.array(x_res)[0], label='my result')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Plot of sin(x) + 1/4 cos(2x) - 1/4')
plt.grid(True)
plt.legend()
plt.xlim(0, 2 * np.pi)  # Set x-axis limits to [0, 2*pi]
#plt.ylim(-1.5, 1.5)    # Set y-axis limits as needed
plt.show()
