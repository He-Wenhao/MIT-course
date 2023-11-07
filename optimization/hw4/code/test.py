import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the datasets
dataA = pd.read_csv('dataA.csv')
dataB = pd.read_csv('dataB.csv')

# convert to float
dataA = dataA.applymap(float)
dataB = dataB.applymap(float)
# Separate into covariates and labels
XA = dataA.iloc[1:, :2].values
yA = dataA.iloc[1:, 2].values

XB = dataB.iloc[1:, :2].values
yB = dataB.iloc[1:, 2].values

# Define the loss function
def loss_function(beta1, beta2):
    loss_A = (yA - beta1 * XA[:, 0] - beta2 * XA[:, 1]) ** 2
    loss_B = (yB - beta1 * XB[:, 0] - beta2 * XB[:, 1]) ** 2
    return (1 / (2 * len(yA))) * np.sum(np.minimum(loss_A, loss_B))

# Calculate loss for a grid of values
beta1_values = np.linspace(-25, 75, 100)
beta2_values = np.linspace(-40, 40, 100)

Z = np.zeros((len(beta1_values), len(beta2_values)))

for i, beta1 in enumerate(beta1_values):
    for j, beta2 in enumerate(beta2_values):
        Z[i, j] = loss_function(beta1, beta2)

# Plot contour plot
plt.contour(beta1_values, beta2_values, Z.T, 50, cmap='viridis')
plt.colorbar(label='Loss Value')
plt.xlabel('$\\beta_1$')
plt.ylabel('$\\beta_2$')
plt.title('Contour plot of the loss function')
plt.show()
