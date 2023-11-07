# %%
week = 12

# %%
import csv
import numpy as np

# Initialize an empty 2D array (list of lists) to store the data
data = []

# Specify the CSV file name
csv_file = "requirements.csv"

# Open the CSV file and read its contents
with open(csv_file, "r") as file:
    # Create a CSV reader object
    csv_reader = csv.reader(file, delimiter=',')
    
    # Iterate over each row in the CSV file
    for row in csv_reader:
        # Convert each element in the row to an integer and store it in a list
        row_data = [x for x in row[1:]]
        
        # Append the row to the 2D array
        data.append(row_data)

# Now, 'data' contains the CSV data as a 2D array
data = data[1:]

t = np.array(data)
t = np.array([[float(cell) for cell in row] for row in t])


print(t.shape)
print(t)

# %%
# Initialize an empty 2D array (list of lists) to store the data
data = []

# Specify the CSV file name
csv_file = "profit.csv"

# Open the CSV file and read its contents
with open(csv_file, "r") as file:
    # Create a CSV reader object
    csv_reader = csv.reader(file, delimiter=',')
    
    # Iterate over each row in the CSV file
    for row in csv_reader:
        # Convert each element in the row to an integer and store it in a list
        row_data = [x for x in row[1:]]
        
        # Append the row to the 2D array
        data.append(row_data)

data = data[1:]
# Now, 'data' contains the CSV data as a 2D array

k = np.matrix(data)
k = k.transpose()
k = np.array(k)

k = [[float(cell) for cell in row] for row in k]
k = k[0]
k = np.array(k)
print(np.array(k).shape)
print(k)


# %%
# Initialize an empty 2D array (list of lists) to store the data
hc = 0

# Specify the CSV file name
csv_file = "holding.csv"

# Open the CSV file and read its contents
with open(csv_file, "r") as file:
    
    # Create a CSV reader object
    csv_reader = csv.reader(file, delimiter=',')
    
    # Iterate over each row in the CSV file
    for row in csv_reader:
        if row[0] == str(week):
            hc = row[1]

h = [[hc]*500]
h = np.array(h)
h = [[float(cell) for cell in row] for row in h]
h = h[0]
h = np.array(h)
print(np.array(h).shape)
print(h)

# %%
# Initialize an empty 2D array (list of lists) to store the data
a = []

# Specify the CSV file name
csv_file = "availability.csv"

# Open the CSV file and read its contents
with open(csv_file, "r") as file:
    # Create a CSV reader object
    csv_reader = csv.reader(file, delimiter=',')
    
    # Iterate over each row in the CSV file
    for row in csv_reader:
        a.append([row[week]])

a = a[1:]
a = [[float(cell) for cell in row] for row in a]
a = np.matrix(a)
a = a.transpose()
a = np.array(a)[0]
print(a.shape)
print(a)

# %%
# Initialize an empty 2D array (list of lists) to store the data
d = []

# Specify the CSV file name
csv_file = "demand.csv"

# Open the CSV file and read its contents
with open(csv_file, "r") as file:
    # Create a CSV reader object
    csv_reader = csv.reader(file, delimiter=',')
    
    # Iterate over each row in the CSV file
    for row in csv_reader:
        d.append([row[week]])

d = d[1:]
d = [[float(cell) for cell in row] for row in d]
d = np.matrix(d)
d =d.transpose()
d = np.array(d)[0]
print(d.shape)
print(d)

# %%
import cvxpy as cp
import numpy as np

# Define the variables
#n = len(d)  # Assuming d and a have the same length
p = cp.Variable(100)
m = cp.Variable(500)

# Define the objective function to maximize

objective = cp.Maximize(k @ p - h @ m)

# Define the constraints
constraints = [
    m == t @ p,
    0 <= p, p <= d,
    0 <= m, m <= a
]

# Create the optimization problem
problem = cp.Problem(objective, constraints)

# Solve the problem
problem.solve()
# Extract and print the results
p_optimal = p.value  # Optimal values for p
m_optimal = m.value  # Optimal values for m

objective_value = problem.value  # Maximum value of k*p - h*m

print("Optimal Solution:")
print(f"p = {p_optimal}")
print(f"m = {m_optimal}")
print(f"Optimal Objective Function Value: k*p - h*m = {objective_value}")




