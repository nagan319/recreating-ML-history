# goal: digit classification AI

import numpy as np

# inner dimensions must match -> OK

# input = np.array([[1.0, 2.0, 3.0, 4.0]])
# weights = np.array([[1.0], [2.0], [3.0], [4.0]])
# print(input @ weights)

# now we can write a function to handle forward propagation:

N = 5

node_amts = [256, 64, 64, 10]

# inputs are column vectors
# weights are row-major

input = np.array([1, 2, 3, 4])
w = np.array([2, 3, 4, 5])
b = np.array([1, 1, 1, 1])

print(np.dot(input.T, w) + b)
