# Recreating 'Learning Representations by Backpropagation' (Rumelhart et al.)

## Structure
- Forward propagation (python)
- Gradients (math)
- Backpropagation (math)
- Backpropagation (python)
- Complete training
## Introduction

Rumelhart et al.'s 'Learning Representations by Backpropagation' introduces the concept of a multilayer perceptron network (MLP). We will recreate this neural network using both raw Numpy as well as more advanced libraries, covering the underlying mathematical concepts along the way.

## Matrix Multiplication - Conceptual

There are a few things we need to understand about matrix multiplication before we attempt to build a neural network similar to the one described in Rumelhart et al.

Two matrices A, B can only be multiplied if the number of **rows** in the first matrix is equal to the number of **columns** in the second matrix. 

In math, the first value in the shape of a matrix is the number of columns, while the second is the number of rows.

Two matrices with the indicated shapes (m, n) and (n, p) will result in a product with the shape (m, p): 

$$
A_{(m,n)}*B_{(n,p)}=C_{(m,p)}
$$

When you multiply two matrices, each element in the resultant matrix comes from the dot product of a row from the first matrix and a column from the second matrix:

$$
C_{[0, 0]}=A_{[0, 0]}B_{[0, 0]}+A_{[0, 1]}B_{[1, 0]}+A_{[0, 2]}B_{[2, 0]}
$$

The value-by-value matrix multiplication as a whole will resemble the following:

```math
\begin{bmatrix} 
A_{[0, 0]}&A_{[0, 1]}&A_{[0, 2]}\\
A_{[1, 0]}&A_{[1, 1]}&A_{[1, 2]}\\
\end{bmatrix}
\begin{bmatrix} 
B_{[0, 0]}&B_{[0, 1]}\\
B_{[1, 0]}&B_{[1, 1]}\\
B_{[2, 0]}&B_{[2, 1]}\\
\end{bmatrix}
=
```

$$
\begin{bmatrix} 
A_{[0, 0]}B_{[0, 0]}+A_{[0, 1]}B_{[1, 0]}+A_{[0, 2]}B_{[2, 0]}&&A_{[0, 0]}B_{[0, 1]}+A_{[0, 1]}B_{[1, 1]}+A_{[0, 2]}B_{[2, 1]}\\
A_{[1, 0]}B_{[0, 0]}+A_{[1, 1]}B_{[1, 0]}+A_{[1, 2]}B_{[2, 0]}&&A_{[1, 0]}B_{[0, 1]}+A_{[0, 1]}B_{[1, 1]}+A_{[0, 2]}B_{[2, 1]}\\
\end{bmatrix}
$$

The transpose of a matrix is simply the same matrix with the rows and columns flipped:

$$
A^T=
\begin{bmatrix}
A_{[0, 0]}&A_{[1, 0]}\\
A_{[0, 1]}&A_{[1, 2]}\\
A_{[0, 2]}&A_{[1, 3]}\\
\end{bmatrix}
$$

## Matrix Multiplication - Numpy

Now that we understand matrix multiplication at a conceptual level, we can see how it can be implemented in Python's Numpy library, which introduces the array object as an efficient representation of matrices.

A 3 by 2 matrix can be defined as follows:

```python
import numpy as np

A = np.array([[1, 2], [3, 4], [5, 6]])
```

This code block is equivalent to the following matrix:

```math
\begin{bmatrix}
1&2\\
3&4\\
5&6
\end{bmatrix}
```

Matrix multiplication can be performed using any of the following functions:

```python
A = np.array([[1, 2, 3], [4, 5, 6]])
B = np.array([[1, 2], [3, 4], [5, 6]])

C = np.dot(A, B)
C = A @ B
C = np.matmul(A, B)
```

This code block is equivalent to the following expression:

```math
\begin{bmatrix} 
1&2&3\\
1&5&6
\end{bmatrix}
\begin{bmatrix} 
1&2\\
3&4\\
5&6
\end{bmatrix}
=
\begin{bmatrix} 
22&28\\
46&58
\end{bmatrix}
```

The transpose of a matrix in Numpy can be represented as follows:

```python
A_transposed = A.T
```

 The shape of Numpy arrays can be found using the 'shape' function, and they can be easily reshaped using 'reshape':
 
 ```python
A = np.array([[1, 2], [3, 4], [5, 6]])
print(A.shape)  # Output: (3, 2)

B = np.array([1, 2, 3, 4, 5, 6])
reshaped_B = B.reshape((2, 3))  
print(reshaped_B)

# [[1 2 3]
# [4 5 6]]
```


## Sigmoid Activation Function

The activation function ensures that all activation values are normalized and within the range 0 and 1. The simplest activation function is the sigmoid function.

$$
\sigma(x)=\frac{1}{1+e^{-x}}
$$

The derivative of the sigmoid function is the following (use the quotient rule, then re-express the function in terms of itself):

$$
\sigma'(x)=\sigma(x)(1-\sigma(x))
$$

In Rumelhart et al.'s MLP, it is applied element-wise to vectors representing various layers of the network. It can be computed in Numpy in the following way:

```python
v = np.array([1.0, 2.0, -1.0, 0.5])

sigmoid_v = 1 / (1 + np.exp(-v))

sigmoid_derivative_v = sigmoid_v * (1 - sigmoid_v)
```


## Forward Propagation - Conceptual

Forward propagation is fairly simple.

The input values can be represented simply as a vector of n input values:

$$
\textbf{X}=
\begin{bmatrix}
x_1\\
x_2\\
...\\
x_n
\end{bmatrix}
$$

We'll use A(n) to represent the input vector at the Lth layer of the network. For the zeroth layer, it is simply equal to X:
$$
\textbf{A}^{(0)}=\textbf{X}
$$

The weights for a given layer can be represented by a matrix of the shape (m, n), where m is the number of neurons in the next layer and n is the number of inputs:

$$
\textbf{W}^{(L)}=
\begin{bmatrix}
w_{1, 1}&w_{1, 2}&...&w_{1,n}\\
w_{2, 1}&w_{2, 2}&...&w_{2,n}\\
\vdots&\vdots&\ddots&\vdots\\
w_{m, 1}&w_{m, 2}&...&w_{m,n}\\
\end{bmatrix}
$$

The product of the matrices is added to a vector of biases and fed through the activation function in order to obtain the values at the next layer of the network:

$$
A^{(L)}=\sigma(\textbf{W}^{(L-1)}\textbf{A}^{(L-1)}+\textbf{b}^{(L)})
$$
```math
=\sigma(
\begin{bmatrix}
w_{1, 1}&w_{1, 2}&...&w_{1,n}\\
w_{2, 1}&w_{2, 2}&...&w_{2,n}\\
\vdots&\vdots&\ddots&\vdots\\
w_{m, 1}&w_{m, 2}&...&w_{m,n}\\
\end{bmatrix}^{(L-1)}
\begin{bmatrix}
a_{1}\\a_{2}\\\vdots\\a_{n}
\end{bmatrix}^{(L-1)}
+
\begin{bmatrix}
b_1\\b_2\\\vdots\\b_m
\end{bmatrix}^{(L)}
)
```
```math
=\sigma{(
\begin{bmatrix}
w_{1, 1}a_1+w_{1, 2}a_1+...+w_{1, n}a_1+b_1\\
w_{2, 1}a_2+w_{2, 2}a_2+...+w_{2, n}a_2+b_2\\
\vdots\\
w_{m, 1}a_m+w_{m, 2}a_m+...+w_{m, n}a_m+b_m\\
\end{bmatrix}
)}
```
```math
=\begin{bmatrix}
a_1\\a_2\\\vdots\\a_m
\end{bmatrix}^{(L)}
```

This process is iteratively repeated until the final, output layer is reached.

```
input = [...] 
weights = [W1, W2, ..., WL] 
biases = [...]  

for layer in range(1, L+1):

    weighted_sum = dot_product(input, weights[layer-1]) + biases[layer-1]
    output = sigmoid(weighted_sum)
    input = output

final_output = input
```

## Forward Propagation - Numpy

In Numpy, forward propagation across a 3-layer MLP can be represented as follows: (check matrix dimensions!)

```python
import numpy as np

input_size = 5 
output_size = 3  
L = 3

input = np.random.randn(input_size)
weights = [np.random.randn(input_size, output_size) for _ in range(L)]
biases = [np.random.randn(output_size) for _ in range(L)]

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

for layer in range(L):
    weighted_sum = np.dot(input, weights[layer]) + biases[layer]  
    output = sigmoid(weighted_sum)  
    input = output

final_output = input

print("Final Output:", final_output)
```

### Backpropagation - Conceptual


