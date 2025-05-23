# Recreating 'Learning Representations by Backpropagation' (Rumelhart et al.)

## Structure
- Backpropagation (math)
- Backpropagation (python)
- Complete training
## Introduction

Rumelhart et al.'s 'Learning Representations by Backpropagation' introduces the concept of a multilayer perceptron network (MLP), a neural network that consists of multiple perceptron layers united by weight and bias matrices in addition to activation functions. We will recreate the architecture introduced in Rumelhart et al. using both raw Numpy and more advanced libraries while covering the underlying mathematical concepts.

## Matrix Multiplication - Conceptual

There are a few things we need to understand about matrices before we attempt to build a neural network similar to the one described in Rumelhart et al.

Two matrices A and B can only be multiplied if the number of **rows** in the first matrix is equal to the number of **columns** in the second matrix. 

In standard mathematical notation, the first value in the shape of a matrix indicates the number of columns, while the second indicates the number of rows in the matrix.

The product of two matrices with the indicated shapes *(m, n)* and *(n, p)* will result in a matrix with the shape *(m, p)*: 

$$
A_{(m,n)}*B_{(n,p)}=C_{(m,p)}
$$

When two matrices are multiplied, each element in the resultant matrix comes from the dot product of a row from the first matrix and a column from the second matrix:

$$
C_{[0, 0]}=A_{[0, 0]}B_{[0, 0]}+A_{[0, 1]}B_{[1, 0]}+A_{[0, 2]}B_{[2, 0]}
$$

In this case, the value-by-value matrix multiplication as a whole resembles the following:

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

### Transpose of a Matrix

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

Now that we understand matrix multiplication at a conceptual level, we will implement matrix operations in Python's Numpy library using its 'array' object, a computationally efficient means of representing matrices programatically.

Using Numpy, a 3 by 2 matrix can be defined as follows:

```python
import numpy as np

A = np.array([[1, 2], [3, 4], [5, 6]])
```

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

 The shape of Numpy arrays can be found using the '.shape' function, and they can be easily reshaped using '.reshape':
 
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

In MLPs similar to the one introduced in Rumelhart et al., the activation function is applied to resultant matrices to ensure that all activation values are normalized within the range (0, 1). The simplest activation function is the sigmoid function:

$$
\sigma(x)=\frac{1}{1+e^{-x}}
$$

The derivative of the sigmoid function (as can be found using the quotient rule and re-expression in terms of itself) is the following:

$$
\sigma'(x)=\sigma(x)(1-\sigma(x))
$$

In Rumelhart et al.'s MLP, the sigmoid function is applied element-wise to vectors representing activation at layers of the network. In Numpy, the operation can be represented in the following way:

```python
v = np.array([1.0, 2.0, -1.0, 0.5])

sigmoid_v = 1 / (1 + np.exp(-v))

sigmoid_derivative_v = sigmoid_v * (1 - sigmoid_v)
```


## Forward Propagation - Conceptual

Forward propagation is fairly simple.

The input values in an MLP can be represented as a vector of n input values:

$$
\textbf{X}=
\begin{bmatrix}
x_1\\
x_2\\
\vdots\\
x_n
\end{bmatrix}
$$

We'll use A(L) to represent the input vector at the Lth layer of the network. For the zeroth layer, it is simply equal to X, the inital input vector:

$$
\textbf{A}^{(0)}=\textbf{X}
$$

The weights for a given layer of the network can be represented by a matrix of the shape (m, n), where m is the number of neurons in the next layer and n is the number of inputs, or neurons in the preceding layer:

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

This process is iteratively repeated until the final, output layer is reached. Using pseudo-code, the overall process can be represented as follows:

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

input = np.random.randn(1, 8)
w = [np.random.randn(8, 64), np.random.randn(64, 64), np.random.randn(64, 8)]
b = [np.random.randn(1, 64), np.random.randn(1, 64), np.random.randn(1, 8)]

def sigmoid(input: np.ndarray):
    return 1 / (1 + np.exp(-input))

for i in range(len(w)):
    input = sigmoid(np.dot(input, w[i]) + b[i])

print(input)
```

## Gradients and Backpropagation - Conceptual

In order to understand backpropagation, the process of gradually adjusting weights and biases in order to optimize the behavior of a neural network, we must first understand the mathematical concept of a **gradient**.

Essentially, a gradient is just a vector that contains the partial derivatives of a function with respect to its variables.

For a function *f* that takes a vector input of *n* real numbers and outputs a real number, the gradient is represented as the following:

```math
\nabla f(\textbf{x})=
\begin{bmatrix}
\frac{\delta f}{\delta x_1}\\
\frac{\delta f}{\delta x_2}\\
\vdots\\
\frac{\delta f}{\delta x_n}
\end{bmatrix}
```

Here, $\nabla f(\textbf{x})$ denotes the gradient vector while $\frac{\delta f}{\delta x_i}$ denotes the partial derivative with respect to each term.

As another example, for the following function:

$$
f(x, y)=x^2+y^2
$$

The gradient is calculated as:

```math
\nabla f(x, y)=
\begin{bmatrix}
\frac{\delta}{\delta x}(x^2 + y^2)\\
\frac{\delta}{\delta y}(x^2 + y^2)
\end{bmatrix}
=
\begin{bmatrix}
2x\\
2y
\end{bmatrix}
```

In order to effectively optimize our network, we need to determine a **loss function** that can be minimized in order to receive the best output.

A common loss function, and the one used in Rumelhart et al., is the **Mean Squared Error** function:

```math
L(\textbf{A}^{(L)}, \textbf{Y})=\frac{1}{2n}\sum_{i=1}^{n}(a_i^{(L)}-y_i)^2
```

The loss, in this case, is simply a scalar value that represents the sum of the squared differences between the expected output value $\textbf{Y}$ of the network and its actual value $\textbf{A}^{(L)}$ after forward propagation. It is defined for a batch of size *n*, and $\frac{1}{2}$ is used for convenience since it is canceled out during differentiation.

Although the loss is a scalar value, it's still mathematicallly possible to take its gradient with respect to a vector or matrix. 

The gradient with respect to each node in the 
