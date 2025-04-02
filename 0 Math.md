# Math

Here I'll document all the relevant mathematical concepts for each paper.

## Rumelhart Et Al.

*Rumelhart Et Al.* requires a good understanding of basic linear algebra, especially matrix manipulation, as well as gradients and differentiation.

### Linear Algebra

Attempting to implement an MLP without a good understanding of matrices is confusing. So let's understand matrices.

#### Linear Equations

A linear equation is simply an equation where all terms are either constant or the product of a constant and a variable.

The generic form of a linear equation is the following:

$$
a_1 x_1+a_2 x_2+...+a_n x_n=b
$$

The generic equation has a set of solutions, with each value corresponding to a variable

$$
t_1, t_2, ..., t_n
$$

All linear equations with more than one variable have infinitely many solutions. Here's why:

Imagine the following linear equation:
$$ a_1 x_1 + a_2 x_2 = b $$

If we solve for x1, we will obtain the following form:
$$ x_1 = \frac{b_1}{a_1}-\frac{a_2}{a_1}x_2 $$

Clearly, we can choose any value pairs for the two variables as long as the equation holds true. In this case, x2 is a **free variable**, meaning it can be set to any real value.

#### Systems of Linear Equations

A system of equations is simply two or more linear equations where the solution for the system is the union for solutions of individual equations.

A system of equations can either be **consistent**, meaning solutions exist, or **inconsistent**, meaning there are no real solutions to the equation.

If the system of two equations is visualized on a 2D plane, 3 types of solutions can occur:
- Infinitely many solutions: in this case, the lines representing solutions for individual equations overlap.
- One solution: The two lines intersect at a single point.
- No solutions: The two lines are parallel.

For systems of 3 or more equations, infinite solutions can be obtained via parallel solutions on 1 or more planes (i.e. two lines or two 2D slices overlapping).

#### Solving Systems of Linear Equations Using Matrices

A matrix is just a rectangular array of elements with rows and columns.

The following system of equations:
$$
a_{11}x_{11}+a_{12}x_{12}+...+a_{1m}x_{1m}=b_1
$$
$$
a_{21}x_{21}+a_{22}x_{22}+...+a_{2m}x_{2m}=b_1
$$
$$
\vdots
$$
$$
a_{n1}x_{n1}+a_{n2}x_{n2}+...+a_{nm}x_{nm}=b_n
$$

Can be represented by the following matrix:
$$
\begin{bmatrix}
a_{11}&a_{12}&...&a_{1m}&b_{1} \\
a_{21}&a_{22}&...&a_{2m}&b_{2} \\
\:&\:&\vdots\\
a_{n1}&a_{n2}&...&a_{nm}&b_{n} \\
\end{bmatrix}
$$

The above is called an **augmented matrix** - it contains both the coefficients and the constants.

A **coefficient matrix** does not contain constants, only coefficients (wow):
$$
\begin{bmatrix}
a_{11}&a_{12}&...&a_{1m} \\
a_{21}&a_{22}&...&a_{2m} \\
\:&\:&\vdots\\
a_{n1}&a_{n2}&...&a_{nm} \\
\end{bmatrix}
$$

When solving systems of linear equations using matrices, there are 3 different operations that we can perform on matrix rows in order to transform it in a way that's beneficial to us:
- Multiply the row by a constant
- Switch two rows
- Add a constant amount of one row to another row
$$
cR_i,\:R_i\leftrightarrow R_j,\:R_j+cR_i
$$


There are two forms of matrices that we can reduce the augmented matrix to in order for solving:
- **Row-Echelon form** if we are solving using **Gaussian elimination**
- **Reduced Row-Echelon form** if we are solving using **Gauss-Jordan elimination**

#### Row-Echelon Form and Gaussian Elimination

There are three criteria that must be true in order for a matrix to fulfill row-echelon form:
- If any all-zero rows are present, they must be at the bottom of the matrix
- If a row is non-zero, the first value must be a 1 (these are called **leading ones**)
- The leading one in a lower row must be further to the right of a matrix than the leading one in the upper row

The following matrix fulfills row-echelon form:
$$
\begin{bmatrix}
1&-6&9&1&0&0\\
0&0&1&-4&-5&3\\
0&0&0&1&1&2\\
0&0&0&0&0&0
\end{bmatrix}
$$

In order to solve a system of linear equations using Gaussian elimination, we must first find the row-echelon form of the system, then perform **back substitution**, where we essentially convert the matrix back into a system of (a lot easier) equations where one of the variables is simply a constant.

#### Reduced Row-Echelon Form

In order for a matrix to satisfy reduced row-echelon form, it must fulfill all the criteria of row-echelon form as well as an additional criteria:
- If a column contains a leading one, all other values in the column must be zero

The following matrices satisfy reduced row-echelon form:
$$
\begin{bmatrix}
0&0\\
0&0\\
\end{bmatrix}
$$

$$
\begin{bmatrix}
1&0&9\\
0&0&3\\
0&1&2\\
0&0&0\\
\end{bmatrix}
$$

In order to solve a system of linear equations using **Gauss-Jordan elimination**, we get the matrix into reduced row-echelon form and basically have our answers right in the matrix.

#### Solving Using Matrices - No Solutions, Infinite Solutions

In cases where no valid solutions exist, the matrix will look something like the following:

$$
\begin{bmatrix}
1&0&5&-4\\
0&1&-1&-1\\
0&0&0&7
\end{bmatrix}
$$

This basically states that 7=0, so we know no solutions exist to satisfy the equation.

In cases with infinite possible solutions, the matrix typically looks like the following:

$$
\begin{bmatrix}
1&0&5&-4\\
0&1&-1&-1\\
0&0&0&0
\end{bmatrix}
$$

Since 0 is always equal to 0, we know infinite solutions exist.

#### Homogenous System

Maybe this is important (?) but a homogenous system is just a system of equations where all constants are zero.

#### Properties of Matrices

- The size of a matrix is the number of columns times the number of rows:

$$
\begin{bmatrix}
7&8&9\\
1&2&0
\end{bmatrix}_{2*3}
$$

- Matrices are typically abbreviated using the following notation:

$$
\begin{bmatrix}
a_{ij}
\end{bmatrix}\:
\begin{bmatrix}
a_{ij}
\end{bmatrix}_{m*n}\:
A_{m*n}
$$

- A **column vector** or **row vector** is just a 1D matrix consisting of a single column or row.

- Vectors are typically denoted using the following notations:

$$
\textbf{a}=\vec{a}=
\begin{bmatrix}
a_1\\
a_2\\
\vdots\\
a_n
\end{bmatrix}
$$

- The **main diagonal** of a square matrix is its longest diagonal (0 0 to n n)

- A matrix can be **partitioned** into an arbitrary number of sub-matrices:

$$
A=\begin{bmatrix}A_{11}&A_{12}\\A_{21}&A_{22}\end{bmatrix}
$$

- A **zero matrix** is just a matrix full of zeros:

$$
0_{2*4}=\begin{bmatrix}0&0&0&0\\0&0&0&0\end{bmatrix}
$$

- An **identity matrix** is a square matrix where all values are zero except those along the main diagonal, which are 1s:

$$
I_{2}=\begin{bmatrix}1&0\\0&1\end{bmatrix}
$$

#### Matrix Operations and Equality

- Matrices A and B are equal if they are the same shape and all individual terms in A are equal to those in B

$$
\begin{bmatrix}1&3\\2&1\end{bmatrix}=\begin{bmatrix}1&3\\2&1\end{bmatrix}
$$

- Matrices of the same shape can be added and subtracted simply by performing the operation on all individual elements

$$
\begin{bmatrix}1&2\\3&4\end{bmatrix}+\begin{bmatrix}1&3\\2&1\end{bmatrix}=
\begin{bmatrix}2&5\\5&5\end{bmatrix}
$$

- A matrix can be multiplied by a scalar, in which case all individual values are multiplied by the scalar

$$
\begin{bmatrix}1&2\\3&4\end{bmatrix}*2=
\begin{bmatrix}2&4\\6&8\end{bmatrix}
$$

- The **linear composition** of a vector of matrices and a vector of scalars is simply the following:

$$
c_1A_1,\:c_2A_2\:,...,c_nA_n
$$

- The **transpose** of a matrix is the matrix with the rows and columns switched:

$$
A=\begin{bmatrix}1&2\\3&4\\5&6\end{bmatrix}
$$
$$
A^T=\begin{bmatrix}1&3&5\\2&4&6\end{bmatrix}
$$

- A **symmetric matrix** is a matrix that is identical to its transform

- The **trace** of a square matrix is the sum of all elements along the main diagonal:

$$
tr(I_3)=3
$$

#### Matrix Multiplication 

Matrix multiplication is different from scalar multiplication because it's not commutative.

The product of two matrices with the indicated shapes *(m, n)* and *(n, p)* will result in a matrix with the shape *(m, p)*: 

$$
A_{m*n}*B_{n*p}=C_{m*p}
$$

The *ij*th value in a resultant matrix is the sum of the following:

$$
(AB)_{ij}=a_{i1}b_{1j}+a_{i2}b_{2j}+...+a_{ip}b_{pj}
$$

The **dot product** of a horizontal and vertical vector is a special instance of a resultant that can be found in the following way:

$$
\begin{bmatrix}a_1&a_2&...&a_n\end{bmatrix}
\begin{bmatrix}b_1\\b_2\\\vdots\\b_n\end{bmatrix}=
a_1b_1+a_2b_2+...+a_nb_n
$$
