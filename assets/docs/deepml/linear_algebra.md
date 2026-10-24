# Deep-ML Knowledge Base: Linear Algebra

Total verified problems: 33

Source: [Deep-ML (Open-Deep-ML)](https://www.deep-ml.com/)

---

## Problem 1: Matrix-Vector Dot Product
- **Difficulty**: `easy`
- **Category**: Linear Algebra

### Description & Spec
Write a Python function that computes the dot product of a matrix and a vector. The function should return a list representing the resulting vector if the operation is valid, or -1 if the matrix and vector dimensions are incompatible. A matrix (a list of lists) can be dotted with a vector (a list) only if the number of columns in the matrix equals the length of the vector. For example, an n x m matrix requires a vector of length m.

### Mathematical Derivation & Intuition
## Matrix-Vector Dot Product

Consider a matrix $A$ and a vector $v$:

**Matrix $A$ (n x m):**
$$
A = \begin{pmatrix}
a_{11} & a_{12} & \cdots & a_{1m} \\
a_{21} & a_{22} & \cdots & a_{2m} \\
\vdots & \vdots & \ddots & \vdots \\
a_{n1} & a_{n2} & \cdots & a_{nm}
\end{pmatrix}
$$

**Vector $v$ (length m):**
$$
v = \begin{pmatrix}
v_1 \\
v_2 \\
\vdots \\
v_m
\end{pmatrix}
$$

The dot product $A \cdot v$ produces a new vector of length $n$:
$$
A \cdot v = \begin{pmatrix}
a_{11}v_1 + a_{12}v_2 + \cdots + a_{1m}v_m \\
a_{21}v_1 + a_{22}v_2 + \cdots + a_{2m}v_m \\
\vdots \\
a_{n1}v_1 + a_{n2}v_2 + \cdots + a_{nm}v_m
\end{pmatrix}
$$

### Key Requirement:
The number of columns in the matrix ($m$) must equal the length of the vector ($m$). If not, the operation is undefined, and the function should return -1.

### Verified Pure NumPy Solution (From Scratch)
```python
def matrix_dot_vector(a: list[list[int|float]], b: list[int|float]) -> list[int|float]:
    if len(a[0]) != len(b):
        return -1
    result = []
    for row in a:
        total = 0
        for i in range(len(row)):
            total += row[i] * b[i]
        result.append(total)
    return result
```

### Verified Sample Trace Data
```json
{
  "test": "print(matrix_dot_vector([[1, 2, 3], [2, 4, 5], [6, 8, 9]], [1, 2, 3]))",
  "expected_output": "[14, 25, 49]"
}
```

---

## Problem 2: Transpose of a Matrix
- **Difficulty**: `easy`
- **Category**: Linear Algebra

### Description & Spec
Write a Python function that computes the transpose of a given matrix.

### Mathematical Derivation & Intuition
## Transpose of a Matrix

Let's consider a matrix $M$ and its transpose $M^T$:

**Original Matrix $M$:**
$$
M = \begin{pmatrix} 
a & b & c \\ 
d & e & f 
\end{pmatrix}
$$

**Transposed Matrix $M^T$:**
$$
M^T = \begin{pmatrix} 
a & d \\ 
b & e \\ 
c & f 
\end{pmatrix}
$$

### Explanation:
Transposing a matrix involves converting its rows into columns and vice versa. This operation is fundamental in linear algebra for various computations and transformations.

### Verified Pure NumPy Solution (From Scratch)
```python
def transpose_matrix(a: list[list[int|float]]) -> list[list[int|float]]:
    return [list(i) for i in zip(*a)]
```

### Verified Sample Trace Data
```json
{
  "test": "print(transpose_matrix([[1,2],[3,4],[5,6]]))",
  "expected_output": "[[1, 3, 5], [2, 4, 6]]"
}
```

---

## Problem 3: Reshape Matrix
- **Difficulty**: `easy`
- **Category**: Linear Algebra

### Description & Spec
Write a Python function that reshapes a given matrix into a specified shape. if it cant be reshaped return back an empty list `[ ]`

### Mathematical Derivation & Intuition
## Reshaping a Matrix

Matrix reshaping involves changing the shape of a matrix without altering its data. This is essential in many machine learning tasks where the input data needs to be formatted in a specific way.

For example, consider a matrix $M$:

**Original Matrix $M$:**
$$
M = \begin{pmatrix} 
1 & 2 & 3 & 4 \\ 
5 & 6 & 7 & 8 
\end{pmatrix}
$$

**Reshaped Matrix $M'$ with shape (4, 2):**
$$
M' = \begin{pmatrix} 
1 & 2 \\ 
3 & 4 \\ 
5 & 6 \\ 
7 & 8 
\end{pmatrix}
$$

### Important Note:
Ensure the total number of elements remains constant during reshaping.

### Verified Pure NumPy Solution (From Scratch)
```python
import numpy as np

def reshape_matrix(a: list[list[int|float]], new_shape: tuple[int|float]) -> list[list[int|float]]:
    # Not compatible case
    if len(a)*len(a[0]) != new_shape[0]*new_shape[1]:
        return []
    return np.array(a).reshape(new_shape).tolist()
```

### Verified Sample Trace Data
```json
{
  "test": "print(reshape_matrix([[1,2,3,4],[5,6,7,8]], (4, 2)))",
  "expected_output": "[[1, 2], [3, 4], [5, 6], [7, 8]]"
}
```

---

## Problem 4: Calculate Mean by Row or Column
- **Difficulty**: `easy`
- **Category**: Linear Algebra

### Description & Spec
Write a Python function that calculates the mean of a matrix either by row or by column, based on a given mode. The function should take a matrix (list of lists) and a mode ('row' or 'column') as input and return a list of means according to the specified mode.

### Mathematical Derivation & Intuition
## Calculate Mean by Row or Column

Calculating the mean of a matrix by row or column involves averaging the elements across the specified dimension. This operation provides insights into the distribution of values within the dataset, useful for data normalization and scaling.

### Row Mean
The mean of a row is computed by summing all elements in the row and dividing by the number of elements. For row $i$, the mean is:
$$
\mu_{\text{row } i} = \frac{1}{n} \sum_{j=1}^{n} a_{ij}
$$
where $a_{ij}$ is the matrix element in the $i^{\text{th}}$ row and $j^{\text{th}}$ column, and $n$ is the total number of columns.

### Column Mean
Similarly, the mean of a column is found by summing all elements in the column and dividing by the number of elements. For column $j$, the mean is:
$$
\mu_{\text{column } j} = \frac{1}{m} \sum_{i=1}^{m} a_{ij}
$$
where $m$ is the total number of rows.

This mathematical formulation helps in understanding how data is aggregated across different dimensions, a critical step in various data preprocessing techniques.

### Verified Pure NumPy Solution (From Scratch)
```python
def calculate_matrix_mean(matrix: list[list[float]], mode: str) -> list[float]:
    if mode == 'column':
        return [sum(col) / len(matrix) for col in zip(*matrix)]
    elif mode == 'row':
        return [sum(row) / len(row) for row in matrix]
    else:
        raise ValueError("Mode must be 'row' or 'column'")
```

### Verified Sample Trace Data
```json
{
  "test": "print(calculate_matrix_mean([[1, 2, 3], [4, 5, 6], [7, 8, 9]], 'column'))",
  "expected_output": "[4.0, 5.0, 6.0]"
}
```

---

## Problem 5: Scalar Multiplication of a Matrix
- **Difficulty**: `easy`
- **Category**: Linear Algebra

### Description & Spec
Write a Python function that multiplies a matrix by a scalar and returns the result.

### Mathematical Derivation & Intuition
## Scalar Multiplication of a Matrix

When a matrix $A$ is multiplied by a scalar $k$, the operation is defined as multiplying each element of $A$ by $k$.

Given a matrix $A$:
$$
A = \begin{pmatrix} 
a_{11} & a_{12} \\ 
a_{21} & a_{22} 
\end{pmatrix}
$$

And a scalar $k$, the result of the scalar multiplication $kA$ is:
$$
kA = \begin{pmatrix} 
ka_{11} & ka_{12} \\ 
ka_{21} & ka_{22} 
\end{pmatrix}
$$

This operation scales the matrix by $k$ without changing its dimension or the relative proportion of its elements.

### Verified Pure NumPy Solution (From Scratch)
```python
def scalar_multiply(matrix: list[list[int|float]], scalar: int|float) -> list[list[int|float]]:
    return [[element * scalar for element in row] for row in matrix]
```

### Verified Sample Trace Data
```json
{
  "test": "print(scalar_multiply([[1,2],[3,4]], 2))",
  "expected_output": "[[2, 4], [6, 8]]"
}
```

---

## Problem 6: Calculate Eigenvalues of a Matrix
- **Difficulty**: `medium`
- **Category**: Linear Algebra

### Description & Spec
Write a Python function that calculates the eigenvalues of a 2x2 matrix. The function should return a list containing the eigenvalues, sort values from highest to lowest.

### Mathematical Derivation & Intuition
## Calculate Eigenvalues

Eigenvalues of a matrix offer significant insight into the matrix's behavior, particularly in the context of linear transformations and systems of linear equations.

### Definition
For a square matrix $A$, eigenvalues are scalars $\lambda$ that satisfy the equation for some non-zero vector $v$ (eigenvector):
$$
Av = \lambda v
$$

### Calculation for a 2x2 Matrix
The eigenvalues of a 2x2 matrix $A$, given by:
$$
A = \begin{pmatrix} 
a & b \\ 
c & d 
\end{pmatrix}
$$
are determined by solving the characteristic equation:
$$
\det(A - \lambda I) = 0
$$

This simplifies to a quadratic equation:
$$
\lambda^2 - \text{tr}(A) \lambda + \det(A) = 0
$$

Here, the trace of $A$, denoted as $\text{tr}(A)$, is $a + d$, and the determinant of $A$, denoted as $\det(A)$, is $ad - bc$. Solving this equation yields the eigenvalues, $\lambda$.

### Significance
Understanding eigenvalues is essential for analyzing the effects of linear transformations represented by the matrix. They are crucial in various applications, including stability analysis, vibration analysis, and Principal Component Analysis (PCA) in machine learning.

### Verified Pure NumPy Solution (From Scratch)
```python
def calculate_eigenvalues(matrix: list[list[float]]) -> list[float]:
    a, b, c, d = matrix[0][0], matrix[0][1], matrix[1][0], matrix[1][1]
    trace = a + d
    determinant = a * d - b * c
    # Calculate the discriminant of the quadratic equation
    discriminant = trace**2 - 4 * determinant
    # Solve for eigenvalues
    lambda_1 = (trace + discriminant**0.5) / 2
    lambda_2 = (trace - discriminant**0.5) / 2
    return [lambda_1, lambda_2]
```

### Verified Sample Trace Data
```json
{
  "test": "print(calculate_eigenvalues([[2, 1], [1, 2]]))",
  "expected_output": "[3.0, 1.0]"
}
```

---

## Problem 7: Matrix Transformation 
- **Difficulty**: `medium`
- **Category**: Linear Algebra

### Description & Spec
Write a Python function that transforms a given matrix A using the operation  $T^{-1} A S$, where T and S are invertible matrices. The function should first validate if the matrices T and S are invertible, and then perform the transformation. In cases where there is no solution return -1

### Mathematical Derivation & Intuition
### Matrix Transformation using $T^{-1} A S$

Transforming a matrix $A$ using the operation $T^{-1} A S$ involves several steps. This operation changes the basis of matrix $A$ using two matrices $T$ and $S$, with $T$ and $S$ being invertible, to avoid loss of information.
(Multiplying by non-invertible $S$ would result in a loss of dimensions)

### Steps for Transformation

Given matrices $A$, $T$, and $S$:

1. **Check Invertibility**: Verify that $T$ and $S$ are invertible by ensuring their determinants are non-zero; otherwise, return $-1$.
2. **Compute Inverses**: Find the invers of $T$, denoted as $T^{-1}$.
3. **Perform Matrix Multiplication**: Calculate the transformed matrix:

   $$
   A' = T^{-1} A S
   $$

### Example

If:

$$
A =
\begin{pmatrix} 
1 & 2 \\ 
3 & 4 
\end{pmatrix}
$$

$$
T =
\begin{pmatrix} 
2 & 0 \\ 
0 & 2 
\end{pmatrix}
$$

$$
S =
\begin{pmatrix} 
1 & 1 \\ 
0 & 1 
\end{pmatrix}
$$

#### Check Invertibility:

- $\det(T) = 4 \neq 0$
- $\det(S) = 1 \neq 0$

#### Compute Inverses:

$$
T^{-1} =
\begin{pmatrix} 
\frac{1}{2} & 0 \\ 
0 & \frac{1}{2} 
\end{pmatrix}
$$

#### Perform the Transformation:

$$
A' = T^{-1} A S
$$

$$
A' =
\begin{pmatrix} 
\frac{1}{2} & 0 \\ 
0 & \frac{1}{2} 
\end{pmatrix}
\begin{pmatrix} 
1 & 2 \\ 
3 & 4 
\end{pmatrix}
\begin{pmatrix} 
1 & 1 \\ 
0 & 1 
\end{pmatrix}
$$

$$
A' =
\begin{pmatrix} 
0.5 & 1.5 \\ 
1.5 & 3.5 
\end{pmatrix}
$$

### Verified Pure NumPy Solution (From Scratch)
```python
import numpy as np

def transform_matrix(A: list[list[int|float]], T: list[list[int|float]], S: list[list[int|float]]) -> list[list[int|float]]:
    # Convert to numpy arrays for easier manipulation
    A = np.array(A, dtype=float)
    T = np.array(T, dtype=float)
    S = np.array(S, dtype=float)
    
    # Check if the matrices T and S are invertible
    if np.linalg.det(T) == 0 or np.linalg.det(S) == 0:
        # raise ValueError("The matrices T and/or S are not invertible.")
        return -1
    
    # Compute the inverse of T
    T_inv = np.linalg.inv(T)

    # Perform the matrix transformation; use @ for better readability
    transformed_matrix = np.round(T_inv @ A @ S, 3)
    
    return transformed_matrix.tolist()
```

### Verified Sample Trace Data
```json
{
  "test": "print(transform_matrix([[1, 2], [3, 4]], [[2, 0], [0, 2]], [[1, 1], [0, 1]]))",
  "expected_output": "[[0.5, 1.5], [1.5, 3.5]]"
}
```

---

## Problem 8: Calculate 2x2 Matrix Inverse
- **Difficulty**: `medium`
- **Category**: Linear Algebra

### Description & Spec
Write a Python function that calculates the inverse of a 2x2 matrix. Return 'None' if the matrix is not invertible.

### Mathematical Derivation & Intuition
## Calculating the Inverse of a 2x2 Matrix

The inverse of a matrix \( A \) is another matrix, often denoted \( A^{-1} \), such that:
$$
AA^{-1} = A^{-1}A = I
$$
where \( I \) is the identity matrix. For a 2x2 matrix:
$$
A = \begin{pmatrix} 
a & b \\ 
c & d 
\end{pmatrix}
$$

The inverse is given by:
$$
A^{-1} = \frac{1}{\det(A)} \begin{pmatrix} 
d & -b \\ 
-c & a 
\end{pmatrix}
$$

provided that the determinant \( \det(A) = ad - bc \) is non-zero. If \( \det(A) = 0 \), the matrix does not have an inverse.

### Importance
Calculating the inverse of a matrix is essential in various applications, such as solving systems of linear equations, where the inverse is used to find solutions efficiently.

### Verified Pure NumPy Solution (From Scratch)
```python
def inverse_2x2(matrix: list[list[float]]) -> list[list[float]]:
    a, b, c, d = matrix[0][0], matrix[0][1], matrix[1][0], matrix[1][1]
    determinant = a * d - b * c
    if determinant == 0:
        return None
    inverse = [[d/determinant, -b/determinant], [-c/determinant, a/determinant]]
    return inverse
```

### Verified Sample Trace Data
```json
{
  "test": "print(inverse_2x2([[4, 7], [2, 6]]))",
  "expected_output": "[[0.6, -0.7], [-0.2, 0.4]]"
}
```

---

## Problem 9: Matrix times Matrix 
- **Difficulty**: `medium`
- **Category**: Linear Algebra

### Description & Spec
multiply two matrices together (return -1 if shapes of matrix dont aline), i.e. $C = A \cdot B$

### Mathematical Derivation & Intuition
## Matrix Multiplication

Consider two matrices \( A \) and \( B \) to demonstrate their multiplication, defined as follows:

**Matrix \( A \):**
$$
A = \begin{pmatrix} 
a_{11} & a_{12} \\ 
a_{21} & a_{22} 
\end{pmatrix}
$$

**Matrix \( B \):**
$$
B = \begin{pmatrix} 
b_{11} & b_{12} \\ 
b_{21} & b_{22} 
\end{pmatrix}
$$

The multiplication of matrix \( A \) by matrix \( B \) is calculated as:
$$
A \times B = \begin{pmatrix} 
a_{11}b_{11} + a_{12}b_{21} & a_{11}b_{12} + a_{12}b_{22} \\ 
a_{21}b_{11} + a_{22}b_{21} & a_{21}b_{12} + a_{22}b_{22} 
\end{pmatrix}
$$

This operation results in a new matrix where each element is the result of the dot product between the rows of matrix \( A \) and the columns of matrix \( B \).

### Verified Pure NumPy Solution (From Scratch)
```python
def matrixmul(a:list[list[int|float]],
              b:list[list[int|float]])-> list[list[int|float]]:
    if len(a[0]) != len(b):
        return -1
    
    vals = []
    for i in range(len(a)):
        hold = []
        for j in range(len(b[0])):
            val = 0
            for k in range(len(b)):
                val += a[i][k] * b[k][j]
                           
            hold.append(val)
        vals.append(hold)

    return vals
```

### Verified Sample Trace Data
```json
{
  "test": "print(matrixmul([[1,2,3],[2,3,4],[5,6,7]],[[3,2,1],[4,3,2],[5,4,3]]))",
  "expected_output": "[[26, 20, 14], [38, 29, 20], [74, 56, 38]]"
}
```

---

## Problem 11: Solve Linear Equations using Jacobi Method
- **Difficulty**: `medium`
- **Category**: Linear Algebra

### Description & Spec
Write a Python function that uses the Jacobi method to solve a system of linear equations given by Ax = b. The function should iterate n times, rounding each intermediate solution to four decimal places, and return the approximate solution x.

### Mathematical Derivation & Intuition
## Solving Linear Equations Using the Jacobi Method

The Jacobi method is an iterative algorithm used for solving a system of linear equations \( Ax = b \). This method is particularly useful for large systems where direct methods, such as Gaussian elimination, are computationally expensive.


### Algorithm Overview

For a system of equations represented by \( Ax = b \), where \( A \) is a matrix and \( x \) and \( b \) are vectors, the Jacobi method involves the following steps:

1. **Initialization**: Start with an initial guess for \( x \).

2. **Iteration**: For each equation \( i \), update \( x[i] \) using:
   $$
   x[i] = \frac{1}{a_{ii}} \left(b[i] - \sum_{j \neq i} a_{ij} x[j]\right)
   $$
   where \( a_{ii} \) are the diagonal elements of \( A \), and \( a_{ij} \) are the off-diagonal elements.

3. **Convergence**: Repeat the iteration until the changes in \( x \) are below a certain tolerance or until a maximum number of iterations is reached.

This method assumes that all diagonal elements of \( A \) are non-zero and that the matrix is diagonally dominant or properly conditioned for convergence.

### Practical Considerations

- The method may not converge for all matrices.
- Choosing a good initial guess can improve convergence.
- Diagonal dominance of \( A \) ensures the convergence of the Jacobi method.

### Verified Pure NumPy Solution (From Scratch)
```python
import numpy as np

def solve_jacobi(A: np.ndarray, b: np.ndarray, n: int) -> list:
    d_a = np.diag(A)
    nda = A - np.diag(d_a)
    x = np.zeros(len(b))
    x_hold = np.zeros(len(b))
    for _ in range(n):
        for i in range(len(A)):
            x_hold[i] = (1/d_a[i]) * (b[i] - sum(nda[i]*x))
        x = x_hold.copy()
    return np.round(x,4).tolist()
```

### Verified Sample Trace Data
```json
{
  "test": "print(solve_jacobi(np.array([[5, -2, 3], [-3, 9, 1], [2, -1, -7]]), np.array([-1, 2, 3]),2))",
  "expected_output": "[0.146, 0.2032, -0.5175]"
}
```

---

## Problem 12: Singular Value Decomposition (SVD)
- **Difficulty**: `hard`
- **Category**: Linear Algebra

### Description & Spec
Write a Python function called svd_2x2_singular_values(A) that finds an approximate singular value decomposition of a real 2 x 2 matrix using one Jacobi rotation.
Input
A: a NumPy array of shape (2, 2)

Rules
You may use basic NumPy operations (matrix multiplication, transpose, element wise math, etc.).
Do not call numpy.linalg.svd or any other high-level SVD routine.
Stick to a single Jacobi step no iterative refinements.

Return
A tuple (U, Σ, V_T) where
U is a 2 x 2 orthogonal matrix,
Σ is a length 2 NumPy array containing the singular values, and
V_T is the transpose of the right-singular-vector matrix V.

### Mathematical Derivation & Intuition
## Learn Section — Deriving an SVD for a 2 x 2 Matrix

The singular-value decomposition (SVD) rewrites any real matrix $A\!\in\!\mathbb R^{m\times n}$ as

$$
A \;=\; U\,\Sigma\,V^{\!\top},
$$

where

- $U\!\in\!\mathbb R^{m\times m}$ and $V\!\in\!\mathbb R^{n\times n}$ are **orthogonal** ($U^{\!\top}U=I$ and $V^{\!\top}V=I$),
- $\Sigma$ is diagonal with **non-negative** entries $\sigma_1\ge\sigma_2\ge\cdots$ called **singular values**.

When $A$ is $2\times2$ the factorisation can be produced **analytically** with only one Givens (Jacobi) rotation.  
Below we walk through the math used in the `svd_2x2_singular_values` function.

---

### 1. From $A$ to a Symmetric Matrix

Because  
$
A^{\!\top}A \;=\; V\,\Sigma^{\!\top}\Sigma\,V^{\!\top}
$  
is symmetric and positive-semidefinite, its eigenvectors form the right-singular vectors of $A$ and its eigenvalues are the squares of the singular values.

For a $2\times2$ matrix

$$
A \;=\; \begin{bmatrix} a_{11} & a_{12} \\ a_{21} & a_{22} \end{bmatrix},
\quad
A^{\!\top}A \;=\;
\begin{bmatrix}
a_{11}^2 + a_{21}^2 & a_{11}a_{12}+a_{21}a_{22}\\[4pt]
a_{11}a_{12}+a_{21}a_{22} & a_{12}^2 + a_{22}^2
\end{bmatrix}\!,
$$

which we label $B$ in the code (`a_2`).

---

### 2. A Single Jacobi Rotation

To diagonalise $B$ we seek a rotation matrix

$$
R(\theta)=
\begin{bmatrix}
\cos\theta & -\sin\theta\\
\sin\theta & \phantom{-}\cos\theta
\end{bmatrix},
$$

such that $R^{\!\top}BR$ is diagonal.

For $2\times2$ Jacobi iterations the optimal angle is

$$
\theta \;=\;
\begin{cases}
\dfrac{\pi}{4}, & B_{11}=B_{22},\\[8pt]
\dfrac12\,\arctan\!\bigl(\dfrac{2B_{12}}{B_{11}-B_{22}}\bigr), & \text{otherwise}.
\end{cases}
$$

The function computes this `theta`, builds $R$, and updates

$$
D \;=\; R^{\!\top}BR
$$

which now satisfies $D_{12}=D_{21}=0$.

Because we only need *one* rotation to zero the off-diagonal term, the loop runs exactly once.

---

### 3. Extracting Singular Values

The diagonal entries of $D$ are the **eigenvalues** $\lambda_1,\lambda_2$ of $B$.  
Singular values are their square-roots:

$$
\sigma_1 = \sqrt{\lambda_1}, \quad
\sigma_2 = \sqrt{\lambda_2},
\quad
\sigma_1\ge\sigma_2\ge 0.
$$

In code:

```python
s = np.sqrt([d[0,0], d[1,1]])
```

---

### 4. Building $U$

Given $V = R$ (or $V=VR$ after several rotations), the left-singular vectors are

$$
U \;=\; A\,V\,\Sigma^{-1},
$$

where $\Sigma^{-1}=\operatorname{diag}\!\bigl(\tfrac1{\sigma_1},\tfrac1{\sigma_2}\bigr)$.

The multiplication `a @ v @ s_inv` yields an orthogonal $U$.

---

### 5. Putting It All Together

The function finally returns the triple $(U,\; \sigma,\; V^{\!\top})$, giving the exact SVD of $A$:

```python
return (u, s, v.T)
```

---

### Why This Works

* **Eigen-trick** - Diagonalising $A^{\!\top}A$ exposes singular values.  
* **Jacobi rotation** - For $2\times2$ a **single** rotation nulls the off-diagonal term.  
* **Orthonormality** - Both $R$ and $U$ are orthogonal, preserving lengths and angles.

This compact derivation is highly instructive: it shows how SVD generalises *rotation + scaling* in $\mathbb R^2$ and illustrates numerically stable ways to compute the decomposition without heavy linear-algebra libraries.

### Verified Pure NumPy Solution (From Scratch)
```python
import numpy as np 


def svd_2x2_singular_values(A: np.ndarray) -> tuple:
   # stick to lowercase
   a = A

   a_t = np.transpose(a)
   a_2 = a_t @ a

   v = np.eye(2)

   for _ in range(1):
       # Compute rotation angle for a 2x2 matrix
       if a_2[0,0] == a_2[1,1]:
           theta = np.pi/4
       else:
           theta = 0.5 * np.arctan2(2 * a_2[0,1], a_2[0,0] - a_2[1,1])
       
       # Create rotation matrix
       r = np.array(
           [
               [np.cos(theta), -np.sin(theta)],
               [np.sin(theta), np.cos(theta)]
               ]
           )
       
       # apply rotation
       d = np.transpose(r) @ a_2 @ r

       # update a_2
       a_2 = d

       # accumulate v
       v = v @ r

   # sigma is the diagonal elements squared
   s = np.sqrt([d[0,0], d[1,1]])
   s_inv = np.array([[1/s[0], 0], [0, 1/s[1]]])
   
   u = a @ v @ s_inv
   
   return (u, s, v.T)
```

### Verified Sample Trace Data
```json
{
  "test": "print(svd_2x2_singular_values(np.array([[2, 1], [1, 2]])))",
  "expected_output": "(array([[ 0.70710678, -0.70710678],\n       [ 0.70710678,  0.70710678]]), array([3., 1.]), array([[ 0.70710678,  0.70710678],\n       [-0.70710678,  0.70710678]]))"
}
```

---

## Problem 13: Determinant of a 4x4 Matrix using Laplace's Expansion (hard)
- **Difficulty**: `hard`
- **Category**: Linear Algebra

### Description & Spec
Write a Python function that calculates the determinant of a 4x4 matrix using Laplace's Expansion method. The function should take a single argument, a 4x4 matrix represented as a list of lists, and return the determinant of the matrix. The elements of the matrix can be integers or floating-point numbers. Implement the function recursively to handle the computation of determinants for the 3x3 minor matrices.

### Mathematical Derivation & Intuition
## Determinant of a 4x4 Matrix using Laplace's Expansion

Laplace's Expansion, also known as cofactor expansion, is a method to calculate the determinant of a square matrix of any size. For a 4x4 matrix \( A \), this method involves expanding \( A \) into minors and cofactors along a chosen row or column.

Consider a 4x4 matrix \( A \):
$$
A = \begin{pmatrix}
a_{11} & a_{12} & a_{13} & a_{14} \\
a_{21} & a_{22} & a_{23} & a_{24} \\
a_{31} & a_{32} & a_{33} & a_{34} \\
a_{41} & a_{42} & a_{43} & a_{44}
\end{pmatrix}
$$

The determinant of \( A \), \( \det(A) \), can be calculated by selecting any row or column (e.g., the first row) and using the formula that involves the elements of that row (or column), their corresponding cofactors, and the determinants of the 3x3 minor matrices obtained by removing the row and column of each element. This process is recursive, as calculating the determinants of the 3x3 matrices involves further expansions.

The expansion formula for the first row is:
$$
\det(A) = a_{11}C_{11} - a_{12}C_{12} + a_{13}C_{13} - a_{14}C_{14}
$$

### Explanation of Terms
- **Cofactor \( C_{ij} \)**: The cofactor of element \( a_{ij} \) is given by:
  $$
  C_{ij} = (-1)^{i+j} \det(\text{Minor of } a_{ij})
  $$
  where the minor of \( a_{ij} \) is the determinant of the 3x3 matrix obtained by removing the \( i \)th row and \( j \)th column from \( A \).

### Notes
- The choice of row or column for expansion can be based on convenience, often selecting one with the most zeros to simplify calculations.
- The process is recursive, breaking down the determinant calculation into smaller 3x3 determinants until reaching 2x2 determinants, which are simpler to compute.

This method is fundamental in linear algebra and provides a systematic approach for determinant calculation, especially for matrices larger than 3x3.

### Verified Pure NumPy Solution (From Scratch)
```python
def determinant_4x4(matrix: list[list[int|float]]) -> float:
    # Base case: If the matrix is 1x1, return its single element
    if len(matrix) == 1:
        return matrix[0][0]
    # Recursive case: Calculate determinant using Laplace's Expansion
    det = 0
    for c in range(len(matrix)):
        minor = [row[:c] + row[c+1:] for row in matrix[1:]]  # Remove column c
        cofactor = ((-1)**c) * determinant_4x4(minor)  # Compute cofactor
        det += matrix[0][c] * cofactor  # Add to running total
    return det
```

### Verified Sample Trace Data
```json
{
  "test": "print(determinant_4x4([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]))",
  "expected_output": "0"
}
```

---

## Problem 27: Transformation Matrix from Basis B to C
- **Difficulty**: `easy`
- **Category**: Linear Algebra

### Description & Spec
Given basis vectors in two different bases B and C for R^3, write a Python function to compute the transformation matrix P from basis B to C.

### Mathematical Derivation & Intuition
## Understanding Transformation Matrices

A transformation matrix allows us to convert the coordinates of a vector in one basis to coordinates in another basis. For bases \( B \) and \( C \) of a vector space, the transformation matrix \( P \) from \( B \) to \( C \) is calculated as follows:

### Steps to Calculate the Transformation Matrix
1. **Inverse of Basis \( C \)**: First, find the inverse of the matrix representing basis \( C \), denoted \( C^{-1} \).
2. **Matrix Multiplication**: Multiply \( C^{-1} \) by the matrix of basis \( B \). The result is the transformation matrix:
   $$
   P = C^{-1} \cdot B
   $$

This matrix \( P \) can be used to transform any vector coordinates from the \( B \) basis to the \( C \) basis.

### Verified Pure NumPy Solution (From Scratch)
```python
import numpy as np
def transform_basis(B, C):
    C = np.array(C)
    B = np.array(B)
    C_inv = np.linalg.inv(C)
    P = np.dot(C_inv, B)
    return P.tolist()
```

### Verified Sample Trace Data
```json
{
  "test": "print(transform_basis([[1, 0, 0], [0, 1, 0], [0, 0, 1]], [[1, 2.3, 3], [4.4, 25, 6], [7.4, 8, 9]]))",
  "expected_output": "[[-0.6772, -0.0126, 0.2342], [-0.0184, 0.0505, -0.0275], [0.5732, -0.0345, -0.0569]]"
}
```

---

## Problem 28: SVD of a 2x2 Matrix using eigen values & vectors
- **Difficulty**: `hard`
- **Category**: Linear Algebra

### Description & Spec
Given a 2x2 matrix, write a Python function to compute its Singular Value Decomposition (SVD). The function should return the matrices U, S, and V such that A = U * S * V, use the method described in this post https://metamerist.blogspot.com/2006/10/linear-algebra-for-graphics-geeks-svd.html

### Mathematical Derivation & Intuition
## Understanding Singular Value Decomposition (SVD)

Singular Value Decomposition (SVD) is a method in linear algebra for decomposing a matrix into three other matrices. For a given matrix \( A \), SVD is represented as:
$$
A = U \cdot S \cdot V^T
$$

### Step-by-Step Method to Calculate the SVD of a 2x2 Matrix by Hand

1. **Calculate \( A^T A \) and \( A A^T \)**  
   Compute the product of the matrix with its transpose and the transpose of the matrix with itself. These matrices share the same eigenvalues.

2. **Find the Eigenvalues**  
   To find the eigenvalues of a 2x2 matrix, solve the characteristic equation:
   $$
   \det(A - \lambda I) = 0
   $$
   This results in a quadratic equation.

3. **Compute the Singular Values**  
   The singular values, which form the diagonal elements of the matrix \( S \), are the square roots of the eigenvalues.

4. **Calculate the Eigenvectors**  
   For each eigenvalue, solve the equation:
   $$
   (A - \lambda I) \mathbf{x} = 0
   $$
   to find the corresponding eigenvector. Normalize these eigenvectors to form the columns of \( U \) and \( V \).

5. **Form the Matrices \( U \), \( S \), and \( V \)**  
   Combine the singular values and eigenvectors to construct the matrices \( U \), \( S \), and \( V \) such that:
   $$
   A = U \cdot S \cdot V^T
   $$

### Additional Notes
- This method involves solving quadratic equations to find eigenvalues and eigenvectors and normalizing these vectors to unit length.
- **Resources**:  
  - *Linear Algebra for Graphics Geeks (SVD-IX) by METAMERIST* [Google Search]  
  - *Robust Algorithm for 2×2 SVD*

This explanation provides a clear and structured overview of how to calculate the SVD of a 2x2 matrix by hand.

### Verified Pure NumPy Solution (From Scratch)
```python
import numpy as np

def svd_2x2(A: np.ndarray) -> tuple:
    y1, x1 = (A[1, 0] + A[0, 1]), (A[0, 0] - A[1, 1])
    y2, x2 = (A[1, 0] - A[0, 1]), (A[0, 0] + A[1, 1])

    h1 = np.sqrt(y1**2 + x1**2)
    h2 = np.sqrt(y2**2 + x2**2)

    t1 = x1 / h1
    t2 = x2 / h2

    cc = np.sqrt((1.0 + t1) * (1.0 + t2))
    ss = np.sqrt((1.0 - t1) * (1.0 - t2))
    cs = np.sqrt((1.0 + t1) * (1.0 - t2))
    sc = np.sqrt((1.0 - t1) * (1.0 + t2))

    c1, s1 = (cc - ss) / 2.0, (sc + cs) / 2.0
    U = np.array([[-c1, -s1], [-s1, c1]])

    s = np.array([(h1 + h2) / 2.0, abs(h1 - h2) / 2.0])

    V = np.diag(1.0 / s) @ U.T @ A

    return U, s, V
```

### Verified Sample Trace Data
```json
{
  "test": "U,s,V = svd_2x2(np.array([[-10, 8], [10, -1]]))\nresult = U @ np.diag(s) @ V\nprint(result)",
  "expected_output": "[[-10, 8], [10, -1]]"
}
```

---

## Problem 35: Convert Vector to Diagonal Matrix
- **Difficulty**: `easy`
- **Category**: Linear Algebra

### Description & Spec
Write a Python function to convert a 1D numpy array into a diagonal matrix. The function should take in a 1D numpy array x and return a 2D numpy array representing the diagonal matrix.

### Mathematical Derivation & Intuition
## Understanding Diagonal Matrices

A diagonal matrix is a square matrix in which the entries outside the main diagonal are all zero. The main diagonal is the set of entries extending from the top left to the bottom right of the matrix.

### Problem Overview
In this problem, you will write a function to convert a 1D numpy array (vector) into a diagonal matrix. The resulting matrix will have the elements of the input vector on its main diagonal, with zeros elsewhere.

### Mathematical Representation
Given a vector $\mathbf{x} = [x_1, x_2, \ldots, x_n]$, the corresponding diagonal matrix $\mathbf{D}$ is:
$$
\mathbf{D} = \begin{bmatrix}
x_1 & 0 & 0 & \cdots & 0 \\
0 & x_2 & 0 & \cdots & 0 \\
0 & 0 & x_3 & \cdots & 0 \\
\vdots & \vdots & \vdots & \ddots & \vdots \\
0 & 0 & 0 & \cdots & x_n
\end{bmatrix}
$$

### Importance
Diagonal matrices are important in various mathematical and scientific computations due to their simple structure and useful properties.

### Verified Pure NumPy Solution (From Scratch)
```python
import numpy as np

def make_diagonal(x):
    identity_matrix = np.identity(np.size(x))
    return (identity_matrix*x)
```

### Verified Sample Trace Data
```json
{
  "test": "print(make_diagonal(np.array([1, 2, 3])))",
  "expected_output": "[[1., 0., 0.], [0., 2., 0.], [0., 0., 3.]]"
}
```

---

## Problem 37: Calculate Correlation Matrix
- **Difficulty**: `medium`
- **Category**: Linear Algebra

### Description & Spec
Write a Python function to calculate the correlation matrix for a given dataset. The function should take in a 2D numpy array X and an optional 2D numpy array Y. If Y is not provided, the function should calculate the correlation matrix of X with itself. It should return the correlation matrix as a 2D numpy array.

### Mathematical Derivation & Intuition
## Understanding Correlation Matrix

A correlation matrix is a table showing the correlation coefficients between variables. Each cell in the table shows the correlation between two variables, with values ranging from -1 to 1. These values indicate the strength and direction of the linear relationship between the variables.

### Mathematical Definition
The correlation coefficient between two variables \( X \) and \( Y \) is given by:
$$
\text{corr}(X, Y) = \frac{\text{cov}(X, Y)}{\sigma_X \sigma_Y}
$$

#### Where:
- $$ \text{cov}(X, Y) $$ is the covariance between \( X \) and \( Y \).
- $$ \sigma_X $$ and $$ \sigma_Y $$ are the standard deviations of \( X \) and \( Y \), respectively.

### Problem Overview
In this problem, you will write a function to calculate the correlation matrix for a given dataset. The function will take in a 2D numpy array \( X \) and an optional 2D numpy array \( Y \). If \( Y \) is not provided, the function will calculate the correlation matrix of \( X \) with itself.

### Verified Pure NumPy Solution (From Scratch)
```python
import numpy as np

def calculate_correlation_matrix(X, Y=None):
    # Helper function to calculate standard deviation
    def calculate_std_dev(A):
        return np.sqrt(np.mean((A - A.mean(0))**2, axis=0))
    
    if Y is None:
        Y = X
    n_samples = np.shape(X)[0]
    # Calculate the covariance matrix
    covariance = (1 / n_samples) * (X - X.mean(0)).T.dot(Y - Y.mean(0))
    # Calculate the standard deviations
    std_dev_X = np.expand_dims(calculate_std_dev(X), 1)
    std_dev_y = np.expand_dims(calculate_std_dev(Y), 1)
    # Calculate the correlation matrix
    correlation_matrix = np.divide(covariance, std_dev_X.dot(std_dev_y.T))

    return np.array(correlation_matrix, dtype=float)
```

### Verified Sample Trace Data
```json
{
  "test": "print(calculate_correlation_matrix(np.array([[1, 2], [3, 4], [5, 6]])))",
  "expected_output": "[[1., 1.], [1., 1.]]"
}
```

---

## Problem 48: Implement Reduced Row Echelon Form (RREF) Function
- **Difficulty**: `medium`
- **Category**: Linear Algebra

### Description & Spec
In this problem, your task is to implement a function that converts a given matrix into its Reduced Row Echelon Form (RREF). The RREF of a matrix is a special form where each leading entry in a row is 1, and all other elements in the column containing the leading 1 are zeros, except for the leading 1 itself.

However, there are some additional details to keep in mind:

- Diagonal entries can be 0 if the matrix is reducible (i.e., the row corresponding to that position can be eliminated entirely).
- Some rows may consist entirely of zeros.
- If a column contains a pivot (a leading 1), all other entries in that column should be zero.

Your task is to implement the RREF algorithm, which must handle these cases and convert any given matrix into its RREF.

### Mathematical Derivation & Intuition
## Understanding the RREF Algorithm

The Reduced Row Echelon Form (RREF) of a matrix is a specific form achieved through a sequence of elementary row operations. This algorithm will convert any matrix into its RREF, which is useful for solving linear equations and understanding the properties of the matrix.

Here’s a step-by-step guide to implementing the RREF algorithm:

1. **Start with the leftmost column**:  
   Set the initial leading column to the first column of the matrix. Move this "lead" to the right as you progress through the algorithm.

2. **Select the pivot row**:  
   Identify the first non-zero entry in the current leading column. This entry is the pivot. If necessary, swap rows to bring the pivot into position to avoid having a zero in the pivot position.

3. **Scale the pivot row**:  
   Divide the entire pivot row by the pivot value to make the leading entry equal to 1.
   $$
   \text{Row}_r = \frac{\text{Row}_r}{\text{pivot}}
   $$
   For example, if the pivot is 3, divide the entire row by 3 to make the leading entry 1.

4. **Eliminate above and below the pivot**:  
   Subtract multiples of the pivot row from all the other rows to create zeros in the rest of the pivot column. This ensures the pivot is the only non-zero entry in its column.
   $$
   \text{Row}_i = \text{Row}_i - (\text{Row}_r \times \text{lead coefficient})
   $$
   Repeat this step for each row $ i $ where $ i \neq r $, ensuring all entries above and below the pivot are zero.

5. **Move to the next column**:  
   Move the lead one column to the right and repeat the process from step 2. Continue until there are no more columns to process or the remaining submatrix is all zeros.

By following these steps, the matrix will be converted into its Reduced Row Echelon Form, where each leading entry is 1, and all other entries in the leading columns are zero.

### Verified Pure NumPy Solution (From Scratch)
```python
import numpy as np

def rref(matrix):
    # Convert to float for division operations
    A = matrix.astype(np.float32)
    n, m = A.shape
    row = 0  # Current row index for pivot placement
    
    # Iterate over columns, up to the number of columns m
    for col in range(m):
        if row >= n:  # No more rows to process
            break
        
        # Find a row with a non-zero entry in the current column
        nonzero_rel_id = np.nonzero(A[row:, col])[0]
        if len(nonzero_rel_id) == 0:  # No pivot in this column
            continue
        
        # Swap the current row with the row containing the non-zero entry
        k = nonzero_rel_id[0] + row
        A[[row, k]] = A[[k, row]]
        
        # Normalize the pivot row to make the pivot 1
        A[row] = A[row] / A[row, col]
        
        # Eliminate all other entries in this column
        for j in range(n):
            if j != row:
                A[j] -= A[j, col] * A[row]
        
        row += 1  # Move to the next row for the next pivot
    
    return A
```

### Verified Sample Trace Data
```json
{
  "test": "import numpy as np\n\nmatrix = np.array([\n    [1, 2, -1, -4],\n    [2, 3, -1, -11],\n    [-2, 0, -3, 22]\n])\n\noutput = rref(matrix)\nprint(output)",
  "expected_output": "[[ 1.,  0.,  0., -8.], [ 0.,  1.,  0.,  1.], [-0., -0.,  1., -2.]]"
}
```

---

## Problem 55: 2D Translation Matrix Implementation
- **Difficulty**: `medium`
- **Category**: Linear Algebra

### Description & Spec
## Task: Implement a 2D Translation Matrix

Your task is to implement a function that applies a 2D translation matrix to a set of points. A translation matrix is used to move points in 2D space by a specified distance in the x and y directions.

Write a function `translate_object(points, tx, ty)` where `points` is a list of [x, y] coordinates and `tx` and `ty` are the translation distances in the x and y directions, respectively.

The function should return a new list of points after applying the translation matrix.

### Mathematical Derivation & Intuition
## 2D Translation Matrix Implementation

The translation matrix is a fundamental concept in linear algebra and computer graphics, used to move points or objects in a 2D space.

### Concept Overview

For a 2D translation, we use a 3x3 matrix to move a point \( (x, y) \) by \( x_t \) units in the x-direction and \( y_t \) units in the y-direction.

Any point \( P \) in 2D Cartesian space with coordinates \( (x, y) \) can be represented in homogeneous coordinates as \( (x, y, 1) \):

$$
P_{\text{Cartesian}} = (x, y) \rightarrow P_{\text{Homogeneous}} = (x, y, 1)
$$

More generally, any scalar multiple of \( (x, y, 1) \) represents the same point in 2D space. Thus, \( (kx, ky, k) \) for any non-zero \( k \) also represents the same point \( (x, y) \).

The addition of this third coordinate allows us to represent translation as a linear transformation.

### Translation Matrix

The translation matrix \( T \) is defined as:

$$
T = \begin{bmatrix}
1 & 0 & x_t \\
0 & 1 & y_t \\
0 & 0 & 1
\end{bmatrix}
$$

### Applying the Translation

To translate a point \( (x, y) \), we first convert it to homogeneous coordinates: \( (x, y, 1) \). The transformation is then performed using matrix multiplication:

$$
\begin{bmatrix}
1 & 0 & x_t \\
0 & 1 & y_t \\
0 & 0 & 1
\end{bmatrix}
\begin{bmatrix}
x \\
y \\
1
\end{bmatrix}
=
\begin{bmatrix}
x + x_t \\
y + y_t \\
1
\end{bmatrix}
$$

### Explanation of Parameters

1. **Original Point**: \( (x, y) \)  
2. **Translation in x-direction**: \( x_t \)  
3. **Translation in y-direction**: \( y_t \)  
4. **Translated Point**: \( (x + x_t, y + y_t) \)

This process effectively shifts the original point \( (x, y) \) by \( x_t \) and \( y_t \), resulting in the new coordinates \( (x + x_t, y + y_t) \).

### Verified Pure NumPy Solution (From Scratch)
```python
import numpy as np

def translate_object(points, tx, ty):
    translation_matrix = np.array([
        [1, 0, tx],
        [0, 1, ty],
        [0, 0, 1]
    ])
    
    homogeneous_points = np.hstack([np.array(points), np.ones((len(points), 1))])
    
    translated_points = np.dot(homogeneous_points, translation_matrix.T)
    
    return translated_points[:, :2].tolist()
```

### Verified Sample Trace Data
```json
{
  "test": "import numpy as np\n\ntriangle = [[0, 0], [1, 0], [0.5, 1]]\ntx, ty = 2, 3\nprint(translate_object(triangle, tx, ty))",
  "expected_output": "[[2.0, 3.0], [3.0, 3.0], [2.5, 4.0]]"
}
```

---

## Problem 57: Gauss-Seidel Method for Solving Linear Systems
- **Difficulty**: `medium`
- **Category**: Linear Algebra

### Description & Spec
## Task: Implement the Gauss-Seidel Method

Your task is to implement the Gauss-Seidel method, an iterative technique for solving a system of linear equations \(Ax = b\).

The function should iteratively update the solution vector \(x\) by using the most recent values available during the iteration process.

Write a function `gauss_seidel(A, b, n, x_ini=None)` where:

- `A` is a square matrix of coefficients,
- `b` is the right-hand side vector,
- `n` is the number of iterations,
- `x_ini` is an optional initial guess for \(x\) (if not provided, assume a vector of zeros).

The function should return the approximated solution vector \(x\) after performing the specified number of iterations.

### Mathematical Derivation & Intuition
## Understanding the Gauss-Seidel Method

The Gauss-Seidel method is a technique for solving linear systems of equations $Ax = b$. Unlike fixed-point Jacobi, Gauss-Seidel uses previously computed results as soon as they are available. This increases convergence, resulting in fewer iterations, but it is not as easily parallelizable as fixed-point Jacobi.

### Mathematical Formulation

1. **Initialization**: Start with an initial guess for $x$.

2. **Iteration**: For each equation $i$, update $x[i]$ using:

$$
x_{i}^{(k+1)} = \frac{1}{a_{ii}} \left( b[i] - \sum_{j < i} a_{ij} x_{j}^{(k+1)} - \sum_{j > i} a_{ij} x_{j}^{(k)} \right)
$$

where $a_{ii}$ represents the diagonal elements of $A$, and $a_{ij}$ represents the off-diagonal elements.

3. **Convergence**: Repeat the iteration until the changes in $x$ are below a set tolerance or until a maximum number of iterations is reached.

### Matrix Form

The Gauss-Seidel method can also be expressed in matrix form using the diagonal matrix $D$, lower triangle $L$, and upper triangle $U$:

$$
x^{(k+1)} = D^{-1} \left( b - Lx^{(k+1)} - Ux^{(k)} \right)
$$

### Example Calculation

Let’s solve the system of equations:

$$
3x_1 + x_2 = 5 \quad x_1 + 2x_2 = 5
$$

1. Initialize $ x_1^{(0)} = 0 $ and $ x_2^{(0)} = 0 $.

2. **First iteration**:

For $ x_1^{(1)} $:

$$
x_1^{(1)} = \frac{1}{3} \left( 5 - 1 \cdot x_2^{(0)} \right) = \frac{5}{3} \approx 1.6667
$$

For $ x_2^{(1)} $:

$$
x_2^{(1)} = \frac{1}{2} \left( 5 - 1 \cdot x_1^{(1)} \right) = \frac{1}{2} \left( 5 - 1.6667 \right) \approx 1.6667
$$

After the first iteration, the values are $ x_1^{(1)} = 1.6667 $ and $ x_2^{(1)} = 1.6667 $.

Continue iterating until the results converge to a desired tolerance.

### Applications

The Gauss-Seidel method and other iterative solvers are commonly used in data science, computational fluid dynamics, and 3D graphics.

### Verified Pure NumPy Solution (From Scratch)
```python
import numpy as np

def gauss_seidel_it(A, b, x):
    rows, cols = A.shape
    for i in range(rows):
        x_new = b[i]
        for j in range(cols):
            if i != j:
                x_new -= A[i, j] * x[j]
        x[i] = x_new / A[i, i]
    return x

def gauss_seidel(A, b, n, x_ini=None):
    x = x_ini or np.zeros_like(b)
    for _ in range(n):
        x = gauss_seidel_it(A, b, x)
    return x
```

### Verified Sample Trace Data
```json
{
  "test": "import numpy as np\n\nA = np.array([[4, 1, 2], [3, 5, 1], [1, 1, 3]], dtype=float)\nb = np.array([4, 7, 3], dtype=float)\nn = 5\nprint(gauss_seidel(A, b, n))",
  "expected_output": "[0.5008, 0.99968, 0.49984]"
}
```

---

## Problem 58: Gaussian Elimination for Solving Linear Systems
- **Difficulty**: `medium`
- **Category**: Linear Algebra

### Description & Spec
## Task: Implement the Gaussian Elimination Method

Your task is to implement the Gaussian Elimination method, which transforms a system of linear equations into an upper triangular matrix. This method can then be used to solve for the variables using backward substitution.

Write a function `gaussian_elimination(A, b)` that performs Gaussian Elimination with partial pivoting to solve the system \(Ax = b\).

The function should return the solution vector \(x\).

### Mathematical Derivation & Intuition
## Understanding Gaussian Elimination

Gaussian Elimination is used to replace matrix coefficients with a row-echelon form matrix, which can be more easily solved via backwards substitution.

### Row-Echelon Form Criteria

- **Non-zero rows** are above any rows of all zeros.
- The **leading entry** of each non-zero row is to the right of the leading entry of the previous row.
- The **leading entry** in any non-zero row is 1, and all entries below it in the same column are zeros.

### Augmented Matrix

For a linear system $Ax = b$, an augmented matrix is a way of displaying all the numerical information in a linear system in a single matrix. This combines the coefficient matrix $A$ and vector source $b$ as follows:

$$
\begin{pmatrix} 
a_{11} & a_{21} & a_{31} & b_1\\ 
a_{12} & a_{22} & a_{32} & b_2\\ 
a_{31} & a_{32} & a_{33} & b_3 
\end{pmatrix}
$$

### Partial Pivoting

In linear algebra, diagonal elements of a matrix are referred to as the "pivot". To solve a linear system, the diagonal is used as a divisor for other elements within the matrix. This means that Gaussian Elimination will fail if there is a zero pivot.

In this case, pivoting is used to interchange rows, ensuring a non-zero pivot. Specifically, **partial pivoting** looks at all other rows in the current column to find the row with the highest absolute value. This row is then interchanged with the current row. This not only increases the numerical stability of the solution, but also reduces round-off errors caused by dividing by small entries.

### Gaussian Elimination Mathematical Formulation

**Gaussian Elimination:**

- For $k = 1$ to $ \text{number of rows} - 1$:
  - Apply partial pivoting to the current row.
  - For $i = k + 1$ to $ \text{number of rows}$:
    - $ m_{ik} = \frac{a_{ik}}{a_{kk}} $
    - For $j = k$ to $ \text{number of columns}$:
      - $ a_{ij} = a_{ij} - m_{ik} \times a_{kj} $
    - $ b_i = b_i - m_{ik} \times b_k $

**Backwards Substitution:**

- For $k = \text{number of rows}$ to $1$:
  - For $i = \text{number of columns} - 1$ to $1$:
    - $ b_k = b_k - a_{ki} \times b_i $
  - $ b_k = \frac{b_k}{a_{kk}} $

### Example Calculation

Let’s solve the system of equations:

$$
A = \begin{pmatrix} 
2 & 8 & 4\\ 
5 & 5 & 1 \\ 
4 & 10 & -1 
\end{pmatrix} 
\quad \text{and} \quad 
b = \begin{pmatrix} 
2 \\ 5 \\ 1 
\end{pmatrix} 
$$

1. Apply **partial pivoting** to increase the magnitude of the pivot. For $A_{11}$, calculate the factor for the elimination of $A_{12}$: 

$$ 
m_{12} = \frac{A_{12}}{A_{11}} = \frac{2}{5} = 0.4 
$$

2. Apply this scaling to row 1 and subtract this from row 2, eliminating $A_{12}$:

$$
A = \begin{pmatrix} 
5 & 5 & 1 \\ 
0 & 6 & 3.6 \\ 
4 & 10 & -1 
\end{pmatrix} 
\quad \text{and} \quad 
b = \begin{pmatrix} 
5 \\ 0 \\ 1 
\end{pmatrix} 
$$

After the full **Gaussian Elimination** process has been applied to $A$ and $b$, we get the following:

$$
A = \begin{pmatrix} 
5 & 5 & 1\\ 
0 & 6 & 3.6 \\ 
0 & 0 & -5.4 
\end{pmatrix} 
\quad \text{and} \quad 
b = \begin{pmatrix} 
5 \\ 0 \\ 3 
\end{pmatrix} 
$$

To calculate $x$, we apply **backward substitution** by substituting in the currently solved values and dividing by the pivot. This gives the following for the first iteration:

$$ 
x_3 = \frac{b_3}{A_{33}} = \frac{3}{-5.4} = -0.56 
$$

This process can be repeated iteratively for all rows to solve the linear system, substituting in the solved values for the rows below.

### Applications

Gaussian Elimination and linear solvers have a wide range of real-world applications, including their use in:

- Machine learning
- Computational fluid dynamics
- 3D graphics

### Verified Pure NumPy Solution (From Scratch)
```python
import numpy as np

def partial_pivoting(A_aug, row_num, col_num):
    rows, cols = A_aug.shape
    max_row = row_num
    max_val = abs(A_aug[row_num, col_num])
    for i in range(row_num, rows):
        current_val = abs(A_aug[i, col_num])
        if current_val > max_val:
            max_val = current_val
            max_row = i
    if max_row != row_num:
        A_aug[[row_num, max_row]] = A_aug[[max_row, row_num]]
    return A_aug

def gaussian_elimination(A, b):
    rows, cols = A.shape
    A_aug = np.hstack((A, b.reshape(-1, 1)))

    for i in range(rows-1):
        A_aug = partial_pivoting(A_aug, i, i)
        for j in range(i+1, rows):
            A_aug[j, i:] -= (A_aug[j, i] / A_aug[i, i]) * A_aug[i, i:]

    x = np.zeros_like(b, dtype=float)
    for i in range(rows-1, -1, -1):
        x[i] = (A_aug[i, -1] - np.dot(A_aug[i, i+1:cols], x[i+1:])) / A_aug[i, i]
    return x
```

### Verified Sample Trace Data
```json
{
  "test": "import numpy as np\n\nA = np.array([[2,8,4], [2,5,1], [4,10,-1]], dtype=float)\nb = np.array([2,5,1], dtype=float)\nprint(gaussian_elimination(A, b))",
  "expected_output": "[11.0, -4.0, 3.0]"
}
```

---

## Problem 63: Implement the Conjugate Gradient Method for Solving Linear Systems
- **Difficulty**: `hard`
- **Category**: Linear Algebra

### Description & Spec
## Task: Implement the Conjugate Gradient Method for Solving Linear Systems

Your task is to implement the Conjugate Gradient (CG) method, an efficient iterative algorithm for solving large, sparse, symmetric, positive-definite linear systems. Given a matrix `A` and a vector `b`, the algorithm will solve for `x` in the system \( Ax = b \).

Write a function `conjugate_gradient(A, b, n, x0=None, tol=1e-8)` that performs the Conjugate Gradient method as follows:

- `A`: A symmetric, positive-definite matrix representing the linear system.
- `b`: The vector on the right side of the equation.
- `n`: Maximum number of iterations.
- `x0`: Initial guess for the solution vector.
- `tol`: Tolerance for stopping criteria.

The function should return the solution vector `x`.

### Mathematical Derivation & Intuition
## Understanding The Conjugate Gradient Method

The Conjugate Gradient (CG) method is an iterative algorithm used to solve large systems of linear equations, particularly those that are symmetric and positive-definite.

### Concepts

The CG gradient method is often applied to the quadratic form of a linear system, $Ax = b$:

$$
f(x) = \frac{1}{2} x^T A x - b^T x
$$

The quadratic form is used due to its differential reducing to the following for a symmetric $A$. Therefore, $x$ satisfies $Ax = b$ at the optimum:

$$
f'(x) = Ax - b
$$

The conjugate gradient method uses search directions that are conjugate to all the previous search directions. This is satisfied when search directions are $A$-orthogonal, i.e.

$$
p_i^T A p_j = 0 \quad \text{for } i \neq j
$$

This results in a more efficient algorithm, as it ensures that the algorithm gathers all information in a search direction at once and then doesn't need to search in that direction again. This is opposed to steepest descent, where the algorithm steps a bit in one direction and then may search in that direction again later.

### Algorithm Steps

1) **Initialization**:
   - $ x_0 $: Initial guess for the variable vector.
   - $ r_0 = b - A x_0 $: Initial residual vector.
   - $ p_0 = r_0 $: Initial search direction.

2) **Iteration $k$**:
   - $ \alpha_k = \frac{r_k^T r_k}{p_k^T A p_k} $: Step size.
   - $ x_{k+1} = x_k + \alpha_k p_k $: Update solution.
   - $ r_{k+1} = r_k - \alpha_k A p_k $: Update residual.
   - Check convergence: $ \| r_{k+1} \| < \text{tolerance} $.
   - $ \beta_k = \frac{r_{k+1}^T r_{k+1}}{r_k^T r_k} $: New direction scaling. This ensures search directions are $A$-orthogonal.
   - $ p_{k+1} = r_{k+1} + \beta_k p_k $: Update search direction.

3) **Termination**:
   - Stop when $ \| r_{k+1} \| < \text{tolerance} $ or after a set number of iterations.

### Example Calculation

Let's solve the system of equations:

$$
4x_1 + x_2 = 6 \quad x_1 + 3x_2 = 6
$$

1) **Initialize**: $ x_0 = [0, 0]^T $, $ r_0 = b - A x_0 = [6, 6]^T $, and $ p_0 = r_0 = [6, 6]^T $.

2) **First iteration**:
   - Compute $ \alpha_0 $:

   $$
   \alpha_0 = \frac{r_0^T r_0}{p_0^T A p_0} = \frac{72}{324} = 0.2222
   $$

   - Update solution $ x_1 $:

   $$
   x_1 = x_0 + \alpha_0 p_0 = [0, 0]^T + 0.2222 \cdot [6, 6]^T = [1.3333, 1.3333]^T
   $$

   - Update residual $ r_1 $:

   $$
   r_1 = r_0 - \alpha_0 A p_0 = [6, 6]^T - 0.2222 \cdot \begin{bmatrix} 4 & 1 \\ 1 & 3 \end{bmatrix} \cdot [6, 6]^T = [6.67, 5.33]^T
   $$

   - Compute $ \beta_0 $:

   $$
   \beta_0 = \frac{r_1^T r_1}{r_0^T r_0} = \frac{6.67^2 + 5.33^2}{6^2 + 6^2} \approx 0.99
   $$

   - Update search direction $ p_1 $:

   $$
   p_1 = r_1 + \beta_0 p_0 = [6.67, 5.33]^T + 0.99 \cdot [6, 6]^T = [12.60, 11.26]^T
   $$

3) **Second iteration**:
   - Compute $ \alpha_1 $, $ x_2 $, $ r_2 $, and repeat until convergence.

### Applications

The conjugate gradient method is often used because it's more efficient than other iterative solvers, such as steepest descent, and direct solvers, such as Gaussian Elimination. Iterative linear solvers are commonly used in:

- Optimization
- Machine Learning
- Computational Fluid Dynamics

### Verified Pure NumPy Solution (From Scratch)
```python
import numpy as np

def conjugate_gradient(A: np.array, b: np.array, n: int, x0: np.array=None, tol=1e-8) -> np.array:

    # calculate initial residual vector
    x = np.zeros_like(b)
    r = residual(A, b, x) # residual vector
    rPlus1 = r
    p = r # search direction vector

    for i in range(n):

        # line search step value - this minimizes the error along the current search direction
        alp = alpha(A, r, p)

        # new x and r based on current p (the search direction vector)
        x = x + alp * p
        rPlus1 = r - alp * (A@p)

        # calculate beta - this ensures that all vectors are A-orthogonal to each other
        bet = beta(r, rPlus1)

        # update x and r
        # using a othogonal search direction ensures we get all the information we need in more direction and then don't have to search in that direction again
        p = rPlus1 + bet * p

        # update residual vector
        r = rPlus1

        # break if less than tolerance
        if np.linalg.norm(residual(A,b,x)) < tol:
            break

    return x

def residual(A: np.array, b: np.array, x: np.array) -> np.array:
    # calculate linear system residuals
    return b - A @ x

def alpha(A: np.array, r: np.array, p: np.array) -> float:

    # calculate step size
    alpha_num = np.dot(r, r)
    alpha_den = np.dot(p @ A, p)

    return alpha_num/alpha_den

def beta(r: np.array, r_plus1: np.array) -> float:

    # calculate direction scaling
    beta_num = np.dot(r_plus1, r_plus1)
    beta_den = np.dot(r, r)

    return beta_num/beta_den
```

### Verified Sample Trace Data
```json
{
  "test": "import numpy as np\n\nA = np.array([[4, 1], [1, 3]])\nb = np.array([1, 2])\nn = 5\nprint(conjugate_gradient(A, b, n))",
  "expected_output": "[0.09090909, 0.63636364]"
}
```

---

## Problem 65: Implement Compressed Row Sparse Matrix (CSR) Format Conversion
- **Difficulty**: `easy`
- **Category**: Linear Algebra

### Description & Spec
## Task: Convert a Dense Matrix to Compressed Row Sparse (CSR) Format

Your task is to implement a function that converts a given dense matrix into the Compressed Row Sparse (CSR) format, an efficient storage representation for sparse matrices. The CSR format only stores non-zero elements and their positions, significantly reducing memory usage for matrices with a large number of zeros.

Write a function `compressed_row_sparse_matrix(dense_matrix)` that takes a 2D list `dense_matrix` as input and returns a tuple containing three lists:

- **Values array**: List of all non-zero elements in row-major order.
- **Column indices array**: Column index for each non-zero element in the values array.
- **Row pointer array**: Cumulative number of non-zero elements per row, indicating the start of each row in the values array.

### Mathematical Derivation & Intuition
## Understanding the Compressed Row Sparse Matrix Format

The Compressed Row Sparse (CSR) format is a data-efficient representation of sparse matrices, where most of the elements are zero. This format is particularly useful in large-scale scientific computing and machine learning applications, where memory efficiency is critical.

### Concepts

A sparse matrix is a matrix that contains a large number of zero elements. Storing such matrices in their full form can be inefficient, both in terms of memory and computational resources. The CSR format addresses this problem by storing only the non-zero elements and their positions in the matrix. In the CSR format, a matrix is represented by three one-dimensional arrays:

- **Values array**: Contains all the non-zero elements of the matrix, stored row by row.
- **Column indices array**: Stores the column index corresponding to each value in the values array.
- **Row pointer array**: Stores the cumulative number of non-zero elements in each row, allowing quick access to each row's data.

### Structure

Given a matrix:

$$
\begin{bmatrix} 
1 & 0 & 0 & 0 \\
0 & 2 & 0 & 0 \\
3 & 0 & 4 & 0 \\
1 & 0 & 0 & 5 
\end{bmatrix}
$$

The CSR representation would be:

- **Values array**: [1, 2, 3, 4, 1, 5]
- **Column indices array**: [0, 1, 0, 2, 0, 3]
- **Row pointer array**: [0, 1, 2, 4, 6]

### Explanation:

- The **values array** holds the non-zero elements in the matrix, in row-major order.
- The **column indices array** stores the corresponding column index of each non-zero element.
- The **row pointer array** keeps track of where each row starts in the values array. For example, row 1 starts at index 0, row 2 starts at index 1, row 3 starts at index 2, and so on.

### Applications

The CSR format is widely used in high-performance computing applications such as:

- Finite element analysis (FEA)
- Solving large sparse linear systems (e.g., in numerical simulations)
- Machine learning algorithms (e.g., support vector machines with sparse input)
- Graph-based algorithms where adjacency matrices are often sparse

The CSR format improves both memory efficiency and the speed of matrix operations by focusing only on non-zero elements.

### Verified Pure NumPy Solution (From Scratch)
```python
import numpy as np

def compressed_row_sparse_matrix(dense_matrix):
    vals = []
    col_idx = []
    row_ptr = [0]

    for row in dense_matrix:
        for j, val in enumerate(row):
            if val != 0:
                vals.append(val)
                col_idx.append(j)
        row_ptr.append(len(vals))

    return vals, col_idx, row_ptr
```

### Verified Sample Trace Data
```json
{
  "test": "dense_matrix = [\n    [1, 0, 0, 0],\n    [0, 2, 0, 0],\n    [3, 0, 4, 0],\n    [1, 0, 0, 5]\n]\nvals, col_idx, row_ptr = compressed_row_sparse_matrix(dense_matrix)\nprint(\"Values array:\", vals)\nprint(\"Column indices array:\", col_idx)\nprint(\"Row pointer array:\", row_ptr)",
  "expected_output": "Values array: [1, 2, 3, 4, 1, 5]\nColumn indices array: [0, 1, 0, 2, 0, 3]\nRow pointer array: [0, 1, 2, 4, 6]"
}
```

---

## Problem 66: Implement Orthogonal Projection of a Vector onto a Line
- **Difficulty**: `easy`
- **Category**: Linear Algebra

### Description & Spec
## Task: Compute the Orthogonal Projection of a Vector

Your task is to implement a function that calculates the orthogonal projection of a vector **v** onto another vector **L**. This projection results in the vector on **L** that is closest to **v**.

Write a function `orthogonal_projection(v, L)` that takes in two lists, `v` (the vector to be projected) and `L` (the line vector), and returns the orthogonal projection of `v` onto `L`. The function should output a list representing the projection vector rounded to three decimal places.

### Mathematical Derivation & Intuition
## Understanding Orthogonal Projection in Vector Spaces

Orthogonal projection is a fundamental concept in linear algebra, used to project one vector onto another. The projection of vector $v$ onto a line defined by vector $L$ results in a new vector that lies on $L$, representing the closest point to $v$ on that line. This can be thought of as $v$'s shadow on $L$ if a light was shown directly down on $v$.

To project a vector $v$ onto a non-zero vector $L$ in space, we calculate the scalar projection of $v$ onto the unit vector of $L$, which represents the magnitude of the projection. The resulting projection vector lies along the direction of $L$.

For any vector $v$ in Cartesian space, the orthogonal projection onto $L$ is calculated using the formula:

$$
\text{proj}_{L} (v) = \frac{v \cdot L}{L \cdot L} L
$$

Where:

1) $v$ is the vector being projected,  
2) $L$ is the vector defining the line of projection,  
3) $v \cdot L$ is the dot product of $v$ and $L$,  
4) $L \cdot L$ is the dot product of $L$ with itself, which gives the magnitude squared of $L$.

The resulting projection vector lies along the direction of $L$ and represents the component of $v$ that is parallel to $L$.

More generally, the projection of $v$ onto a unit vector $ \hat{L} $ (the normalized version of $L$) simplifies to:

$$
\text{proj}_{L} (v) = (v \cdot \hat{L}) \hat{L}
$$

### Applications of Orthogonal Projection

Orthogonal projection has a wide range of applications across various fields in mathematics, physics, computer science, and engineering. Some of the most common applications include:

1) **Computer Graphics**: In 3D rendering, orthogonal projections are used to create 2D views of 3D objects. This projection helps in reducing dimensional complexity and displaying models from different angles.
2) **Data Science and Machine Learning**: In high-dimensional data, projection methods are used to reduce dimensions (e.g., Principal Component Analysis) by projecting data onto lower-dimensional subspaces, helping with data visualization and reducing computational complexity.

### Verified Pure NumPy Solution (From Scratch)
```python
def dot(v1, v2):
    return sum([ax1 * ax2 for ax1, ax2 in zip(v1, v2)])

def scalar_mult(scalar, v):
    return [scalar * ax for ax in v]

def orthogonal_projection(v, L):
    L_mag_sq = dot(L, L)
    proj_scalar = dot(v, L) / L_mag_sq
    proj_v = scalar_mult(proj_scalar, L)
    return [round(x, 3) for x in proj_v]
```

### Verified Sample Trace Data
```json
{
  "test": "v = [3, 4]\nL = [1, 0]\nprint(orthogonal_projection(v, L))",
  "expected_output": "[3.0, 0.0]"
}
```

---

## Problem 67: Implement Compressed Column Sparse Matrix Format (CSC)
- **Difficulty**: `easy`
- **Category**: Linear Algebra

### Description & Spec
## Task: Create a Compressed Column Sparse Matrix Representation

Your task is to implement a function that converts a dense matrix into its Compressed Column Sparse (CSC) representation. The CSC format stores only non-zero elements of the matrix and is efficient for matrices with a high number of zero elements.

Write a function `compressed_col_sparse_matrix(dense_matrix)` that takes in a two-dimensional list `dense_matrix` and returns a tuple of three lists:

- `values`: List of non-zero elements, stored in column-major order.
- `row indices`: List of row indices corresponding to each value in the values array.
- `column pointer`: List that indicates the starting index of each column in the values array.

### Mathematical Derivation & Intuition
## Understanding the Compressed Row Sparse Matrix Format

The Compressed Row Sparse (CSR) format is a data-efficient representation of sparse matrices, where most of the elements are zero. This format is particularly useful in large-scale scientific computing and machine learning applications, where memory efficiency is critical.

### Concepts

A sparse matrix is a matrix that contains a large number of zero elements. Storing such matrices in their full form can be inefficient, both in terms of memory and computational resources. The CSR format addresses this problem by storing only the non-zero elements and their positions in the matrix. In the CSR format, a matrix is represented by three one-dimensional arrays:

1) **Values array**: Contains all the non-zero elements of the matrix, stored row by row.  
2) **Column indices array**: Stores the column index corresponding to each value in the values array.  
3) **Row pointer array**: Stores the cumulative number of non-zero elements in each row, allowing quick access to each row's data. This means that it points to the position within the column indices array at which the row starts.

### Structure

Given a matrix:

$$
\begin{bmatrix}
1 & 0 & 0 & 0 \\
0 & 2 & 0 & 0 \\
3 & 0 & 4 & 0 \\
1 & 0 & 0 & 5
\end{bmatrix}
$$

The CSR representation would be:

1) **Values array**: [1, 2, 3, 4, 1, 5]  
2) **Column indices array**: [0, 1, 0, 2, 0, 3]  
3) **Row pointer array**: [0, 1, 2, 4, 6]

### Explanation:

1) The **values array** holds the non-zero elements in the matrix, in row-major order.
2) The **column indices array** stores the corresponding column index of each non-zero element.
3) The **row pointer array** keeps track of where each row starts in the values array. For example, row 1 starts at index 0, row 2 starts at index 1, row 3 starts at index 2, within the columns indices array, and so on.

### Applications

The CSR format is widely used in high-performance computing applications such as:

1) **Finite element analysis (FEA)**
2) **Solving large sparse linear systems** (e.g., in numerical simulations)
3) **Machine learning algorithms** (e.g., support vector machines with sparse input)
4) **Graph-based algorithms** where adjacency matrices are often sparse

The CSR format improves both memory efficiency and the speed of matrix operations by focusing only on non-zero elements.

### Verified Pure NumPy Solution (From Scratch)
```python
def compressed_col_sparse_matrix(dense_matrix):
    vals = []
    row_idx = []
    col_ptr = [0]

    rows, cols = len(dense_matrix), len(dense_matrix[0])

    for i in range(cols):
        for j in range(rows):
            val = dense_matrix[j][i]
            if val != 0:
                vals.append(val)
                row_idx.append(j)
        col_ptr.append(len(vals))

    return vals, row_idx, col_ptr
```

### Verified Sample Trace Data
```json
{
  "test": "dense_matrix = [\n    [0, 0, 0],\n    [0, 0, 0],\n    [0, 0, 0]\n]\nvals, row_idx, col_ptr = compressed_col_sparse_matrix(dense_matrix)\nprint(vals)\nprint(row_idx)\nprint(col_ptr)",
  "expected_output": "[]\n[]\n[0, 0, 0, 0]"
}
```

---

## Problem 68: Find the Image of a Matrix Using Row Echelon Form
- **Difficulty**: `medium`
- **Category**: Linear Algebra

### Description & Spec
## Task: Compute the Column Space of a Matrix

In this task, you are required to implement a function `matrix_image(A)` that calculates the column space of a given matrix `A`. The column space, also known as the image or span, consists of all linear combinations of the columns of `A`. To find this, you'll use concepts from linear algebra, focusing on identifying independent columns that span the matrix's image.
**Your task:** Implement the function `matrix_image(A)` to return the basis vectors that span the column space of `A`. These vectors should be extracted from the original matrix and correspond to the independent columns.

### Mathematical Derivation & Intuition
## Matrix Image, Spans, and How to Calculate It

In linear algebra, the column space, also called the **image** or **span**, of a matrix is the set of all possible linear combinations of its columns. The column space gives important information about the matrix, such as the dimensions and dependencies between columns. It is useful for solving linear systems and understanding the structure of the data in the matrix. The image of a function can also be thought of as all the values the function takes in its codomain. The image of a matrix is the span of its columns - all linear combinations of its columns.

Consider the following matrix $ A $:

$$
A = \begin{bmatrix} 
1 & 2 & 3 \\
4 & 5 & 6 \\
7 & 8 & 9
\end{bmatrix}
$$

The column space of $ A $ is the set of all linear combinations of its columns. In other words, any vector in the column space of $ A $ can be written as:

$$
\text{span}(A) = c_1 \begin{bmatrix} 
1 \\
4 \\
7
\end{bmatrix} + c_2 \begin{bmatrix} 
2 \\
5 \\
8
\end{bmatrix} + c_3 \begin{bmatrix} 
3 \\
6 \\
9
\end{bmatrix}
$$

Where $ c_1 $, $ c_2 $, and $ c_3 $ are scalars representing the linear combination of the columns of matrix $ A $.

### The Image of a Matrix

The image of a matrix is spanned by its pivot columns. To find the image of a matrix, you can use the following steps:

1) **Convert to Row Echelon Form (RREF)**

The first step is to convert the matrix to its RREF using **Gauss-Jordan Elimination**. This finds the independent equations within the matrix. In RREF form:
- Each non-zero row begins with a leading 1, called a pivot
- Rows of all zeros are at the bottom of the matrix
- Each leading 1 is to the right of the leading 1 in the row above

2) **Identify Pivot Columns**

Once the matrix is in RREF, the pivot columns are the columns that contain the leading 1s in each non-zero row. These columns represent the independent directions that span the column space of the matrix.

3) **Extract Pivot Columns from the Original Matrix**

Finally, to find the column space of the original matrix, you take the columns from the original matrix corresponding to the pivot columns in RREF.

### Applications of Matrix Image

The matrix image has important applications in:

- **Solving systems of linear equations**
- **Determining the rank of a matrix**
- **Understanding linear transformations**

It is also used in areas such as **data compression**, **computer graphics**, and **signal processing** to analyze and manipulate data effectively.

### Verified Pure NumPy Solution (From Scratch)
```python
import numpy as np

def rref(A):
    # Convert to float for division operations
    A = A.astype(np.float32)
    n, m = A.shape

    for i in range(n):
        if A[i, i] == 0:
            nonzero_current_row = np.nonzero(A[i:, i])[0] + i
            if len(nonzero_current_row) == 0:
                continue
            A[[i, nonzero_current_row[0]]] = A[[nonzero_current_row[0], i]]

        A[i] = A[i] / A[i, i]

        for j in range(n):
            if i != j:
                A[j] -= A[i] * A[j, i]
    return A

def find_pivot_columns(A):
    n, m = A.shape
    pivot_columns = []
    for i in range(n):
        nonzero = np.nonzero(A[i, :])[0]
        if len(nonzero) != 0:
            pivot_columns.append(nonzero[0])
    return pivot_columns

def matrix_image(A):
    # Find the RREF of the matrix
    Arref = rref(A)
    # Find the pivot columns
    pivot_columns = find_pivot_columns(Arref)
    # Extract the pivot columns from the original matrix
    image_basis = A[:, pivot_columns]
    return image_basis
```

### Verified Sample Trace Data
```json
{
  "test": "\nimport numpy as np\nmatrix = np.array([[1, 0], [0, 1]])\nprint(matrix_image(matrix))\n",
  "expected_output": "[[1, 0], [0, 1]]"
}
```

---

## Problem 74: Create Composite Hypervector for a Dataset Row
- **Difficulty**: `medium`
- **Category**: Linear Algebra

### Description & Spec
## Task: Generate a Composite Hypervector Using Hyperdimensional Computing

Your task is to implement the function `create_row_hv(row, dim, random_seeds)` to generate a composite hypervector for a given dataset row using Hyperdimensional Computing (HDC). Each feature in the row is represented by binding hypervectors for the feature name and its value. The hypervectors for the values are created using the same feature seed provided in the `random_seeds` dictionary to ensure reproducibility. All feature hypervectors are then bundled to create a composite hypervector for the row.

### Input:
- `row`: A dictionary representing a dataset row, where keys are feature names and values are their corresponding values.
- `dim`: The dimensionality of the hypervectors.
- `random_seeds`: A dictionary where keys are feature names and values are seeds to ensure reproducibility of hypervectors.

### Output:
- A composite hypervector representing the entire row.

### Mathematical Derivation & Intuition
## Hyperdimensional Computing
Hyperdimensional Computing (HDC) is a computational model inspired by the brain's ability to represent and process information using high-dimensional vectors, based on hypervectors being quasi-orthogonal. It uses vectors with a large number of dimensions to represent data, where each vector is typically filled with binary (1 or 0) or bipolar values (1 or -1). To represent complex data patterns, binding and bundling operations are used. 

In HDC, different data types such as numeric and categorical variables are projected into high-dimensional space through specific encoding processes. Categorical variables are assigned unique hypervectors, often randomly generated binary or bipolar vectors, that serve as representations for each category. Numeric variables are encoded by discretizing the continuous values and mapping discrete bins to hypervectors. These projections allow HDC models to integrate various data types into a unified high-dimensional representation, preserving information across complex, multi-feature datasets.

---

## Binding Operation
The binding operation between two hypervectors is performed element-wise using multiplication. This operation is used to represent associations between different pieces of information:

$$
\text{bind}(\text{hv1}, \text{hv2}) = \text{hv1} \times \text{hv2}
$$

Where $ \text{hv1} $ and $ \text{hv2} $ are bipolar vectors, and their element-wise multiplication results in a new vector where each element is either 1 or -1.

---

## Bundling Operation
The bundling operation sums multiple hypervectors to combine information, typically using element-wise addition for bipolar vectors and XOR operations for binary vectors. This operation aggregates information and creates a composite hypervector that represents the overall data or concept. For example, for a set of $ n $ hypervectors $ \text{hv1}, \text{hv2}, \dots, \text{hvn} $, the bundled vector is:

$$
\text{bundle}(\text{hv1}, \text{hv2}, \dots, \text{hvn}) = \sum_{i=1}^{n} \text{hvi}
$$

This bundled vector is then normalized to ensure it remains bipolar.

---

## Normalization
Normalization ensures that the final bundled vector contains only bipolar or binary values. The normalization function typically applies a thresholding process that transforms any value greater than zero to +1 and any value less than zero to -1. Zero values are then typically assigned to either +1 or -1.

---

## Operations in Practice: Example
Consider a scenario where we want to represent and combine information from each feature in a row of a dataset. Each feature, whether numeric or categorical, is represented by a hypervector, and these hypervectors are combined to form a composite vector that represents the entire row of data.

For instance, if we have a dataset row with features Feature A and Feature B, we would:
1. Create a hypervector for the column Feature A and another for its specific feature value.
2. Create a hypervector for the column Feature B and another for its specific feature value.
3. Bind each feature’s column hypervector with the hypervector representing its value to form a unique vector for each feature.
4. Bundle all the feature hypervectors for this row to create a single composite vector representing the entire row.
5. Normalize the bundled vector to maintain bipolar values.

---

## Applications of HDC
Hyperdimensional computing has a variety of applications, including:
1. **Data Classification**: Using high-dimensional vectors to represent data points and classifying them based on their properties.
2. **Pattern Recognition**: Recognizing complex patterns in data through binding and bundling operations.
3. **Natural Language Processing**: Representing words and phrases as high-dimensional vectors to analyze and process text data.

### Verified Pure NumPy Solution (From Scratch)
```python
import numpy as np

def create_hv(dim):
    return np.random.choice([-1, 1], dim)

def create_col_hvs(dim, seed):
    np.random.seed(seed)
    return create_hv(dim), create_hv(dim)

def bind(hv1, hv2):
    return hv1 * hv2

def bundle(hvs, dim):
    bundled = np.sum(list(hvs.values()), axis=0)
    return sign(bundled)

def sign(vector, threshold=0.01):
    return np.array([1 if v >= 0 else -1 for v in vector])

def create_row_hv(row, dim, random_seeds):
    row_hvs = {col: bind(*create_col_hvs(dim, random_seeds[col])) for col in row.keys()}
    return bundle(row_hvs, dim)
```

### Verified Sample Trace Data
```json
{
  "test": "\nrow = {\"FeatureA\": \"value1\", \"FeatureB\": \"value2\"}\ndim = 5\nrandom_seeds = {\"FeatureA\": 42, \"FeatureB\": 7}\nprint(create_row_hv(row, dim, random_seeds))\n",
  "expected_output": "[1, -1, 1, 1, 1]"
}
```

---

## Problem 76: Calculate Cosine Similarity Between Vectors
- **Difficulty**: `easy`
- **Category**: Linear Algebra

### Description & Spec
## Task: Implement Cosine Similarity

In this task, you need to implement a function `cosine_similarity(v1, v2)` that calculates the cosine similarity between two vectors. Cosine similarity measures the cosine of the angle between two vectors, indicating their directional similarity.

### Input:
- `v1` and `v2`: Numpy arrays representing the input vectors.

### Output:
- A float representing the cosine similarity, rounded to three decimal places.

### Constraints:
- Both input vectors must have the same shape.
- Input vectors cannot be empty or have zero magnitude.

### Mathematical Derivation & Intuition
## Cosine Similarity

Cosine similarity measures the cosine of the angle between two vectors. It doesn't consider the magnitude of the vectors but focuses on the angle between them.

### Cosine Similarity Formula
$$
\cos(\theta) = \frac{\sum_{i=1}^{p} A_i B_i}{\sqrt{\sum_{i=1}^{p} A_i^2} \sqrt{\sum_{i=1}^{p} B_i^2}}
$$

### Implementation Steps for Cosine Similarity
1. **Handle Input**: Ensure input vectors have the same dimensions and handle edge cases (e.g., zero vectors).
2. **Dot Product**: Compute $\sum_{i=1}^{p} A_i B_i $ for the two vectors.
3. **Magnitudes**: Compute the L2 norms $ \sqrt{\sum_{i=1}^{p} A_i^2} $ and $ \sqrt{\sum_{i=1}^{p} B_i^2} $.
4. **Final Result**: Divide the dot product by the product of the magnitudes.

### Use Cases
1. **Text and Image Similarity**
2. **Recommendation Systems**
3. **Query Matching**

### Pitfalls
1. **Magnitude Blindness**:
   - Example:
     - $ \text{vector1} = (1, 1) $
     - $ \text{vector2} = (1000, 1000) $
     - Cosine similarity $ = 1 $, despite the vastly different magnitudes.
2. **Sparse Data Issues**:
   - In high-dimensional spaces, where data is often sparse, cosine similarity may become less reliable.
3. **Non-Negative Data Limitation**:
   - If all values are positive, cosine similarity cannot capture negative relationships or inverse trends.

### Verified Pure NumPy Solution (From Scratch)
```python
import numpy as np

def cosine_similarity(v1, v2):
    if v1.shape != v2.shape:
        raise ValueError("Arrays must have the same shape")

    if v1.size == 0:
        raise ValueError("Arrays cannot be empty")

    # Flatten arrays in case of 2D
    v1_flat = v1.flatten()
    v2_flat = v2.flatten()

    dot_product = np.dot(v1_flat, v2_flat)
    magnitude1 = np.sqrt(np.sum(v1_flat**2))
    magnitude2 = np.sqrt(np.sum(v2_flat**2))

    if magnitude1 == 0 or magnitude2 == 0:
        raise ValueError("Vectors cannot have zero magnitude")

    return round(dot_product / (magnitude1 * magnitude2), 3)
```

### Verified Sample Trace Data
```json
{
  "test": "\nimport numpy as np\n\nv1 = np.array([1, 2, 3])\nv2 = np.array([2, 4, 6])\nprint(cosine_similarity(v1, v2))\n",
  "expected_output": "1.0"
}
```

---

## Problem 83: Dot Product Calculator
- **Difficulty**: `easy`
- **Category**: Linear Algebra

### Description & Spec
Write a Python function to calculate the dot product of two vectors. The function should take two 1D NumPy arrays as input and return the dot product as a single number.

### Mathematical Derivation & Intuition
## Calculating the Dot Product of Two Vectors

The dot product, also known as the scalar product, is a mathematical operation that takes two equal-length vectors and returns a single number. It is widely used in physics, geometry, and linear algebra.

### 1. Formula for the Dot Product

The dot product of two vectors $\mathbf{a}$ and $\mathbf{b}$, each of length $n$, is calculated as follows:

$$
\mathbf{a} \cdot \mathbf{b} = \sum_{i=1}^{n} a_i b_i
$$

This means multiplying corresponding elements of the two vectors and summing up the results.

### 2. Geometric Interpretation

In geometric terms, the dot product can also be expressed as:

$$
\mathbf{a} \cdot \mathbf{b} = |\mathbf{a}| |\mathbf{b}| \cos \theta
$$

Where:

- $|\mathbf{a}|$ and $|\mathbf{b}|$ are the magnitudes of the vectors.
- $\theta$ is the angle between the two vectors.

### 3. Properties of the Dot Product

1. **Commutative**:  
   $$
   \mathbf{a} \cdot \mathbf{b} = \mathbf{b} \cdot \mathbf{a}
   $$

2. **Distributive**:  
   $$
   \mathbf{a} \cdot (\mathbf{b} + \mathbf{c}) = \mathbf{a} \cdot \mathbf{b} + \mathbf{a} \cdot \mathbf{c}
   $$

3. **Orthogonal Vectors**:  
   If:  
   $$
   \mathbf{a} \cdot \mathbf{b} = 0
   $$  
   Then $\mathbf{a}$ and $\mathbf{b}$ are perpendicular.

### 4. Example Calculation

Given two vectors:

- $\mathbf{a} = [1, 2, 3]$
- $\mathbf{b} = [4, 5, 6]$

The dot product is calculated as:

$$
\mathbf{a} \cdot \mathbf{b} = (1 \cdot 4) + (2 \cdot 5) + (3 \cdot 6) = 4 + 10 + 18 = 32
$$

### Conclusion

The dot product is a fundamental operation in vector algebra, useful in determining angles between vectors, projections, and in many applications across physics and engineering.

### Verified Pure NumPy Solution (From Scratch)
```python
import numpy as np

def calculate_dot_product(vec1, vec2):
    """
    Calculate the dot product of two vectors.
    Args:
        vec1 (numpy.ndarray): 1D array representing the first vector.
        vec2 (numpy.ndarray): 1D array representing the second vector.
    Returns:
        float: Dot product of the two vectors.
    """
    return np.dot(vec1, vec2)
```

### Verified Sample Trace Data
```json
{
  "test": "print(calculate_dot_product(np.array([1, 2, 3]), np.array([4, 5, 6])))",
  "expected_output": "32"
}
```

---

## Problem 84: Phi Transformation for Polynomial Features
- **Difficulty**: `easy`
- **Category**: Linear Algebra

### Description & Spec
Write a Python function to perform a Phi Transformation that maps input features into a higher-dimensional space by generating polynomial features. The transformation allows models like linear regression to fit nonlinear data by introducing new feature dimensions that represent polynomial combinations of the original input features. The function should take a list of numerical data and a degree as inputs, and return a nested list where each inner list represents the transformed features of a data point. If the degree is less than 0, the function should return an empty list.

### Mathematical Derivation & Intuition
## Phi Transformation

The Phi Transformation maps input features into a higher-dimensional space by generating polynomial features. This allows models like linear regression to fit nonlinear data by introducing new feature dimensions that represent polynomial combinations of the original input features.

### Why Use Phi Transformation?

- To increase the expressive power of simple models such as linear models.
- To enable better fitting of nonlinear relationships in the data.

### Equations

For an input value $x$, the Phi Transformation expands it as:

$$
\Phi(x) = [1, x, x^2, x^3, \dots, x^d]
$$

Where $d$ is the specified degree, and $\Phi(x)$ represents the transformed feature vector.

### Example 1: Polynomial Expansion for One Value

Given $x = 3$ and $d = 3$, the Phi Transformation is:

$$
\Phi(3) = [1, 3, 9, 27]
$$

### Example 2: Transformation for Multiple Values

For $\text{data} = [1, 2]$ and $d = 2$, the Phi Transformation is:

$$
\Phi([1, 2]) = \begin{bmatrix} 1 & 1 & 1 \\ 1 & 2 & 4 \end{bmatrix}
$$

### Verified Pure NumPy Solution (From Scratch)
```python
import numpy as np

def phi_transform(data: list[float], degree: int) -> list[list[float]]:
	"""
	Perform a Phi Transformation to map input features into a higher-dimensional space by generating polynomial features.

	Args:
		data (list[float]): A list of numerical values to transform.
		degree (int): The degree of the polynomial expansion.

	Returns:
		list[list[float]]: A nested list where each inner list represents the transformed features of a data point.
	"""
	if degree < 0 or not data:
		return []
	return np.array([[x ** i for i in range(degree + 1)] for x in data]).tolist()
```

### Verified Sample Trace Data
```json
{
  "test": "print(phi_transform([], 2))",
  "expected_output": "[]"
}
```

---

## Problem 117: Compute Orthonormal Basis for 2D Vectors
- **Difficulty**: `medium`
- **Category**: Linear Algebra

### Description & Spec
Implement a function that computes an orthonormal basis for the subspace spanned by a list of 2D vectors using the Gram-Schmidt process. The function should take a list of 2D vectors and a tolerance value (tol) to determine linear independence, returning a list of orthonormal vectors (unit length and orthogonal to each other) that span the same subspace. This is a fundamental concept in linear algebra with applications in machine learning, such as feature orthogonalization.

### Mathematical Derivation & Intuition
## Understanding the Gram-Schmidt Process

The Gram-Schmidt process transforms a set of vectors into an orthonormal basis vectors that are orthogonal (perpendicular) and have unit length for the subspace they span.

### Mathematical Definition

Given vectors $v_1, v_2, \ldots$, the process constructs an orthonormal set $u_1, u_2, \ldots$ as follows:
1. $u_1 = \frac{v_1}{\|v_1\|}$ (normalize the first vector).
2. For subsequent vectors $v_k$:
   - Subtract projections: $$w_k = v_k - \sum_{i=1}^{k-1} \text{proj}_{u_i}(v_k),$$ where $\text{proj}_{u_i}(v_k) = (v_k \cdot u_i) u_i$.
   - Normalize: $$u_k = \frac{w_k}{\|w_k\|},$$ if $\|w_k\| > \text{tol}$.

### Why Orthonormal Bases?

- Orthogonal vectors simplify computations (e.g., their dot product is zero).
- Unit length ensures equal scaling, useful in $PCA$, $QR$ decomposition, and neural network optimization.

### Special Case

If a vector's norm is less than or equal to $\text{tol}$ (default $1e-10$), it's considered linearly dependent and excluded from the basis.

### Example

For vectors `[[1, 0], [1, 1]]` with $\text{tol} = 1e-10$:
1. $v_1 = [1, 0]$, $\|v_1\| = 1$, so $u_1 = [1, 0]$.
2. $v_2 = [1, 1]$, projection on $u_1$: $(v_2 \cdot u_1) u_1 = 1 \cdot [1, 0] = [1, 0]$.
   - $w_2 = [1, 1] - [1, 0] = [0, 1]$.
   - $\|w_2\| = 1 > 1e-10$, so $u_2 = [0, 1]$.

Result: `[[1, 0], [0, 1]]`, rounded to 4 decimal places.

### Verified Pure NumPy Solution (From Scratch)
```python
import numpy as np

def orthonormal_basis(vectors: list[list[float]], tol: float = 1e-10) -> list[np.ndarray]:
    basis = []
    for v in vectors:
        v = np.array(v, dtype=float)
        for b in basis:
            v = v - np.dot(v, b) * b
        norm = np.sqrt(np.dot(v, v))
        if norm > tol:
            v = v / norm
            basis.append(v)
    return basis
```

### Verified Sample Trace Data
```json
{
  "test": "basis = orthonormal_basis([[1, 0], [1, 1]]); print([b.round(4) for b in basis])",
  "expected_output": "[array([1., 0.]), array([0., 1.])]"
}
```

---

## Problem 118: Compute the Cross Product of Two 3D Vectors
- **Difficulty**: `easy`
- **Category**: Linear Algebra

### Description & Spec
Implement a function to compute the cross product of two 3-dimensional vectors. The cross product of two vectors results in a third vector that is perpendicular to both and follows the right-hand rule. This concept is fundamental in physics, engineering, and 3D graphics.

### Mathematical Derivation & Intuition
## Understanding the Cross Product

The **cross product** of two vectors $\vec{a}$ and $\vec{b}$ in 3D space is a vector that is perpendicular to both $\vec{a}$ and $\vec{b}$.

### Properties
- Defined only in 3 dimensions.
- The result $\vec{c} = \vec{a} \times \vec{b}$ is perpendicular to both $\vec{a}$ and $\vec{b}$.
- Follows the right-hand rule.

### Mathematical Formula

Given:
- $\vec{a} = [a_1, a_2, a_3]$
- $\vec{b} = [b_1, b_2, b_3]$

The cross product is:

$$
\vec{a} \times \vec{b} = [a_2 b_3 - a_3 b_2,\ a_3 b_1 - a_1 b_3,\ a_1 b_2 - a_2 b_1]
$$

### Use Cases
- Calculating normals in 3D graphics.
- Determining torque and angular momentum in physics.
- Verifying orthogonality in machine learning geometry.

### Example
For $\vec{a} = [1, 0, 0]$ and $\vec{b} = [0, 1, 0]$:

$$
\vec{a} \times \vec{b} = [0, 0, 1]
$$

The result points in the $z$-axis direction, confirming perpendicularity to both $\vec{a}$ and $\vec{b}$.

### Verified Pure NumPy Solution (From Scratch)
```python
import numpy as np

def cross_product(a, b):
    """
    Compute the cross product of two 3D vectors a and b.
    Parameters:
        a (array-like): A 3-element vector.
        b (array-like): A 3-element vector.
    Returns:
        numpy.ndarray: The cross product vector.
    """
    a = np.array(a)
    b = np.array(b)

    if a.shape != (3,) or b.shape != (3,):
        raise ValueError("Both input vectors must be of length 3.")

    cross = np.array([
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0]
    ])
    return cross
```

### Verified Sample Trace Data
```json
{
  "test": "print(cross_product([1, 0, 0], [0, 1, 0]))",
  "expected_output": "[0, 0, 1]"
}
```

---

## Problem 119: Solve System of Linear Equations Using Cramer's Rule
- **Difficulty**: `medium`
- **Category**: Linear Algebra for Machine Learning

### Description & Spec
Implement a function to solve a system of linear equations $Ax = b$ using Cramer's Rule. The function should take a square coefficient matrix $A$ and a constant vector $b$, and return the solution vector $x$. If the system has no unique solution (i.e., the determinant of $A$ is zero), return -1.

### Mathematical Derivation & Intuition
## Understanding Cramer's Rule

Cramer's Rule is a method to solve a system of linear equations $Ax = b$ using determinants.

### Requirements
- The coefficient matrix $A$ must be square ($n \times n$).
- The determinant of $A$, $\det(A)$, must be non-zero for a unique solution to exist.

### Formula
For each variable $x_i$, replace the $i$-th column of $A$ with vector $b$ and compute:

$$
x_i = \frac{\det(A_i)}{\det(A)}
$$

Where:
- $A_i$ is the matrix formed by replacing the $i$-th column of $A$ with $b$
- $\det(A)$ is the determinant of the original matrix $A$

### Steps
1. Compute $\det(A)$. If it's 0, return -1.
2. For each variable $x_i$:
   - Replace column $i$ in $A$ with $b$
   - Compute $\det(A_i)$
   - Compute $x_i = \frac{\det(A_i)}{\det(A)}$

### Example
Given:

$$
A = \begin{bmatrix} 2 & -1 & 3 \\ 4 & 2 & 1 \\ -6 & 1 & -2 \end{bmatrix}, \quad b = \begin{bmatrix} 5 \\ 10 \\ -3 \end{bmatrix}
$$

1. $\det(A) = -36.0$
2. Replace each column with $b$:

- $\det(A_1) = -6.0$
- $\det(A_2) = -120.0$
- $\det(A_3) = -96.0$

Then,

$$
x = \left[ \frac{-6}{-36}, \frac{-120}{-36}, \frac{-96}{-36} \right] = [0.1667, 3.3333, 2.6667]
$$

### Applications
- Solving small systems of equations
- Useful in theoretical linear algebra
- Not practical for large matrices due to computational cost

### Verified Pure NumPy Solution (From Scratch)
```python
import numpy as np

def cramers_rule(A, b):
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)

    n, m = A.shape
    if n != m or b.shape[0] != n:
        return -1

    det_A = np.linalg.det(A)
    if np.isclose(det_A, 0):
        return -1

    x = np.zeros(n)
    for i in range(n):
        A_mod = A.copy()
        A_mod[:, i] = b
        det_A_mod = np.linalg.det(A_mod)
        x[i] = det_A_mod / det_A

    return x
```

### Verified Sample Trace Data
```json
{
  "test": "print(np.round(cramers_rule([[2, -1, 3], [4, 2, 1], [-6, 1, -2]], [5, 10, -3]), 4))",
  "expected_output": "[0.1667, 3.3333, 2.6667]"
}
```

---

## Problem 121: Vector Element-wise Sum
- **Difficulty**: `easy`
- **Category**: Linear Algebra

### Description & Spec
Write a Python function that computes the element-wise sum of two vectors. The function should return a new vector representing the resulting sum if the operation is valid, or -1 if the vectors have incompatible dimensions. Two vectors (lists) can be summed element-wise only if they are of the same length.

### Mathematical Derivation & Intuition
## Understanding Vector Element-wise Sum

In linear algebra, the **element-wise sum** (also known as vector addition) involves adding corresponding entries of two vectors.

### Vector Notation
Given two vectors $a$ and $b$ of the same dimension $n$:

$$
a = \begin{pmatrix} a_1 \\ a_2 \\ \vdots \\ a_n \end{pmatrix}, \quad b = \begin{pmatrix} b_1 \\ b_2 \\ \vdots \\ b_n \end{pmatrix}
$$

The element-wise sum is defined as:

$$
a + b = \begin{pmatrix} a_1 + b_1 \\ a_2 + b_2 \\ \vdots \\ a_n + b_n \end{pmatrix}
$$

### Key Requirement
Vectors $a$ and $b$ must be of the same length $n$ for the operation to be valid. If their lengths differ, element-wise addition is not defined.

### Example
Let:

$$
a = [1, 2, 3], \quad b = [4, 5, 6]
$$

Then:

$$
a + b = [1+4, 2+5, 3+6] = [5, 7, 9]
$$

This simple operation is foundational in many applications such as vector arithmetic, neural network computations, and linear transformations.

### Verified Pure NumPy Solution (From Scratch)
```python
def vector_sum(a: list[int|float], b: list[int|float]) -> list[int|float]:
    if len(a) != len(b):
        return -1
    return [a[i] + b[i] for i in range(len(a))]
```

### Verified Sample Trace Data
```json
{
  "test": "print(vector_sum([1, 2, 3], [4, 5, 6]))",
  "expected_output": "[5, 7, 9]"
}
```

---

