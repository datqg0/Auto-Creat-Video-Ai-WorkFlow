# Deep-ML Knowledge Base: Machine Learning Fundamentals

Total verified problems: 54

Source: [Deep-ML (Open-Deep-ML)](https://www.deep-ml.com/)

---

## Problem 14: Linear Regression Using Normal Equation
- **Difficulty**: `easy`
- **Category**: Machine Learning

### Description & Spec
Write a Python function that performs linear regression using the normal equation. The function should take a matrix X (features) and a vector y (target) as input, and return the coefficients of the linear regression model. Round your answer to four decimal places, -0.0 is a valid result for rounding a very small number.

### Mathematical Derivation & Intuition
## Linear Regression Using the Normal Equation

Linear regression aims to model the relationship between a scalar dependent variable \( y \) and one or more explanatory variables (or independent variables) \( X \). The normal equation provides an analytical solution to find the coefficients \( \theta \) that minimize the cost function for linear regression.

Given a matrix \( X \) (with each row representing a training example and each column a feature) and a vector \( y \) (representing the target values), the normal equation is:
$$
\theta = (X^TX)^{-1}X^Ty
$$

### Explanation of Terms
1. \( X^T \) is the transpose of \( X \).
2. \( (X^TX)^{-1} \) is the inverse of the matrix \( X^TX \).
3. \( y \) is the vector of target values.

### Key Points
- **Feature Scaling**: This method does not require feature scaling.
- **Learning Rate**: There is no need to choose a learning rate.
- **Computational Cost**: Computing the inverse of \( X^TX \) can be computationally expensive if the number of features is very large.

### Practical Implementation
A practical implementation involves augmenting \( X \) with a column of ones to account for the intercept term and then applying the normal equation directly to compute \( \theta \).

### Verified Pure NumPy Solution (From Scratch)
```python
import numpy as np
def linear_regression_normal_equation(X: list[list[float]], y: list[float]) -> list[float]:
    X = np.array(X)
    y = np.array(y).reshape(-1, 1)
    X_transpose = X.T
    theta = np.linalg.inv(X_transpose.dot(X)).dot(X_transpose).dot(y)
    theta = np.round(theta, 4).flatten().tolist()
    return theta
```

### Verified Sample Trace Data
```json
{
  "test": "print(linear_regression_normal_equation([[1, 1], [1, 2], [1, 3]], [1, 2, 3]))",
  "expected_output": "[-0.0, 1.0]"
}
```

---

## Problem 15: Linear Regression Using Gradient Descent
- **Difficulty**: `easy`
- **Category**: Machine Learning

### Description & Spec
Write a Python function that performs linear regression using gradient descent. The function should take NumPy arrays X (features with a column of ones for the intercept) and y (target) as input, along with learning rate alpha and the number of iterations, and return the coefficients of the linear regression model as a NumPy array. Round your answer to four decimal places. -0.0 is a valid result for rounding a very small number.

### Mathematical Derivation & Intuition
## Linear Regression Using Gradient Descent

Linear regression can also be performed using a technique called gradient descent, where the coefficients (or weights) of the model are iteratively adjusted to minimize a cost function (usually mean squared error). This method is particularly useful when the number of features is too large for analytical solutions like the normal equation or when the feature matrix is not invertible.

The gradient descent algorithm updates the weights by moving in the direction of the negative gradient of the cost function with respect to the weights. The updates occur iteratively until the algorithm converges to a minimum of the cost function.

The update rule for each weight is given by:
$$
\theta_j := \theta_j - \alpha \frac{1}{m} \sum_{i=1}^{m} \left( h_{\theta}(x^{(i)}) - y^{(i)} \right)x_j^{(i)}
$$

### Explanation of Terms
1. $\alpha$ is the learning rate.
2. $m$ is the number of training examples.
3. $h_{\theta}(x^{(i)})$ is the hypothesis function at iteration $i$.
4. $x^{(i)}$ is the feature vector of the $i^{\text{th}}$ training example.
5. $y^{(i)}$ is the actual target value for the $i^{\text{th}}$ training example.
6. $x_j^{(i)}$ is the value of feature $j$ for the $i^{\text{th}}$ training example.

### Key Points
- **Learning Rate**: The choice of learning rate is crucial for the convergence and performance of gradient descent. 
  - A small learning rate may lead to slow convergence.
  - A large learning rate may cause overshooting and divergence.
- **Number of Iterations**: The number of iterations determines how long the algorithm runs before it converges or stops.

### Practical Implementation
Implementing gradient descent involves:
1. Initializing the weights theta ($\theta$), with 0s in our case, 
2. Compute the predicted values for all training examples using the current weights
3. Computing the gradient for each iteration step by aggregating the errors over all training examples and scaling it by the learning rate,
4. and iteratively updating the weights according to the update rule.

### Verified Pure NumPy Solution (From Scratch)
```python
import numpy as np
def linear_regression_gradient_descent(X: np.ndarray, y: np.ndarray, alpha: float, iterations: int) -> np.ndarray:
    """
    Perform linear regression using gradient descent.

    m = number of training examples
    n = number of parameters (features), technically n-1 features, 1st column is for intercept

    X: shape (m, n), `m` training examples with `n` input values for each feature
    y: shape (m, 1) array with the target values (ground truth)
    alpha: learning rate
    iterations: number of gradient descent steps
    """

    m, n = X.shape
    y = y.reshape(-1, 1)    # Make sure y is a column vector
    theta = np.zeros((n, 1))                    # 1. Initializing the weights with 0s
    for _ in range(iterations):
        predictions = X @ theta                 # 2. Compute predictions h(x)
        errors = predictions - y                # 3. Compute error = h - y
        gradient = alpha * (X.T @ errors) / m   # 4. Compute gradient
        theta -= gradient                       # 5. Update weights for each iteration
    return np.round(theta.flatten(), 4)
```

### Verified Sample Trace Data
```json
{
  "test": "print(linear_regression_gradient_descent(np.array([[1, 1], [1, 2], [1, 3]]), np.array([1, 2, 3]), 0.01, 1000))",
  "expected_output": "[0.1107, 0.9513]"
}
```

---

## Problem 16: Feature Scaling Implementation
- **Difficulty**: `easy`
- **Category**: Machine Learning

### Description & Spec
Write a Python function that performs feature scaling on a dataset using both standardization and min-max normalization. The function should take a 2D NumPy array as input, where each row represents a data sample and each column represents a feature. It should return two 2D NumPy arrays: one scaled by standardization and one by min-max normalization. Make sure all results are rounded to the nearest 4th decimal.

### Mathematical Derivation & Intuition
## Feature Scaling Techniques

Feature scaling is crucial in many machine learning algorithms that are sensitive to the magnitude of features. This includes algorithms that use distance measures, like k-nearest neighbors, and gradient descent-based algorithms, like linear regression.

### Standardization
Standardization (or Z-score normalization) is the process where features are rescaled so that they have the properties of a standard normal distribution with a mean of zero and a standard deviation of one:
$$
z = \frac{(x - \mu)}{\sigma}
$$
where \( x \) is the original feature, \( \mu \) is the mean of that feature, and \( \sigma \) is the standard deviation.

### Min-Max Normalization
Min-max normalization rescales the feature to a fixed range, typically 0 to 1, or it can be shifted to any range \([a, b]\) by transforming the data using the formula:
$$
x' = \frac{(x - \text{min}(x))}{(\text{max}(x) - \text{min}(x))} \times (\text{max} - \text{min}) + \text{min}
$$
where \( x \) is the original value, \( \text{min}(x) \) is the minimum value for that feature, \( \text{max}(x) \) is the maximum value, and \( \text{min} \) and \( \text{max} \) are the new minimum and maximum values for the scaled data.

### Key Points
- **Equal Contribution**: Implementing these scaling techniques ensures that features contribute equally to the development of the model.
- **Improved Convergence**: Feature scaling can significantly improve the convergence speed of learning algorithms.

This structured explanation outlines the importance of feature scaling and describes two commonly used techniques with their mathematical formulas.

### Verified Pure NumPy Solution (From Scratch)
```python
import numpy as np

def feature_scaling(data):
    # Standardization
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    standardized_data = (data - mean) / std
    
    # Min-Max Normalization
    min_val = np.min(data, axis=0)
    max_val = np.max(data, axis=0)
    normalized_data = (data - min_val) / (max_val - min_val)
    
    return np.round(standardized_data,4).tolist(), np.round(normalized_data,4).tolist()
```

### Verified Sample Trace Data
```json
{
  "test": "print(feature_scaling(np.array([[1, 2], [3, 4], [5, 6]])))",
  "expected_output": "([[-1.2247, -1.2247], [0.0, 0.0], [1.2247, 1.2247]], [[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]])"
}
```

---

## Problem 17: K-Means Clustering
- **Difficulty**: `medium`
- **Category**: Machine Learning

### Description & Spec
Your task is to write a Python function that implements the k-Means clustering algorithm. This function should take specific inputs and produce a list of final centroids. k-Means clustering is a method used to partition `n` points into `k` clusters. The goal is to group similar points together and represent each group by its center (called the *centroid*).

### Function Inputs:

- `points`: A list of points, where each point is a tuple of coordinates (e.g., `(x, y)` for 2D points)
- `k`: An integer representing the number of clusters to form
- `initial_centroids`: A list of initial centroid points, each a tuple of coordinates
- `max_iterations`: An integer representing the maximum number of iterations to perform

### Function Output:

A list of the final centroids of the clusters, where each centroid is rounded to the nearest fourth decimal.

### Mathematical Derivation & Intuition
## K-Means Clustering Algorithm Implementation

### Algorithm Steps

1. **Initialization**  
   Use the provided `initial_centroids` as your starting point. This step is already done for you in the input.

2. **Assignment Step**  
   For each point in your dataset:
   - Calculate its distance to each centroid.
   - Assign the point to the cluster of the nearest centroid.  
   *Hint*: Consider creating a helper function to calculate the Euclidean distance between two points.

3. **Update Step**  
   For each cluster:
   - Calculate the mean of all points assigned to the cluster.
   - Update the centroid to this new mean position.  
   *Hint*: Be careful with potential empty clusters. Decide how you'll handle them (e.g., keep the previous centroid).

4. **Iteration**  
   Repeat steps 2 and 3 until either:
   - The centroids no longer change significantly (this case does not need to be included in your solution), or
   - You reach the `max_iterations` limit.  
   *Hint*: You might want to keep track of the previous centroids to check for significant changes.

5. **Result**  
   Return the list of final centroids, ensuring each coordinate is rounded to the nearest fourth decimal.

### Verified Pure NumPy Solution (From Scratch)
```python
import numpy as np

def euclidean_distance(a, b):
    return np.sqrt(((a - b) ** 2).sum(axis=1))

def k_means_clustering(points, k, initial_centroids, max_iterations):
    points = np.array(points)
    centroids = np.array(initial_centroids)
    
    for iteration in range(max_iterations):
        # Assign points to the nearest centroid
        distances = np.array([euclidean_distance(points, centroid) for centroid in centroids])
        assignments = np.argmin(distances, axis=0)

        new_centroids = np.array([points[assignments == i].mean(axis=0) if len(points[assignments == i]) > 0 else centroids[i] for i in range(k)])
        
        # Check for convergence
        if np.all(centroids == new_centroids):
            break
        centroids = new_centroids
        centroids = np.round(centroids,4)
    return [tuple(centroid) for centroid in centroids]
```

### Verified Sample Trace Data
```json
{
  "test": "print(k_means_clustering([(1, 2), (1, 4), (1, 0), (10, 2), (10, 4), (10, 0)], 2, [(1, 1), (10, 1)], 10))",
  "expected_output": "[(1.0, 2.0), (10.0, 2.0)]"
}
```

---

## Problem 18: Implement K-Fold Cross-Validation
- **Difficulty**: `medium`
- **Category**: Machine Learning

### Description & Spec
Implement a function to generate train and test splits for K-Fold Cross-Validation. Your task is to divide the dataset into k folds and return a list of train-test indices for each fold.

### Mathematical Derivation & Intuition
## Understanding K-Fold Cross-Validation

K-Fold Cross-Validation is a resampling technique used to evaluate machine learning models by partitioning the dataset into multiple folds.

### How it Works
1. The dataset is split into **k** equal (or almost equal) parts called folds.
2. Each fold is used **once** as a test set, while the remaining **k-1** folds form the training set.
3. The process is repeated **k times**, ensuring each fold serves as a test set exactly once.

### Why Use K-Fold Cross-Validation?
- It provides a more **robust** estimate of model performance than a single train-test split.
- Reduces bias introduced by a single training/testing split.
- Allows evaluation across multiple data distributions.

### Implementation Steps
1. Shuffle the data if required.
2. Split the dataset into **k** equal (or nearly equal) folds.
3. Iterate over each fold, using it as the test set while using the remaining data as the training set.
4. Return train-test indices for each iteration.

By implementing this function, you will learn how to **split a dataset for cross-validation**, a crucial step in model evaluation.

### Verified Pure NumPy Solution (From Scratch)
```python
import numpy as np

def k_fold_cross_validation(X: np.ndarray, y: np.ndarray, k=5, shuffle=True):
    """
    Return train and test indices for k-fold cross-validation.
    """
    n_samples = len(X)
    indices = np.arange(n_samples)
    
    if shuffle:
        if random_seed is not None:
        np.random.shuffle(indices)
    
    fold_sizes = np.full(k, n_samples // k, dtype=int)
    fold_sizes[:n_samples % k] += 1

    current = 0
    folds = []
    for fold_size in fold_sizes:
        folds.append(indices[current:current + fold_size])
        current += fold_size

    result = []
    for i in range(k):
        test_idx = folds[i]
        train_idx = np.concatenate(folds[:i] + folds[i+1:])
        result.append((train_idx.tolist(), test_idx.tolist()))
    
    return result
```

### Verified Sample Trace Data
```json
{
  "test": "import numpy as np\nnp.random.seed(42)\nprint(k_fold_cross_validation(np.array([0,1,2,3,4,5,6,7,8,9]), np.array([0,1,2,3,4,5,6,7,8,9]), k=5, shuffle=False))",
  "expected_output": "[([2, 3, 4, 5, 6, 7, 8, 9], [0, 1]), ([0, 1, 4, 5, 6, 7, 8, 9], [2, 3]), ([0, 1, 2, 3, 6, 7, 8, 9], [4, 5]), ([0, 1, 2, 3, 4, 5, 8, 9], [6, 7]), ([0, 1, 2, 3, 4, 5, 6, 7], [8, 9])]"
}
```

---

## Problem 19: Principal Component Analysis (PCA) Implementation
- **Difficulty**: `medium`
- **Category**: Machine Learning

### Description & Spec
Write a Python function that performs Principal Component Analysis (PCA) from scratch. The function should take a 2D NumPy array as input, where each row represents a data sample and each column represents a feature. The function should standardize the dataset, compute the covariance matrix, find the eigenvalues and eigenvectors, and return the principal components (the eigenvectors corresponding to the largest eigenvalues). The function should also take an integer k as input, representing the number of principal components to return.

### Mathematical Derivation & Intuition
## Understanding Eigenvalues in PCA

Principal Component Analysis (PCA) utilizes the concept of eigenvalues and eigenvectors to identify the principal components of a dataset. Here's how eigenvalues fit into the PCA process:

### Eigenvalues and Eigenvectors: The Foundation of PCA
For a given square matrix \( A \), representing the covariance matrix in PCA, eigenvalues \( \lambda \) and their corresponding eigenvectors \( v \) satisfy:
$$
Av = \lambda v
$$

### Calculating Eigenvalues
The eigenvalues of matrix \( A \) are found by solving the characteristic equation:
$$
\det(A - \lambda I) = 0
$$
where \( I \) is the identity matrix of the same dimension as \( A \). This equation highlights the relationship between a matrix, its eigenvalues, and eigenvectors.

### Role in PCA
In PCA, the covariance matrix's eigenvalues represent the variance explained by its eigenvectors. Thus, selecting the eigenvectors associated with the largest eigenvalues is akin to choosing the principal components that retain the most data variance.

### Eigenvalues and Dimensionality Reduction
The magnitude of an eigenvalue correlates with the importance of its corresponding eigenvector (principal component) in representing the dataset's variability. By selecting a subset of eigenvectors corresponding to the largest eigenvalues, PCA achieves dimensionality reduction while preserving as much of the dataset's variability as possible.

### Practical Application
1. **Standardize the Dataset**: Ensure that each feature has a mean of 0 and a standard deviation of 1.
2. **Compute the Covariance Matrix**: Reflects how features vary together.
3. **Find Eigenvalues and Eigenvectors**: Solve the characteristic equation for the covariance matrix.
4. **Select Principal Components**: Choose eigenvectors (components) with the highest eigenvalues for dimensionality reduction.

Through this process, PCA transforms the original features into a new set of uncorrelated features (principal components), ordered by the amount of original variance they explain.

### Verified Pure NumPy Solution (From Scratch)
```python
import numpy as np

def pca(data, k):
    # Standardize the data
    data_standardized = (data - np.mean(data, axis=0)) / np.std(data, axis=0)
    
    # Compute the covariance matrix
    covariance_matrix = np.cov(data_standardized, rowvar=False)
    
    # Eigen decomposition
    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
    
    # Sort the eigenvectors by decreasing eigenvalues
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues_sorted = eigenvalues[idx]
    eigenvectors_sorted = eigenvectors[:,idx]
    
    # Select the top k eigenvectors (principal components)
    principal_components = eigenvectors_sorted[:, :k]
    
    return np.round(principal_components, 4)
```

### Verified Sample Trace Data
```json
{
  "test": "print(pca(np.array([[4,2,1],[5,6,7],[9,12,1],[4,6,7]]),2))",
  "expected_output": "[[0.6855, 0.0776], [0.6202, 0.4586], [-0.3814, 0.8853]]"
}
```

---

## Problem 20: Decision Tree Learning
- **Difficulty**: `hard`
- **Category**: Machine Learning

### Description & Spec
Write a Python function that implements the decision tree learning algorithm for classification. The function should use recursive binary splitting based on entropy and information gain to build a decision tree. It should take a list of examples (each example is a dict of attribute-value pairs) and a list of attribute names as input, and return a nested dictionary representing the decision tree.

### Mathematical Derivation & Intuition
## Decision Tree Learning Algorithm

The decision tree learning algorithm is a method used for classification that predicts the value of a target variable based on several input variables. Each internal node of the tree corresponds to an input variable, and each leaf node corresponds to a class label.

### Algorithm Overview
The recursive binary splitting starts by selecting the attribute that best separates the examples according to the entropy and information gain, calculated as follows:

### Entropy
$$
H(X) = -\sum p(x) \log_2 p(x)
$$

### Information Gain
$$
IG(D, A) = H(D) - \sum \frac{|D_v|}{|D|} H(D_v)
$$

### Explanation of Terms
- **Entropy \( H(X) \)**: Measures the impurity or disorder of the set.
- **Information Gain \( IG(D, A) \)**: Represents the reduction in entropy after splitting the dataset \( D \) on attribute \( A \).
- **\( D_v \)**: The subset of \( D \) for which attribute \( A \) has value \( v \).

### Process
1. **Select Attribute**: Choose the attribute with the highest information gain.
2. **Split Dataset**: Divide the dataset based on the values of the selected attribute.
3. **Recursion**: Repeat the process for each subset until:
   - All data is perfectly classified, or
   - No remaining attributes can be used to make a split.

This recursive process continues until the decision tree can no longer be split further or all examples have been classified.

### Verified Pure NumPy Solution (From Scratch)
```python
import math
from collections import Counter

def calculate_entropy(labels):
    label_counts = Counter(labels)
    total_count = len(labels)
    entropy = -sum((count / total_count) * math.log2(count / total_count) for count in label_counts.values())
    return entropy

def calculate_information_gain(examples, attr, target_attr):
    total_entropy = calculate_entropy([example[target_attr] for example in examples])
    values = set(example[attr] for example in examples)
    attr_entropy = 0
    for value in values:
        value_subset = [example[target_attr] for example in examples if example[attr] == value]
        value_entropy = calculate_entropy(value_subset)
        attr_entropy += (len(value_subset) / len(examples)) * value_entropy
    return total_entropy - attr_entropy

def majority_class(examples, target_attr):
    return Counter([example[target_attr] for example in examples]).most_common(1)[0][0]

def learn_decision_tree(examples, attributes, target_attr):
    if not examples:
        return 'No examples'
    if all(example[target_attr] == examples[0][target_attr] for example in examples):
        return examples[0][target_attr]
    if not attributes:
        return majority_class(examples, target_attr)
    
    gains = {attr: calculate_information_gain(examples, attr, target_attr) for attr in attributes}
    best_attr = max(gains, key=gains.get)
    tree = {best_attr: {}}
    
    for value in set(example[best_attr] for example in examples):
        subset = [example for example in examples if example[best_attr] == value]
        new_attributes = attributes.copy()
        new_attributes.remove(best_attr)
        subtree = learn_decision_tree(subset, new_attributes, target_attr)
        tree[best_attr][value] = subtree
    
    return tree
```

### Verified Sample Trace Data
```json
{
  "test": "print(learn_decision_tree([\n    {'Outlook': 'Sunny', 'Wind': 'Weak', 'PlayTennis': 'No'},\n    {'Outlook': 'Overcast', 'Wind': 'Strong', 'PlayTennis': 'Yes'},\n    {'Outlook': 'Rain', 'Wind': 'Weak', 'PlayTennis': 'Yes'},\n    {'Outlook': 'Sunny', 'Wind': 'Strong', 'PlayTennis': 'No'},\n    {'Outlook': 'Sunny', 'Wind': 'Weak', 'PlayTennis': 'Yes'},\n    {'Outlook': 'Overcast', 'Wind': 'Weak', 'PlayTennis': 'Yes'},\n    {'Outlook': 'Rain', 'Wind': 'Strong', 'PlayTennis': 'No'},\n    {'Outlook': 'Rain', 'Wind': 'Weak', 'PlayTennis': 'Yes'}\n], ['Outlook', 'Wind'], 'PlayTennis'))",
  "expected_output": "{'Outlook': {'Sunny': {'Wind': {'Weak': 'No', 'Strong': 'No'}}, 'Rain': {'Wind': {'Weak': 'Yes', 'Strong': 'No'}}, 'Overcast': 'Yes'}}"
}
```

---

## Problem 21: Pegasos Kernel SVM Implementation
- **Difficulty**: `hard`
- **Category**: Machine Learning

### Description & Spec
Write a Python function that implements a **deterministic** version of the Pegasos algorithm to train a kernel SVM classifier from scratch. The function should take a dataset (as a 2D NumPy array where each row represents a data sample and each column represents a feature), a label vector (1D NumPy array where each entry corresponds to the label of the sample), and training parameters such as the choice of kernel (linear or RBF), regularization parameter (lambda), and the number of iterations. Note that while the original Pegasos algorithm is stochastic (it selects a single random sample at each step), **this problem requires using all samples in every iteration** (i.e., **no random sampling**). The function should perform binary classification and return the model's alpha coefficients and bias.

### Mathematical Derivation & Intuition
## Pegasos Algorithm for Kernel SVM (Deterministic Version)

### Introduction

The **Pegasos Algorithm** (Primal Estimated sub-GrAdient SOlver for SVM) is a fast, iterative algorithm designed to train Support Vector Machines (SVM). While the original Pegasos algorithm uses stochastic updates by selecting one random sample per iteration, this problem requires a **deterministic version**meaning **every data sample is evaluated and considered in each iteration**. This deterministic approach ensures reproducibility and clarity, particularly useful for educational purposes.

---

### Key Concepts

**Kernel Trick**:  
SVM typically separates data classes using a linear hyperplane. However, real-world data isn't always linearly separable. The **Kernel Trick** implicitly maps input data into a higher-dimensional feature space, making it easier to separate non-linear data.

Common kernel functions include:
- **Linear Kernel**: $K(x,y) = x \cdot y$
- **Radial Basis Function (RBF) Kernel**: $K(x,y) = e^{-\frac{\|x-y\|^2}{2\sigma^2}}$

**Regularization Parameter ($\lambda$)**:  
This parameter balances how closely the model fits training data against the complexity of the model, helping to prevent overfitting.

**Sub-gradient Descent**:  
Pegasos optimizes the SVM objective function using iterative parameter updates based on the sub-gradient of the hinge loss.

---

### Deterministic Pegasos Algorithm Steps

Given training samples $(x_i, y_i)$, labels $y_i \in \{-1, 1\}$, kernel function $K$, regularization parameter $\lambda$, and total iterations $T$:

1. **Initialize** alpha coefficients $\alpha_i = 0$ and bias $b = 0$.
2. For each iteration $t = 1, 2, \dots, T$:
    - Compute learning rate: $$\eta_t = \frac{1}{\lambda t}$$
    - For each training sample $(x_i, y_i)$:
        - Compute decision value:
        $$f(x_i) = \sum_{j}\alpha_j y_j K(x_j, x_i) + b$$
        - If the margin constraint $y_i f(x_i) < 1$ is violated, update parameters:
        $$
        \alpha_i \leftarrow \alpha_i + \eta_t(y_i - \lambda \alpha_i)
        $$
        $$
        b \leftarrow b + \eta_t y_i
        $$

---

### Example (Conceptual Explanation)

Consider a simple dataset:

- **Data**:  
$X = [[1,2],[2,3],[3,1],[4,1]]$, $Y = [1,1,-1,-1]$

- **Parameters**: Linear kernel, $\lambda = 0.01$, iterations = $1$

Initially, parameters ($\alpha, b$) start at zero. For each sample, you calculate the decision value. Whenever a sample violates the margin constraint ($y_i f(x_i) < 1$), you update the corresponding $\alpha_i$ and bias $b$ as described. After looping through all samples for the specified iterations, you obtain the trained parameters.

---

### Important Implementation Notes:
- Always iterate through **all samples** in every iteration (**no stochastic/random sampling**).
- Clearly distinguish kernel function choices in your implementation.
- After training, predictions for new data $x$ are made using:
$$
\hat{y}(x) = \text{sign}\left(\sum_{j}\alpha_j y_j K(x_j, x) + b\right)
$$

This deterministic Pegasos variant clearly demonstrates how kernelized SVM training operates and simplifies the understanding of kernel methods.

### Verified Pure NumPy Solution (From Scratch)
```python
import numpy as np

def linear_kernel(x, y):
    return np.dot(x, y)

def rbf_kernel(x, y, sigma=1.0):
    return np.exp(-np.linalg.norm(x-y)**2 / (2 * (sigma ** 2)))

def pegasos_kernel_svm(data, labels, kernel='linear', lambda_val=0.01, iterations=100, sigma=1.0):
    n_samples = len(data)
    alphas = np.zeros(n_samples)
    b = 0

    for t in range(1, iterations + 1):
        for i in range(n_samples):
            eta = 1.0 / (lambda_val * t)
            if kernel == 'linear':
                kernel_func = linear_kernel
            elif kernel == 'rbf':
                kernel_func = lambda x, y: rbf_kernel(x, y, sigma)
    
            decision = sum(alphas[j] * labels[j] * kernel_func(data[j], data[i]) for j in range(n_samples)) + b
            if labels[i] * decision < 1:
                alphas[i] += eta * (labels[i] - lambda_val * alphas[i])
                b += eta * labels[i]

    return np.round(alphas,4).tolist(), np.round(b,4)
```

### Verified Sample Trace Data
```json
{
  "test": "print(pegasos_kernel_svm(np.array([[1, 2], [2, 3], [3, 1], [4, 1]]), np.array([1, 1, -1, -1]), kernel='linear', lambda_val=0.01, iterations=100))",
  "expected_output": "([100.0, 0.0, -100.0, -100.0], -937.4755)"
}
```

---

## Problem 29: Random Shuffle of Dataset
- **Difficulty**: `easy`
- **Category**: Machine Learning

### Description & Spec
Write a Python function to perform a random shuffle of the samples in two numpy arrays, X and y, while maintaining the corresponding order between them. The function should have an optional seed parameter for reproducibility.

### Mathematical Derivation & Intuition
## Understanding Dataset Shuffling

Random shuffling of a dataset is a common preprocessing step in machine learning to ensure that the data is randomly distributed before training a model. This helps to avoid any potential biases that may arise from the order in which data is presented to the model.

### Step-by-Step Method to Shuffle a Dataset

1. **Generate a Random Index Array**  
   Create an array of indices corresponding to the number of samples in the dataset.

2. **Shuffle the Indices**  
   Use a random number generator to shuffle the array of indices.

3. **Reorder the Dataset**  
   Use the shuffled indices to reorder the samples in both \( X \) and \( y \).

### Key Point
This method ensures that the correspondence between \( X \) and \( y \) is maintained after shuffling, preserving the relationship between features and labels.

### Verified Pure NumPy Solution (From Scratch)
```python
import numpy as np

def shuffle_data(X, y, seed=None):
    if seed:
        np.random.seed(seed)
    idx = np.arange(X.shape[0])
    np.random.shuffle(idx)
    return X[idx], y[idx]
```

### Verified Sample Trace Data
```json
{
  "test": "print(shuffle_data(np.array([[1, 2], [3, 4], [5, 6], [7, 8]]), np.array([1, 2, 3, 4]), seed=42))",
  "expected_output": "(array([[3, 4], [7, 8], [1, 2], [5, 6]]), array([2, 4, 1, 3]))"
}
```

---

## Problem 30: Batch Iterator for Dataset
- **Difficulty**: `easy`
- **Category**: Machine Learning

### Description & Spec
Implement a batch iterable function that samples in a numpy array X and an optional numpy array y. The function should return batches of a specified size. If y is provided, the function should return batches of (X, y) pairs; otherwise, it should return batches of X only.

### Mathematical Derivation & Intuition
## Understanding Batch Iteration

Batch iteration is a common technique used in machine learning and data processing to handle large datasets more efficiently. Instead of processing the entire dataset at once, which can be memory-intensive, data is processed in smaller, more manageable batches.

### Step-by-Step Method to Create a Batch Iterator

1. **Determine the Number of Samples**  
   Calculate the total number of samples in the dataset.

2. **Iterate in Batches**  
   Loop through the dataset in increments of the specified batch size.

3. **Yield Batches**  
   For each iteration, yield a batch of samples from \( X \) and, if provided, the corresponding samples from \( y \).

### Key Point
This method ensures efficient processing and can be used for both the training and evaluation phases in machine learning workflows.

### Verified Pure NumPy Solution (From Scratch)
```python
import numpy as np

def batch_iterator(X, y=None, batch_size=64):
    n_samples = X.shape[0]
    batches = []
    for i in np.arange(0, n_samples, batch_size):
        begin, end = i, min(i+batch_size, n_samples)
        if y is not None:
            batches.append([X[begin:end], y[begin:end]])
        else:
            batches.append( X[begin:end])
    return batches
```

### Verified Sample Trace Data
```json
{
  "test": "print(batch_iterator(np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]), np.array([1, 2, 3, 4, 5]), batch_size=2))",
  "expected_output": "[[[[1, 2], [3, 4]], [1, 2]], [[[5, 6], [7, 8]], [3, 4]], [[[9, 10]], [5]]]"
}
```

---

## Problem 31: Divide Dataset Based on Feature Threshold
- **Difficulty**: `medium`
- **Category**: Machine Learning

### Description & Spec
Write a Python function to divide a dataset based on whether the value of a specified feature is greater than or equal to a given threshold. The function should return two subsets of the dataset: one with samples that meet the condition and another with samples that do not.

### Mathematical Derivation & Intuition
## Understanding Dataset Division Based on Feature Threshold

Dividing a dataset based on a feature threshold is a common operation in machine learning, especially in decision tree algorithms. This technique helps in creating splits that can be used for further processing or model training.

### Problem Overview
In this problem, you will write a function to split a dataset based on whether the value of a specified feature is greater than or equal to a given threshold. You'll need to create two subsets:
- One for samples that meet the condition (values greater than or equal to the threshold).
- Another for samples that do not meet the condition.

### Importance
This method is crucial for algorithms that rely on data partitioning, such as decision trees and random forests. By splitting the data, the model can create rules to make predictions based on the threshold values of certain features.

### Verified Pure NumPy Solution (From Scratch)
```python
import numpy as np

def divide_on_feature(X, feature_i, threshold):
    # Define the split function based on the threshold type
    split_func = None
    if isinstance(threshold, int) or isinstance(threshold, float):
        # For numeric threshold, check if feature value is greater than or equal to the threshold
        split_func = lambda sample: sample[feature_i] >= threshold
    else:
        # For non-numeric threshold, check if feature value is equal to the threshold
        split_func = lambda sample: sample[feature_i] == threshold

    # Create two subsets based on the split function
    X_1 = np.array([sample for sample in X if split_func(sample)])
    X_2 = np.array([sample for sample in X if not split_func(sample)])

    # Return the two subsets
    return [X_1, X_2]
```

### Verified Sample Trace Data
```json
{
  "test": "print(divide_on_feature(np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]), 0, 5))",
  "expected_output": "[array([[ 5,  6], [ 7,  8], [ 9, 10]]), array([[1, 2], [3, 4]])]"
}
```

---

## Problem 32: Generate Sorted Polynomial Features
- **Difficulty**: `medium`
- **Category**: Machine Learning

### Description & Spec
Write a Python function that takes a 2-D NumPy array **X** and an integer **degree**, generates all polynomial feature combinations of the columns of **X** up to the given degree **inclusive**, **then sorts the resulting features for each sample from lowest to highest value**. The function should return a new 2-D NumPy array whose rows correspond to the input samples and whose columns are the **ascending-sorted** polynomial features.

### Mathematical Derivation & Intuition
## Understanding Polynomial Features

Generating polynomial features is a method used to create new features for a machine-learning model by raising existing features to a specified power. This technique helps capture non-linear relationships between features.

### Example
Given a dataset with two features $x_1$ and $x_2$, generating polynomial features up to degree 2 will create new features such as:
- $x_1^2$
- $x_2^2$
- $x_1 x_2$

### Problem Overview
In this problem you will write a function to **generate** polynomial features **and then sort each sample's features in ascending order**. Specifically:
- Given a 2-D NumPy array **X** and an integer **degree**, create a new 2-D array with all polynomial combinations of the features up to the specified degree.
- Finally, sort each row from the lowest value to the highest value.

### Importance
Polynomial expansion allows otherwise linear models to handle non-linear data. Sorting the expanded features can be useful for certain downstream tasks (e.g., histogram-based models or feature selection heuristics) and reinforces array-manipulation skills in NumPy.

### Verified Pure NumPy Solution (From Scratch)
```python
import numpy as np
from itertools import combinations_with_replacement

def polynomial_features(X, degree):
    n_samples, n_features = X.shape

    # All index combinations for powers 0 … degree (constant term included)
    combs = [c for d in range(degree + 1)
             for c in combinations_with_replacement(range(n_features), d)]

    # Compute raw polynomial terms
    X_poly = np.empty((n_samples, len(combs)))
    for i, idx in enumerate(combs):
        X_poly[:, i] = 1 if len(idx) == 0 else np.prod(X[:, idx], axis=1)

    # Sort each row from lowest → highest
    X_sorted = np.sort(X_poly, axis=1)
    return X_sorted
```

### Verified Sample Trace Data
```json
{
  "test": "print(polynomial_features(np.array([[2, 3], [3, 4], [5, 6]]), 2))",
  "expected_output": "[[ 1.  2.  3.  4.  6.  9.]\n [ 1.  3.  4.  9. 12. 16.]\n [ 1.  5.  6. 25. 30. 36.]]"
}
```

---

## Problem 33: Generate Random Subsets of a Dataset
- **Difficulty**: `medium`
- **Category**: Machine Learning

### Description & Spec
Write a Python function to generate random subsets of a given dataset. The function should take in a 2D numpy array X, a 1D numpy array y, an integer n_subsets, and a boolean replacements. It should return a list of n_subsets random subsets of the dataset, where each subset is a tuple of (X_subset, y_subset). If replacements is True, the subsets should be created with replacements; otherwise, without replacements.

### Mathematical Derivation & Intuition
## Understanding Random Subsets of a Dataset

Generating random subsets of a dataset is a useful technique in machine learning, particularly in ensemble methods like bagging and random forests. By creating random subsets, models can be trained on different parts of the data, which helps in reducing overfitting and improving generalization.

### Problem Overview
In this problem, you will write a function to generate random subsets of a given dataset. Specifically:
- Given a 2D numpy array $X$, a 1D numpy array $y$, an integer `n_subsets`, and a boolean `replacements`, the function will create a list of `n_subsets` random subsets.
- Each subset will be a tuple of $(X_{\text{subset}}, y_{\text{subset}})$.

### Parameters
- **$X$**: A 2D numpy array representing the features.
- **$y$**: A 1D numpy array representing the labels.
- **$n_{\text{subsets}}$**: The number of random subsets to generate.
- **`replacements`**: A boolean indicating whether to sample with or without replacement.
  - If `replacements` is **True**, subsets will be created *with* replacements, meaning samples can be repeated within a subset.
  - If `replacements` is **False**, subsets will be created *without* replacements, meaning samples cannot be repeated within a subset.

### Importance
By understanding and implementing this technique, you can enhance the performance of your models through methods like bootstrapping and ensemble learning.

### Verified Pure NumPy Solution (From Scratch)
```python
import numpy as np

def get_random_subsets(X, y, n_subsets, replacements=True, seed=42):
    np.random.seed(seed)

    n, m = X.shape
    
    subset_size = n if replacements else n // 2
    idx = np.array([np.random.choice(n, subset_size, replace=replacements) for _ in range(n_subsets)])
    # convert all ndarrays to lists
    return [(X[idx][i].tolist(), y[idx][i].tolist()) for i in range(n_subsets)]
```

### Verified Sample Trace Data
```json
{
  "test": " \nX = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])\ny = np.array([1, 2, 3, 4, 5])\nprint(get_random_subsets(X,y, 3, False, seed=42))",
  "expected_output": "[[[3, 4], [9, 10]], [2, 5], [[7, 8], [3, 4]], [4, 2], [[3, 4], [1, 2]], [2, 1]]"
}
```

---

## Problem 34: One-Hot Encoding of Nominal Values
- **Difficulty**: `easy`
- **Category**: Machine Learning

### Description & Spec
Write a Python function to perform one-hot encoding of nominal values. The function should take in a 1D numpy array x of integer values and an optional integer n_col representing the number of columns for the one-hot encoded array. If n_col is not provided, it should be automatically determined from the input array.

### Mathematical Derivation & Intuition
## Understanding One-Hot Encoding

One-hot encoding is a method used to represent categorical variables as binary vectors. This technique is useful in machine learning when dealing with categorical data that has no ordinal relationship.

### Explanation
In one-hot encoding, each category is represented by a binary vector with a length equal to the number of categories. The vector has a value of 1 at the index corresponding to the category and 0 at all other indices.

### Example
For instance, if you have three categories: 0, 1, and 2, the one-hot encoded vectors would be:
- **0**: $[1, 0, 0]$
- **1**: $[0, 1, 0]$
- **2**: $[0, 0, 1]$

This method ensures that the model does not assume any ordinal relationship between categories, which is crucial for many machine learning algorithms.

### Mathematical Representation
The one-hot encoding process can be mathematically represented as follows:

Given a category $x_i$ from a set of categories $\{0, 1, \ldots, n-1\}$, the one-hot encoded vector $\mathbf{v}$ is:
$$
\mathbf{v}_i = 
\begin{cases} 
1 & \text{if } i = x_i \\
0 & \text{otherwise}
\end{cases}
$$

This vector $\mathbf{v}$ will have a length equal to the number of unique categories.

### Verified Pure NumPy Solution (From Scratch)
```python
import numpy as np

def to_categorical(x, n_col=None):
    # One-hot encoding of nominal values
    # If n_col is not provided, determine the number of columns from the input array
    if not n_col:
        n_col = np.amax(x) + 1
    # Initialize a matrix of zeros with shape (number of samples, n_col)
    one_hot = np.zeros((x.shape[0], n_col))
    # Set the appropriate elements to 1
    one_hot[np.arange(x.shape[0]), x] = 1
    return one_hot
```

### Verified Sample Trace Data
```json
{
  "test": "print(to_categorical(np.array([0, 1, 2, 1, 0])))",
  "expected_output": "[[1., 0., 0.], [0., 1., 0.], [0., 0., 1.], [0., 1., 0.], [1., 0., 0.]]"
}
```

---

## Problem 36: Calculate Accuracy Score
- **Difficulty**: `easy`
- **Category**: Machine Learning

### Description & Spec
Write a Python function to calculate the accuracy score of a model's predictions. The function should take in two 1D numpy arrays: y_true, which contains the true labels, and y_pred, which contains the predicted labels. It should return the accuracy score as a float.

### Mathematical Derivation & Intuition
## Understanding Accuracy Score

Accuracy is a metric used to evaluate the performance of a classification model. It is defined as the ratio of the number of correct predictions to the total number of predictions made. Mathematically, accuracy is given by:
$$
\text{Accuracy} = \frac{\text{Number of Correct Predictions}}{\text{Total Number of Predictions}}
$$

### Problem Overview
In this problem, you will write a function to calculate the accuracy score given the true labels and the predicted labels. The function will compare the two arrays and compute the accuracy as the proportion of matching elements.

### Importance
Accuracy is a straightforward and commonly used metric for classification tasks. It provides a quick way to understand how well a model is performing, but it may not always be the best metric, especially for imbalanced datasets.

### Verified Pure NumPy Solution (From Scratch)
```python
import numpy as np

def accuracy_score(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred, axis=0) / len(y_true)
    return accuracy
```

### Verified Sample Trace Data
```json
{
  "test": "print(accuracy_score(np.array([1, 0, 1, 1, 0, 1]), np.array([1, 0, 0, 1, 0, 1])))",
  "expected_output": "0.8333333333333334"
}
```

---

## Problem 38: Implement AdaBoost Fit Method
- **Difficulty**: `hard`
- **Category**: Machine Learning

### Description & Spec
Write a Python function `adaboost_fit` that implements the fit method for an AdaBoost classifier. The function should take in a 2D numpy array `X` of shape `(n_samples, n_features)` representing the dataset, a 1D numpy array `y` of shape `(n_samples,)` representing the labels, and an integer `n_clf` representing the number of classifiers. The function should initialize sample weights, find the best thresholds for each feature, calculate the error, update weights, and return a list of classifiers with their parameters.

### Mathematical Derivation & Intuition
## Understanding AdaBoost

AdaBoost, short for Adaptive Boosting, is an ensemble learning method that combines multiple weak classifiers to create a strong classifier. The basic idea is to fit a sequence of weak learners on weighted versions of the data.

### Implementing the Fit Method for an AdaBoost Classifier

1. **Initialize Weights**  
   Start by initializing the sample weights uniformly:
   $$
   w_i = \frac{1}{N}, \text{ where } N \text{ is the number of samples}
   $$

2. **Iterate Through Classifiers**  
   For each classifier, determine the best threshold for each feature to minimize the error.

3. **Calculate Error and Flip Polarity**  
   If the error is greater than 0.5, flip the polarity:
   $$
   \text{error} = \sum_{i=1}^N w_i [y_i \neq h(x_i)]
   $$
   $$
   \text{if error} > 0.5: \text{error} = 1 - \text{error}, \text{ and flip the polarity}
   $$

4. **Calculate Alpha**  
   Compute the weight (alpha) of the classifier based on its error rate:
   $$
   \alpha = \frac{1}{2} \ln \left( \frac{1 - \text{error}}{\text{error} + 1e-10} \right)
   $$

5. **Update Weights**  
   Adjust the sample weights based on the classifier's performance and normalize them:
   $$
   w_i = w_i \exp(-\alpha y_i h(x_i))
   $$
   $$
   w_i = \frac{w_i}{\sum_{j=1}^N w_j}
   $$

6. **Save Classifier**  
   Store the classifier with its parameters.

### Key Insight
This method helps in focusing more on the misclassified samples in subsequent rounds, thereby improving the overall performance.

### Verified Pure NumPy Solution (From Scratch)
```python
import math
import numpy as np
def adaboost_fit(X, y, n_clf):
    n_samples, n_features = np.shape(X)
    w = np.full(n_samples, (1 / n_samples))
    clfs = []
    
    for _ in range(n_clf):
        clf = {}
        min_error = float('inf')
        
        for feature_i in range(n_features):
            feature_values = np.expand_dims(X[:, feature_i], axis=1)
            unique_values = np.unique(feature_values)
            
            for threshold in unique_values:
                p = 1
                prediction = np.ones(np.shape(y))
                prediction[X[:, feature_i] < threshold] = -1
                error = sum(w[y != prediction])
                
                if error > 0.5:
                    error = 1 - error
                    p = -1
                
                if error < min_error:
                    clf['polarity'] = p
                    clf['threshold'] = threshold
                    clf['feature_index'] = feature_i
                    min_error = error
        
        clf['alpha'] = 0.5 * math.log((1.0 - min_error) / (min_error + 1e-10))
        predictions = np.ones(np.shape(y))
        negative_idx = (X[:, clf['feature_index']] < clf['threshold'])
        if clf['polarity'] == -1:
            negative_idx = np.logical_not(negative_idx)
        predictions[negative_idx] = -1
        w *= np.exp(-clf['alpha'] * y * predictions)
        w /= np.sum(w)
        clfs.append(clf)

    return clfs
```

### Verified Sample Trace Data
```json
{
  "test": "X = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])\ny = np.array([1, 1, -1, -1])\nn_clf = 3\nclfs = adaboost_fit(X, y, n_clf)\nprint(clfs)",
  "expected_output": "[{'polarity': -1, 'threshold': 3, 'feature_index': 0, 'alpha': 11.512925464970229}, {'polarity': -1, 'threshold': 3, 'feature_index': 0, 'alpha': 11.512925464970229}, {'polarity': -1, 'threshold': 3, 'feature_index': 0, 'alpha': 11.512925464970229}]"
}
```

---

## Problem 43: Implement Ridge Regression Loss Function
- **Difficulty**: `easy`
- **Category**: Machine Learning

### Description & Spec
Write a Python function `ridge_loss` that implements the Ridge Regression loss function. The function should take a 2D numpy array `X` representing the feature matrix, a 1D numpy array `w` representing the coefficients, a 1D numpy array `y_true` representing the true labels, and a float `alpha` representing the regularization parameter. The function should return the Ridge loss, which combines the Mean Squared Error (MSE) and a regularization term.

### Mathematical Derivation & Intuition
## Ridge Regression Loss

Ridge Regression is a linear regression method with a regularization term to prevent overfitting by controlling the size of the coefficients.

### Key Concepts:
1. **Regularization**:  
   Adds a penalty to the loss function to discourage large coefficients, helping to generalize the model.

2. **Mean Squared Error (MSE)**:  
   Measures the average squared difference between actual and predicted values.

3. **Penalty Term**:  
   The sum of the squared coefficients, scaled by the regularization parameter $ \lambda $, which controls the strength of the regularization.

### Ridge Loss Function
The Ridge Loss function combines MSE and the penalty term:
$$
L(\beta) = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 + \lambda \sum_{j=1}^{p} \beta_j^2
$$

### Implementation Steps:
1. **Calculate MSE**:  
   Compute the average squared difference between actual and predicted values.

2. **Add Regularization Term**:  
   Compute the sum of squared coefficients multiplied by $ \lambda $.

3. **Combine and Minimize**:  
   Sum MSE and the regularization term to form the Ridge loss, then minimize this loss to find the optimal coefficients.

### Verified Pure NumPy Solution (From Scratch)
```python
import numpy as np

def ridge_loss(X: np.ndarray, w: np.ndarray, y_true: np.ndarray, alpha: float) -> float:
    loss = np.mean((y_true - X @ w)**2) + alpha * np.sum(w**2)
    return loss
```

### Verified Sample Trace Data
```json
{
  "test": "X = np.array([[1,1],[2,1],[3,1],[4,1]])\nW = np.array([.2,2])\ny = np.array([2,3,4,5])\nalpha = 0.1\noutput = ridge_loss(X, W, y, alpha)\nprint(output)",
  "expected_output": "2.204"
}
```

---

## Problem 45: Linear Kernel Function
- **Difficulty**: `easy`
- **Category**: Machine Learning

### Description & Spec
Write a Python function `kernel_function` that computes the linear kernel between two input vectors `x1` and `x2`. The linear kernel is defined as the dot product (inner product) of two vectors.

### Mathematical Derivation & Intuition
## Understanding the Linear Kernel

A kernel function in machine learning is used to measure the similarity between two data points in a higher-dimensional space without having to compute the coordinates of the points in that space explicitly. The **linear kernel** is one of the simplest and most commonly used kernel functions. It computes the dot product (or inner product) of two vectors.

### Mathematical Definition
The linear kernel between two vectors $ \mathbf{x}_1 $ and $ \mathbf{x}_2 $ is mathematically defined as:
$$
K(\mathbf{x}_1, \mathbf{x}_2) = \mathbf{x}_1 \cdot \mathbf{x}_2 = \sum_{i=1}^{n} x_{1,i} \cdot x_{2,i}
$$
where $ n $ is the number of features, and $ x_{1,i} $ and $ x_{2,i} $ are the components of the vectors $ \mathbf{x}_1 $ and $ \mathbf{x}_2 $ respectively.

The linear kernel is widely used in support vector machines (SVMs) and other machine learning algorithms for linear classification and regression tasks. It is computationally efficient and works well when the data is linearly separable.

### Characteristics
- **Simplicity**: The linear kernel is straightforward to implement and compute.
- **Efficiency**: It is computationally less expensive compared to other complex kernels like polynomial or RBF kernels.
- **Interpretability**: The linear kernel is interpretable because it corresponds directly to the dot product, a well-understood operation in vector algebra.

In this problem, you will implement a function capable of computing the linear kernel between two vectors.

### Verified Pure NumPy Solution (From Scratch)
```python
import numpy as np

def kernel_function(x1, x2):
    return np.inner(x1, x2)
```

### Verified Sample Trace Data
```json
{
  "test": "import numpy as np\nx1 = np.array([1, 2, 3])\nx2 = np.array([4, 5, 6])\nresult = kernel_function(x1, x2)\nprint(result)",
  "expected_output": "32"
}
```

---

## Problem 46: Implement Precision Metric
- **Difficulty**: `easy`
- **Category**: Machine Learning

### Description & Spec
Write a Python function `precision` that calculates the precision metric given two numpy arrays: `y_true` and `y_pred`. The `y_true` array contains the true binary labels, and the `y_pred` array contains the predicted binary labels. Precision is defined as the ratio of true positives to the sum of true positives and false positives.

### Mathematical Derivation & Intuition
## Understanding Precision in Classification

Precision is a key metric used in the evaluation of classification models, particularly in binary classification. It provides insight into the accuracy of the positive predictions made by the model.

### Mathematical Definition
Precision is defined as the ratio of true positives (TP) to the sum of true positives and false positives (FP):
$$
\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}
$$

Where:
- **True Positives (TP)**: The number of positive samples that are correctly identified as positive.
- **False Positives (FP)**: The number of negative samples that are incorrectly identified as positive.

### Characteristics of Precision
- **Range**: Precision ranges from 0 to 1, where 1 indicates perfect precision (no false positives) and 0 indicates no true positives.
- **Interpretation**: High precision means that the model has a low false positive rate, meaning it rarely labels negative samples as positive.
- **Use Case**: Precision is particularly useful when the cost of false positives is high, such as in medical diagnosis or fraud detection.

In this problem, you will implement a function to calculate precision given the true labels and predicted labels of a binary classification task.

### Verified Pure NumPy Solution (From Scratch)
```python
import numpy as np

def precision(y_true, y_pred):
    true_positives = np.sum((y_true == 1) & (y_pred == 1))
    false_positives = np.sum((y_true == 0) & (y_pred == 1))
    return true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
```

### Verified Sample Trace Data
```json
{
  "test": "import numpy as np\ny_true = np.array([1, 0, 1, 1, 0, 1])\ny_pred = np.array([1, 0, 1, 0, 0, 1])\nresult = precision(y_true, y_pred)\nprint(result)",
  "expected_output": "1.0"
}
```

---

## Problem 47: Implement Gradient Descent Variants with MSE Loss
- **Difficulty**: `medium`
- **Category**: Machine Learning

### Description & Spec
In this problem, you need to implement a single function that can perform three variants of gradient descent Stochastic Gradient Descent (SGD), Batch Gradient Descent, and Mini Batch Gradient Descent using Mean Squared Error (MSE) as the loss function. The function will take an additional parameter to specify which variant to use. Note: Do not shuffle the data

### Mathematical Derivation & Intuition
## Understanding Gradient Descent Variants with MSE Loss

Gradient Descent is an optimization algorithm used to minimize the cost function in machine learning models, particularly in linear regression and neural networks. The Mean Squared Error (MSE) loss function is commonly used in regression tasks. There are three main types of gradient descent based on how much data is used to compute the gradient at each iteration:

1. **Batch Gradient Descent**:  
   Batch Gradient Descent computes the gradient of the MSE loss function with respect to the parameters for the entire training dataset. It updates the parameters after processing the entire dataset:
   $$
   \theta = \theta - \alpha \cdot \frac{2}{m} \sum_{i=1}^{m} \left( h_\theta(x^{(i)}) - y^{(i)} \right) x^{(i)}
   $$
   where $ \alpha $ is the learning rate, $ m $ is the number of samples, and $ \nabla_{\theta} J(\theta) $ is the gradient of the MSE loss function.

2. **Stochastic Gradient Descent (SGD)**:  
   Stochastic Gradient Descent updates the parameters for each training example individually, making it faster but more noisy:
   $$
   \theta = \theta - \alpha \cdot 2 \left( h_\theta(x^{(i)}) - y^{(i)} \right) x^{(i)}
   $$
   where $ x^{(i)}, y^{(i)} $ are individual training examples.

3. **Mini-Batch Gradient Descent**:  
   Mini-Batch Gradient Descent is a compromise between Batch and Stochastic Gradient Descent. It updates the parameters after processing a small batch of training examples, without shuffling the data:
   $$
   \theta = \theta - \alpha \cdot \frac{2}{b} \sum_{i=1}^{b} \left( h_\theta(x^{(i)}) - y^{(i)} \right) x^{(i)}
   $$
   where $ b $ is the batch size, a subset of the training dataset.

Each method has its advantages: Batch Gradient Descent is more stable but slower, Stochastic Gradient Descent is faster but noisy, and Mini-Batch Gradient Descent strikes a balance between the two.

### Verified Pure NumPy Solution (From Scratch)
```python
import numpy as np

def gradient_descent(X, y, weights, learning_rate, n_iterations, batch_size=1, method='batch'):
    m = len(y)
    
    for _ in range(n_iterations):
        if method == 'batch':
            # Calculate the gradient using all data points
            predictions = X.dot(weights)
            errors = predictions - y
            gradient = 2 * X.T.dot(errors) / m
            weights = weights - learning_rate * gradient
        
        elif method == 'stochastic':
            # Update weights for each data point individually
            for i in range(m):
                prediction = X[i].dot(weights)
                error = prediction - y[i]
                gradient = 2 * X[i].T.dot(error)
                weights = weights - learning_rate * gradient
        
        elif method == 'mini_batch':
            # Update weights using sequential batches of data points without shuffling
            for i in range(0, m, batch_size):
                X_batch = X[i:i+batch_size]
                y_batch = y[i:i+batch_size]
                predictions = X_batch.dot(weights)
                errors = predictions - y_batch
                gradient = 2 * X_batch.T.dot(errors) / batch_size
                weights = weights - learning_rate * gradient
                
    return weights
```

### Verified Sample Trace Data
```json
{
  "test": "import numpy as np\n\nX = np.array([[1, 1], [2, 1], [3, 1], [4, 1]])\ny = np.array([2, 3, 4, 5])\nweights = np.zeros(X.shape[1])\nlearning_rate = 0.01\nn_iterations = 100\n\n# Test Batch Gradient Descent\noutput = gradient_descent(X, y, weights, learning_rate, n_iterations, method='batch')\nprint(output)",
  "expected_output": "[1.14905239 0.56176776]"
}
```

---

## Problem 50: Implement Lasso Regression using Gradient Descent
- **Difficulty**: `medium`
- **Category**: Machine Learning

### Description & Spec
In this problem, you need to implement the Lasso Regression algorithm using Gradient Descent. Lasso Regression (L1 Regularization) adds a penalty equal to the absolute value of the coefficients to the loss function. Your task is to update the weights and bias iteratively using the gradient of the loss function and the L1 penalty.

The objective function of Lasso Regression is:
$$
J(w, b) = \frac{1}{2n} \sum_{i=1}^{n} \left( y_i - \left( \sum_{j=1}^{p} X_{ij} w_j + b \right) \right)^2 + \alpha \sum_{j=1}^{p} | w_j |
$$

Where:
- $ y_i $ is the actual value for the $ i $-th sample
- $ \hat{y}_i = \sum_{j=1}^{p} X_{ij} w_j + b $ is the predicted value for the $ i $-th sample
- $ w_j $ is the weight associated with the $ j $-th feature
- $ \alpha $ is the regularization parameter
- $ b $ is the bias

Your task is to use the L1 penalty to shrink some of the feature coefficients to zero during gradient descent, thereby helping with feature selection.

### Mathematical Derivation & Intuition
## Understanding Lasso Regression and L1 Regularization

Lasso Regression is a type of linear regression that applies L1 regularization to the model. It adds a penalty equal to the sum of the absolute values of the coefficients, encouraging some of them to be exactly zero. This makes Lasso Regression particularly useful for feature selection, as it can shrink the coefficients of less important features to zero, effectively removing them from the model.

### Steps to Implement Lasso Regression using Gradient Descent

1. **Initialize Weights and Bias**:  
   Start with the weights and bias set to zero.

2. **Make Predictions**:  
   Use the formula:
   $$
   \hat{y}_i = \sum_{j=1}^{p} X_{ij} w_j + b
   $$
   where $ \hat{y}_i $ is the predicted value for the $ i $-th sample.

3. **Compute Residuals**:  
   Find the difference between the predicted values $ \hat{y}_i $ and the actual values $ y_i $. These residuals are the errors in the model.

4. **Update the Weights and Bias**:  
   Update the weights and bias using the gradient of the loss function with respect to the weights and bias:

   1. For weights $ w_j $:
      $$
      \frac{\partial J}{\partial w_j} = \frac{1}{n} \sum_{i=1}^{n} X_{ij}(\hat{y}_i - y_i) + \alpha \cdot \text{sign}(w_j)
      $$

   2. For bias $ b $ (without the regularization term):
      $$
      \frac{\partial J}{\partial b} = \frac{1}{n} \sum_{i=1}^{n} (\hat{y}_i - y_i)
      $$

   3. Update the weights and bias:
      $$
      w_j = w_j - \eta \cdot \frac{\partial J}{\partial w_j}
      $$
      $$
      b = b - \eta \cdot \frac{\partial J}{\partial b}
      $$

5. **Check for Convergence**:  
   The algorithm stops when the L1 norm of the gradient with respect to the weights becomes smaller than a predefined threshold $ \text{tol} $:
   $$
   ||\nabla w ||_1 = \sum_{j=1}^{p} \left| \frac{\partial J}{\partial w_j} \right|
   $$

6. **Return the Weights and Bias**:  
   Once the algorithm converges, return the optimized weights and bias.

### Verified Pure NumPy Solution (From Scratch)
```python
import numpy as np

def l1_regularization_gradient_descent(X: np.array, y: np.array, alpha: float = 0.1, learning_rate: float = 0.01, max_iter: int = 1000, tol: float = 1e-4) -> tuple:
    n_samples, n_features = X.shape
    # Zero out weights and bias
    weights = np.zeros(n_features)
    bias = 0
    
    for iteration in range(max_iter):
        # Predict values
        y_pred = np.dot(X, weights) + bias
        # Calculate error
        error = y_pred - y
        # Gradient for weights with L1 penalty
        grad_w = (1 / n_samples) * np.dot(X.T, error) + alpha * np.sign(weights)
        # Gradient for bias (no penalty for bias)
        grad_b = (1 / n_samples) * np.sum(error)
        
        # Update weights and bias
        weights -= learning_rate * grad_w
        bias -= learning_rate * grad_b
        
        # Check for convergence
        if np.linalg.norm(grad_w, ord=1) < tol:
            break
    
    return weights, bias
```

### Verified Sample Trace Data
```json
{
  "test": "import numpy as np\n\nX = np.array([[0, 0], [1, 1], [2, 2]])\ny = np.array([0, 1, 2])\n\nalpha = 0.1\noutput = l1_regularization_gradient_descent(X, y, alpha=alpha, learning_rate=0.01, max_iter=1000)\nprint(output)",
  "expected_output": "(array([0.42371644, 0.42371644]), 0.15385068459377865)"
}
```

---

## Problem 52: Implement Recall Metric in Binary Classification
- **Difficulty**: `easy`
- **Category**: Machine Learning

### Description & Spec
## Task: Implement Recall in Binary Classification

Your task is to implement the **recall** metric in a binary classification setting. Recall is a performance measure that evaluates how effectively a machine learning model identifies positive instances from all the actual positive cases in a dataset.

You need to write a function `recall(y_true, y_pred)` that calculates the recall metric. The function should accept two inputs:

- `y_true`: A list of true binary labels (0 or 1) for the dataset.
- `y_pred`: A list of predicted binary labels (0 or 1) from the model.

Your function should return the recall value rounded to three decimal places. If the denominator (TP + FN) is zero, the recall should be 0.0 to avoid division by zero.

### Mathematical Derivation & Intuition
## Understanding Recall in Classification

Recall is a metric that measures how often a machine learning model correctly identifies positive instances, also known as true positives, from all the actual positive samples in the dataset.

### Mathematical Definition

Recall, also known as sensitivity, is the fraction of relevant instances that were retrieved. It is calculated using the following equation:
$$
\text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}
$$

Where:
1. **True Positives (TP)**: The number of positive samples that are correctly identified as positive.
2. **False Negatives (FN)**: The number of positive samples that are incorrectly identified as negative.

### Task

In this problem, you will implement a function to calculate recall given the true labels and predicted labels of a binary classification task. The results should be rounded to three decimal places.

### Verified Pure NumPy Solution (From Scratch)
```python
import numpy as np

def recall(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    try:
        return round(tp / (tp + fn), 3)
    except ZeroDivisionError:
        return 0.0
```

### Verified Sample Trace Data
```json
{
  "test": "import numpy as np\n\ny_true = np.array([1, 0, 1, 1, 0, 1])\ny_pred = np.array([1, 0, 1, 0, 0, 1])\nprint(recall(y_true, y_pred))",
  "expected_output": "0.75"
}
```

---

## Problem 61: Implement F-Score Calculation for Binary Classification
- **Difficulty**: `easy`
- **Category**: Machine Learning

### Description & Spec
## Task: Implement F-Score Calculation for Binary Classification

Your task is to implement a function that calculates the F-Score for a binary classification task. The F-Score combines both Precision and Recall into a single metric, providing a balanced measure of a model's performance.

Write a function `f_score(y_true, y_pred, beta)` where:

- `y_true`: A numpy array of true labels (binary).
- `y_pred`: A numpy array of predicted labels (binary).
- `beta`: A float value that adjusts the importance of Precision and Recall. When `beta=1`, it computes the F1-Score, a balanced measure of both Precision and Recall.

The function should return the F-Score rounded to three decimal places.

### Mathematical Derivation & Intuition
## Understanding F-Score in Classification

F-Score, also called F-measure, is a measure of predictive performance that's calculated from the Precision and Recall metrics.

### Mathematical Definition

The $F_{\beta}$ score applies additional weights, valuing one of precision or recall more than the other. When $\beta$ equals 1, also known as the **F1-Score**, it symmetrically represents both precision and recall in one metric. The F-Score can be calculated using the following formula:

$$
F_{\beta} = (1 + \beta^2) \times \frac{\text{precision} \times \text{recall}}{(\beta^2 \times \text{precision}) + \text{recall}}
$$

Where:

- **Recall**: The number of true positive results divided by the number of all samples that should have been identified as positive.
- **Precision**: The number of true positive results divided by the number of all samples predicted to be positive, including those not identified correctly.

### Implementation Instructions

In this problem, you will implement a function to calculate the **F-Score** given the true labels, predicted labels, and the Beta value of a binary classification task. The results should be rounded to three decimal places.

#### Special Case:
If the denominator is zero, the F-Score should be set to **0.0** to avoid division by zero.

### Verified Pure NumPy Solution (From Scratch)
```python
import numpy as np

def f_score(y_true, y_pred, beta):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))

    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0

    op = precision * recall
    div = ((beta**2) * precision) + recall

    if div == 0 or op == 0:
        return 0.0

    score = (1 + (beta ** 2)) * op / div
    return round(score, 3)
```

### Verified Sample Trace Data
```json
{
  "test": "import numpy as np\n\ny_true = np.array([1, 0, 1, 1, 0, 1])\ny_pred = np.array([1, 0, 1, 0, 0, 1])\nbeta = 1\nprint(f_score(y_true, y_pred, beta))",
  "expected_output": "0.857"
}
```

---

## Problem 64: Implement Gini Impurity Calculation for a Set of Classes
- **Difficulty**: `easy`
- **Category**: Machine Learning

### Description & Spec
## Task: Implement Gini Impurity Calculation

Your task is to implement a function that calculates the Gini Impurity for a set of classes. Gini impurity is commonly used in decision tree algorithms to measure the impurity or disorder within a node.

### Mathematical Derivation & Intuition
## Understanding Gini Impurity

Gini impurity is a statistical measurement of the impurity or disorder in a list of elements. It is commonly used in decision tree algorithms to decide the optimal split at tree nodes. It is calculated as follows, where $ p_i $ is the probability of each class, $ \frac{n_i}{n} $:

$$
\text{Gini Impurity} = 1 - \sum_{i=1}^{C} p_i^2
$$

A Gini impurity of 0 indicates a node where all elements belong to the same class, whereas a Gini impurity of 1-1/C indicates maximum impurity, where elements are evenly distributed among each class. This means that a lower impurity implies a less homogeneous distribution of elements, suggesting a good split, as decision trees aim to minimize it at each node.

### Advantages and Limitations

#### Advantages:
- Computationally efficient
- Works for binary and multi-class classification

#### Limitations:
- Biased toward larger classes
- May cause overfitting in deep decision trees

### Example Calculation

Suppose we have the set: $[0, 1, 1, 1, 0]$. The probability of each class is calculated as follows:

$$
p_{0} = \frac{2}{5} \quad p_{1} = \frac{3}{5}
$$

The Gini Impurity is then calculated as follows:

$$
\text{Gini Impurity} = 1 - (p_0^2 + p_1^2) = 1 - \left(\left(\frac{2}{5}\right)^2 + \left(\frac{3}{5}\right)^2\right) = 0.48
$$

### Verified Pure NumPy Solution (From Scratch)
```python
import numpy as np

def gini_impurity(y: list[int]) -> float:

    classes = set(y)
    n = len(y)

    gini_impurity = 0

    for cls in classes:
        gini_impurity += (y.count(cls)/n)**2

    return round(1-gini_impurity,3)
```

### Verified Sample Trace Data
```json
{
  "test": "y = [0, 0, 0, 0, 1, 1, 1, 1]\nprint(gini_impurity(y))",
  "expected_output": "0.5"
}
```

---

## Problem 69: Calculate R-squared for Regression Analysis
- **Difficulty**: `easy`
- **Category**: Machine Learning

### Description & Spec
## Task: Compute the R-squared Value in Regression Analysis

- R-squared, also known as the coefficient of determination, is a measure that indicates how well the independent variables explain the variability of the dependent variable in a regression model. 

- **Your Task**: 
    To implement the function `r_squared(y_true, y_pred)` that calculates the R-squared value, given arrays of true values `y_true` and predicted values `y_pred`.

### Mathematical Derivation & Intuition
# Understanding R-squared (R²) in Regression Analysis

R-squared, also known as the coefficient of determination, is a statistical measure that represents the proportion of the variance for a dependent variable that's explained by an independent variable or variables in a regression model. It provides insight into how well the model fits the data.

### Mathematical Definition

The R-squared value is calculated using the following formula:  
$$
R^2 = 1 - \frac{\text{SSR}}{\text{SST}}
$$
Where:  

1) $ \text{SSR} $ (Sum of Squared Residuals): The sum of the squares of the differences between the actual values and the predicted values.  
2) $ \text{SST} $ (Total Sum of Squares): The sum of the squares of the differences between the actual values and the mean of the actual values.

### Equations for SSR and SST

To calculate SSR and SST, we use the following formulas:  

1) SSR:  
$$
\text{SSR} = \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
$$  

2) SST:  
$$
\text{SST} = \sum_{i=1}^{n} (y_i - \bar{y})^2
$$  

Where:  

1) $ y_i $: Actual value  
2) $ \hat{y}_i $: Predicted value  
3) $ \bar{y} $: Mean of the actual values  

### Significance of R-squared

R-squared is a key metric for evaluating how well a regression model performs. A higher R-squared value indicates a better fit for the model, meaning it can explain more variability in the data. However, it's important to note:  

- A high R-squared does not always imply that the model is good; it can sometimes be misleading if overfitting occurs.  
- It should be used in conjunction with other metrics for comprehensive model evaluation.

### Implementing R-squared Calculation

In this problem, you will implement a function to calculate R-squared given arrays of true and predicted values from a regression task. The results should be rounded to three decimal places.  

In the solution, the implemented $ r\_squared() $ function calculates R-squared by first determining SSR and SST, then applying them to compute $ R^2 $. It handles edge cases such as perfect predictions and situations where all true values are identical.

### Reference

You can refer to this resource for more information:  
[Coefficient of Determination](https://www.ncl.ac.uk/webtemplate/ask-assets/external/maths-resources/statistics/regression-and-correlation/coefficient-of-determination-r-squared.html)

### Verified Pure NumPy Solution (From Scratch)
```python
import numpy as np

def r_squared(y_true, y_pred):
    """
    Calculate the R-squared (R²) coefficient of determination.
    
    Args:
        y_true (numpy.ndarray): Array of true values
        y_pred (numpy.ndarray): Array of predicted values
    
    Returns:
        float: R-squared value rounded to 3 decimal places
    """
    if np.array_equal(y_true, y_pred):
        return 1.0

    # Calculate mean of true values
    y_mean = np.mean(y_true)

    # Calculate Sum of Squared Residuals (SSR)
    ssr = np.sum((y_true - y_pred) ** 2)

    # Calculate Total Sum of Squares (SST)
    sst = np.sum((y_true - y_mean) ** 2)

    try:
        # Calculate R-squared
        r2 = 1 - (ssr / sst)
        if np.isinf(r2):
            return 0.0
        return round(r2, 3)
    except ZeroDivisionError:
        return 0.0
```

### Verified Sample Trace Data
```json
{
  "test": "\nimport numpy as np\ny_true = np.array([1, 2, 3, 4, 5])\ny_pred = np.array([1, 2, 3, 4, 5])\nprint(r_squared(y_true, y_pred))\n",
  "expected_output": "1.0"
}
```

---

## Problem 71: Calculate Root Mean Square Error (RMSE)
- **Difficulty**: `easy`
- **Category**: Machine Learning

### Description & Spec
## Task: Compute Root Mean Square Error (RMSE)

In this task, you are required to implement a function `rmse(y_true, y_pred)` that calculates the Root Mean Square Error (RMSE) between the actual values and the predicted values. RMSE is a commonly used metric for evaluating the accuracy of regression models, providing insight into the standard deviation of residuals.

### Your Task:
Implement the function `rmse(y_true, y_pred)` to:
1. Calculate the RMSE between the arrays `y_true` and `y_pred`.
2. Return the RMSE value rounded to three decimal places.
3. Ensure the function handles edge cases such as:
   - Mismatched array shapes.
   - Empty arrays.
   - Invalid input types.

The RMSE is defined as:

$$
\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_{\text{true}, i} - y_{\text{pred}, i})^2}
$$

Where:
- $ n $ is the number of observations.
- $ y_{\text{true}, i} $ and $ y_{\text{pred}, i} $ are the actual and predicted values for the $ i $-th observation.

### Mathematical Derivation & Intuition
## Root Mean Square Error (RMSE)

RMSE is used to measure the accuracy of predictions in regression models. It represents the difference between the predictions and the actual values. In other words, it is the standard deviation of the residuals or prediction errors.

### **Theory**
The RMSE is defined as:

$$
\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_{\text{true}_i} - y_{\text{pred}_i})^2}
$$

where:
- $ n $: The number of observations.
- $ y_{\text{true}_i} $: The actual values.
- $ y_{\text{pred}_i} $: The predicted values.

### **Steps for Calculation**
1. For each pair of actual and predicted values, calculate the difference $ y_{\text{true}_i} - y_{\text{pred}_i} $.
2. Square each of these differences and find their mean.
3. Take the square root of the mean value.

### **When to Use RMSE vs. MAE**
- **RMSE**: Used when large deviations/errors are more problematic and should be penalized more heavily.
- **MAE**: Used when errors should be treated equally, regardless of their size.

### Verified Pure NumPy Solution (From Scratch)
```python
import numpy as np

def rmse(y_true, y_pred):
    if y_true.shape != y_pred.shape:
        raise ValueError("Arrays must have the same shape")
    if y_true.size == 0:
        raise ValueError("Arrays cannot be empty")
    return round(np.sqrt(np.mean((y_true - y_pred) ** 2)), 3)
```

### Verified Sample Trace Data
```json
{
  "test": "\n# Test Case 1: Normal Case  \ny_true1 = np.array([3, -0.5, 2, 7])\ny_pred1 = np.array([2.5, 0.0, 2, 8])\nprint(rmse(y_true1, y_pred1))\n",
  "expected_output": "0.612"
}
```

---

## Problem 72: Calculate Jaccard Index for Binary Classification
- **Difficulty**: `easy`
- **Category**: Machine Learning

### Description & Spec
## Task: Implement the Jaccard Index

Your task is to implement a function `jaccard_index(y_true, y_pred)` that calculates the Jaccard Index, a measure of similarity between two binary sets. The Jaccard Index is widely used in binary classification tasks to evaluate the overlap between predicted and true labels.

### Your Task:
Implement the function `jaccard_index(y_true, y_pred)` to:
1. Calculate the Jaccard Index between the arrays `y_true` and `y_pred`.
2. Return the Jaccard Index as a float value.
3. Ensure the function handles cases where:
   - There is no overlap between `y_true` and `y_pred`.
   - Both arrays contain only zeros (edge cases).

The Jaccard Index is defined as:

$$
\scriptsize
\text{Jaccard Index} = \frac{\text{Number of elements in the intersection of } y_{\text{true}} \text{ and } y_{\text{pred}}}{\text{Number of elements in the union of } y_{\text{true}} \text{ and } y_{\text{pred}}}
$$


Where:
- $ y_{\text{true}} $ and $ y_{\text{pred}} $ are binary arrays of the same length, representing true and predicted labels.
- The result ranges from 0 (no overlap) to 1 (perfect overlap).

### Mathematical Derivation & Intuition
## Understanding Jaccard Index in Classification

The Jaccard Index, also known as the Jaccard Similarity Coefficient, is a statistic used to measure the similarity between sets. In the context of binary classification, it measures the overlap between predicted and actual positive labels.

### Mathematical Definition

The Jaccard Index is defined as the size of the intersection divided by the size of the union of two sets:

$$
\text{Jaccard Index} = \frac{|A \cap B|}{|A \cup B|} = \frac{|A \cap B|}{|A| + |B| - |A \cap B|}
$$

### In the Context of Binary Classification
1. **Intersection ($A \cap B$):** The number of positions where both the predicted and true labels are 1 (True Positives).  
2. **Union ($A \cup B$):** The number of positions where either the predicted or true labels (or both) are 1.  

### Key Properties
1. **Range:** The Jaccard Index always falls between 0 and 1 (inclusive).  
2. **Perfect Match:** A value of 1 indicates identical sets.  
3. **No Overlap:** A value of 0 indicates disjoint sets.  
4. **Symmetry:** The index is symmetric, meaning $J(A, B) = J(B, A)$.  

### Example
Consider two binary vectors:  
- **True labels:** [1, 0, 1, 1, 0, 1]  
- **Predicted labels:** [1, 0, 1, 0, 0, 1]  

In this case:  
1. **Intersection** (positions where both are 1): 3.  
2. **Union** (positions where either is 1): 4.  
3. **Jaccard Index**: $3 / 4 = 0.75$.

### Usage in Machine Learning
The Jaccard Index is particularly useful in:  
1. Evaluating clustering algorithms.  
2. Comparing binary classification results.  
3. Document similarity analysis.  
4. Image segmentation evaluation.  

When implementing the Jaccard Index, it's important to handle edge cases, such as when both sets are empty (in which case the index is typically defined as 0).

### Verified Pure NumPy Solution (From Scratch)
```python
import numpy as np

def jaccard_index(y_true, y_pred):
    intersection = np.sum((y_true == 1) & (y_pred == 1))
    union = np.sum((y_true == 1) | (y_pred == 1))
    result = intersection / union
    if np.isnan(result):
        return 0.0
    return round(result, 3)
```

### Verified Sample Trace Data
```json
{
  "test": "\n# Test case 1: Perfect match\ny_true = np.array([1, 0, 1, 1, 0, 1])\ny_pred = np.array([1, 0, 1, 1, 0, 1])\nprint(jaccard_index(y_true, y_pred))\n",
  "expected_output": "1.0"
}
```

---

## Problem 73: Calculate Dice Score for Classification
- **Difficulty**: `easy`
- **Category**: Machine Learning

### Description & Spec
## Task: Compute the Dice Score

Your task is to implement a function `dice_score(y_true, y_pred)` that calculates the Dice Score, also known as the Sørensen-Dice coefficient or F1-score, for binary classification. The Dice Score is used to measure the similarity between two sets and is particularly useful in tasks like image segmentation and binary classification.

### Your Task:
Implement the function `dice_score(y_true, y_pred)` to:
1. Calculate the Dice Score between the arrays `y_true` and `y_pred`.
2. Return the Dice Score as a float value rounded to 3 decimal places.
3. Handle edge cases appropriately, such as when there are no true or predicted positives.

The Dice Score is defined as:

$$
\scriptsize
\text{Dice Score} =
\frac{2 \times (\text{Number of elements in the intersection of } y_{\text{true}} \text{ and } y_{\text{pred}})}{\text{Number of elements in } y_{\text{true}} + \text{Number of elements in } y_{\text{pred}}}
$$

Where:
- $ y_{\text{true}} $ and $ y_{\text{pred}} $ are binary arrays of the same length, representing true and predicted labels.
- The result ranges from 0 (no overlap) to 1 (perfect overlap).

### Mathematical Derivation & Intuition
# Understanding Dice Score in Classification

The Dice Score, also known as the Sørensen-Dice coefficient or F1-score, is a statistical measure used to gauge the similarity between two samples. It is particularly popular in image segmentation tasks and binary classification problems.

## Mathematical Definition

The Dice coefficient is defined as twice the intersection divided by the sum of the cardinalities of both sets:

$$
\text{Dice Score} = \frac{2|X \cap Y|}{|X| + |Y|} = \frac{2TP}{2TP + FP + FN}
$$

### In terms of binary classification:
1. **TP (True Positives):** Number of positions where both predicted and true labels are 1.  
2. **FP (False Positives):** Number of positions where the prediction is 1 but the true label is 0.  
3. **FN (False Negatives):** Number of positions where the prediction is 0 but the true label is 1.

## Relationship with F1-Score

The Dice coefficient is identical to the F1-score, which is the harmonic mean of precision and recall:

$$
\text{F1-score} = 2 \cdot \frac{\text{precision} \cdot \text{recall}}{\text{precision} + \text{recall}} = \text{Dice Score}
$$

## Key Properties
1. **Range:** The Dice score always falls between 0 and 1 (inclusive).  
2. **Perfect Score:** A value of 1 indicates perfect overlap.  
3. **No Overlap:** A value of 0 indicates no overlap.  
4. **Sensitivity:** More sensitive to overlap than the Jaccard Index.  
5. **Symmetry:** The score is symmetric, meaning DSC(A,B) = DSC(B,A).

## Example

Consider two binary vectors:  
- **True labels:** [1, 1, 0, 1, 0, 1]  
- **Predicted labels:** [1, 1, 0, 0, 0, 1]  

In this case:  
- **True Positives (TP):** 3  
- **False Positives (FP):** 0  
- **False Negatives (FN):** 1  

$$
\text{Dice Score} = \frac{2 \times 3}{2 \times 3 + 0 + 1} = 0.857
$$

## Advantages Over Jaccard Index

The Dice score offers several advantages:  
1. **Higher Sensitivity to Overlap:** Due to the doubled intersection term.  
2. **Weight on Agreement:** Gives more weight to instances where labels agree.  
3. **Preferred in Medical Imaging:** Often used in medical image segmentation due to its sensitivity to overlap.  
4. **Intuitive Interpretation:** As the harmonic mean of precision and recall.

## Common Applications

The Dice score is widely used in:  
1. **Medical image segmentation evaluation.**  
2. **Binary classification tasks.**  
3. **Object detection overlap assessment.**  
4. **Text similarity measurement.**  
5. **Semantic segmentation evaluation.**

When implementing the Dice score, it is important to handle edge cases properly, such as when both sets are empty. In such cases, the score is typically defined as 0.0 (as per scikit-learn).

### Verified Pure NumPy Solution (From Scratch)
```python
import numpy as np

def dice_score(y_true, y_pred):
    intersection = np.logical_and(y_true, y_pred).sum()
    true_sum = y_true.sum()
    pred_sum = y_pred.sum()

    # Handle edge cases
    if true_sum == 0 or pred_sum == 0:
        return 0.0

    dice = (2.0 * intersection) / (true_sum + pred_sum)
    return round(float(dice), 3)
```

### Verified Sample Trace Data
```json
{
  "test": "\ny_true = np.array([1, 1, 0, 0])\ny_pred = np.array([1, 1, 0, 0])\nprint(dice_score(y_true, y_pred))\n",
  "expected_output": "1.0"
}
```

---

## Problem 75: Generate a Confusion Matrix for Binary Classification
- **Difficulty**: `easy`
- **Category**: Machine Learning

### Description & Spec
## Task: Generate a Confusion Matrix

Your task is to implement the function `confusion_matrix(data)` that generates a confusion matrix for a binary classification problem. The confusion matrix provides a summary of the prediction results on a classification problem, allowing you to visualize how many data points were correctly or incorrectly labeled.

### Input:
- A list of lists, where each inner list represents a pair 
- `[y_true, y_pred]` for one observation. `y_true` is the actual label, and `y_pred` is the predicted label.

### Output:
- A $2 \times 2$ confusion matrix represented as a list of lists.

### Mathematical Derivation & Intuition
## Generate Confusion Matrix

The confusion matrix is a very useful tool to get a better understanding of the performance of a classification model. In it, you can visualize how many data points were labeled according to their correct categories.

For a binary classification problem of a dataset with $ n $ observations, the confusion matrix is a $ 2 \times 2 $ matrix with the following structure:

$$
M = \begin{pmatrix} 
TP & FN \\
FP & TN
\end{pmatrix}
$$

Where:

- **TP**: True positives, the number of observations from the positive label that were correctly labeled as positive.
- **FN**: False negatives, the number of observations from the positive label that were incorrectly labeled as negative.
- **FP**: False positives, the number of observations from the negative label that were incorrectly labeled as positive.
- **TN**: True negatives, the number of observations from the negative label that were correctly labeled as negative.

A confusion matrix is a great starting point for computing more advanced metrics such as precision and recall that capture the model's performance.

### Verified Pure NumPy Solution (From Scratch)
```python
from collections import Counter

def confusion_matrix(data):
    # Count all occurrences
    counts = Counter(tuple(pair) for pair in data)
    # Get metrics
    TP, FN, FP, TN = counts[(1, 1)], counts[(1, 0)], counts[(0, 1)], counts[(0, 0)]
    # Define matrix and return
    confusion_matrix = [[TP, FN], [FP, TN]]
    return confusion_matrix
```

### Verified Sample Trace Data
```json
{
  "test": "\ndata = [[1, 1], [1, 0], [0, 1], [0, 0], [0, 1]]\nprint(confusion_matrix(data))\n",
  "expected_output": "[[1, 1], [2, 1]]"
}
```

---

## Problem 77: Calculate Performance Metrics for a Classification Model
- **Difficulty**: `medium`
- **Category**: Machine Learning

### Description & Spec
### Task: Implement Performance Metrics Calculation

In this task, you are required to implement a function `performance_metrics(actual, predicted)` that computes various performance metrics for a binary classification problem. These metrics include:

- Confusion Matrix
- Accuracy
- F1 Score
- Specificity
- Negative Predictive Value

The function should take in two lists:

- `actual`: The actual class labels (1 for positive, 0 for negative).
- `predicted`: The predicted class labels from the model.

### Output

The function should return a tuple containing:

1. `confusion_matrix`: A 2x2 matrix.
2. `accuracy`: A float representing the accuracy of the model.
3. `f1_score`: A float representing the F1 score of the model.
4. `specificity`: A float representing the specificity of the model.
5. `negative_predictive_value`: A float representing the negative predictive value.

### Constraints

- All elements in the `actual` and `predicted` lists must be either 0 or 1.
- Both lists must have the same length.

### Mathematical Derivation & Intuition
## Performance Metrics

Performance metrics such as accuracy, F1 score, specificity, negative predictive value, precision, and recall are vital to understanding how a model is performing.

How many observations are correctly labeled? Are we mislabeling one category more than the other? Performance metrics can answer these questions and provide an idea of where to focus to improve a model's performance.

For this problem, starting with the confusion matrix is a helpful first step, as all the elements of the confusion matrix can help with calculating other performance metrics.

For a binary classification problem of a dataset with $n$ observations, the confusion matrix is a $2 \times 2$ matrix with the following structure:

$$
M = \begin{pmatrix} 
TP & FN \\
FP & TN
\end{pmatrix}
$$

Where:
- **TP**: True positives, the number of observations from the positive label that were correctly labeled as positive.
- **FN**: False negatives, the number of observations from the positive label that were incorrectly labeled as negative.
- **FP**: False positives, the number of observations from the negative label that were incorrectly labeled as positive.
- **TN**: True negatives, the number of observations from the negative label that were correctly labeled as negative.

### Metrics

#### Accuracy
How many observations are labeled as the actual category they belong to?

$$
\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}
$$

#### Precision
How many elements labeled as positive are actually positive?

$$
\text{Precision} = \frac{TP}{TP + FP}
$$

#### Negative Predictive Value
How many elements labeled as negative are actually negative?

$$
\text{Negative Predictive Value} = \frac{TN}{TN + FN}
$$

#### Recall
Out of all positive elements, how many were correctly labeled?

$$
\text{Recall} = \frac{TP}{TP + FN}
$$

#### Specificity
How well are we labeling the negative elements correctly?

$$
\text{Specificity} = \frac{TN}{TN + FP}
$$

#### F1 Score
How to account for the trade-off of false negatives and positives? The F1 score is the harmonic mean of precision and recall.

$$
\text{F1 Score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}
$$

### Verified Pure NumPy Solution (From Scratch)
```python
from collections import Counter

def performance_metrics(actual: list[int], predicted: list[int]) -> tuple:
    data = list(zip(actual, predicted))
    counts = Counter(tuple(pair) for pair in data)
    TP, FN, FP, TN = counts[(1, 1)], counts[(1, 0)], counts[(0, 1)], counts[(0, 0)]
    confusion_matrix = [[TP, FN], [FP, TN]]
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1 = 2 * precision * recall / (precision + recall)
    negativePredictive = TN / (TN + FN)
    specificity = TN / (TN + FP)
    return confusion_matrix, round(accuracy, 3), round(f1, 3), round(specificity, 3), round(negativePredictive, 3)
```

### Verified Sample Trace Data
```json
{
  "test": "\nactual = [1, 0, 1, 0, 1]\npredicted = [1, 0, 0, 1, 1]\nprint(performance_metrics(actual, predicted))\n",
  "expected_output": "([[2, 1], [1, 1]], 0.6, 0.667, 0.5, 0.5)"
}
```

---

## Problem 86: Detect Overfitting or Underfitting
- **Difficulty**: `easy`
- **Category**: Machine Learning

### Description & Spec
Write a Python function to determine whether a machine learning model is overfitting, underfitting, or performing well based on training and test accuracy values. The function should take two inputs: `training_accuracy` and `test_accuracy`. It should return one of three values: 1 if Overfitting, -1 if Underfitting, or 0 if a Good fit. The rules for determination are as follows: 
- **Overfitting**: The training accuracy is significantly higher than the test accuracy (difference > 0.2).
- **Underfitting**: Both training and test accuracy are below 0.7.
- **Good fit**: Neither of the above conditions is true.

### Mathematical Derivation & Intuition
## Understanding Overfitting and Underfitting

Overfitting and underfitting are two common problems in machine learning models that affect their performance and generalization ability.

### Overfitting
Overfitting occurs when a model learns the training data too well, including noise and irrelevant patterns. This results in high training accuracy but poor performance on unseen data (low test accuracy).

- **Indicators**: Training accuracy >> Test accuracy (large gap).

### Underfitting
Underfitting occurs when a model is too simple to capture the underlying patterns in the data. This leads to poor performance on both training and test datasets.

- **Indicators**: Both training and test accuracy are low.

### Good Fit
A good fit occurs when the model generalizes well to unseen data, with training and test accuracy being close and both reasonably high.

### Remedies
- **For Overfitting**:
  - Use regularization techniques (e.g., L1, L2 regularization).
  - Reduce model complexity by pruning unnecessary features.
  - Add more training data to improve generalization.
- **For Underfitting**:
  - Increase model complexity (e.g., add layers or features).
  - Train the model for more epochs.
  - Enhance feature engineering or input data quality.

### Mathematical Representation
1. Overfitting:
   $$ \text{Training Accuracy} - \text{Test Accuracy} > 0.2 $$

2. Underfitting:
   $$ \text{Training Accuracy} < 0.7 \, \text{and} \, \text{Test Accuracy} < 0.7 $$

3. Good Fit:
   $$ \text{Neither overfitting nor underfitting is true.} $$

### Verified Pure NumPy Solution (From Scratch)
```python
def model_fit_quality(training_accuracy, test_accuracy):
    """
    Determine if the model is overfitting, underfitting, or a good fit based on training and test accuracy.
    :param training_accuracy: float, training accuracy of the model (0 <= training_accuracy <= 1)
    :param test_accuracy: float, test accuracy of the model (0 <= test_accuracy <= 1)
    :return: int, one of '1', '-1', or '0'.
    """
    if training_accuracy - test_accuracy > 0.2:
        return 1
    elif training_accuracy < 0.7 and test_accuracy < 0.7:
        return -1
    else:
        return 0
```

### Verified Sample Trace Data
```json
{
  "test": "print(model_fit_quality(0.95, 0.65))",
  "expected_output": "1"
}
```

---

## Problem 91: Calculate F1 Score from Predicted and True Labels
- **Difficulty**: `easy`
- **Category**: Machine Learning

### Description & Spec
Implement a function to calculate the F1 score given predicted and true labels. The F1 score is a widely used metric in machine learning, combining precision and recall into a single measure. round your solution to the 3rd decimal place

### Mathematical Derivation & Intuition
## **F1 Score**

The F1 score is a widely used metric in machine learning and statistics, particularly for evaluating classification models. It is the harmonic mean of **precision** and **recall**, providing a single measure that balances the trade-off between these two metrics.

### **Key Concepts**

1. **Precision**: Precision is the fraction of true positive predictions out of all positive predictions made by the model. It measures how many of the predicted positive instances are actually correct.

    $$
    \text{Precision} = \frac{\text{True Positives (TP)}}{\text{True Positives (TP)} + \text{False Positives (FP)}}
    $$

2. **Recall**: Recall is the fraction of true positive predictions out of all actual positive instances in the dataset. It measures how many of the actual positive instances were correctly predicted.

    $$
    \text{Recall} = \frac{\text{True Positives (TP)}}{\text{True Positives (TP)} + \text{False Negatives (FN)}}
    $$

3. **F1 Score**: The F1 score is the harmonic mean of precision and recall, providing a balanced measure that takes both metrics into account:

    $$
    \text{F1 Score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}
    $$

### **Why Use the F1 Score?**

The F1 score is particularly useful when the dataset is imbalanced, meaning the classes are not equally represented. It provides a single metric that balances the trade-off between precision and recall, especially in scenarios where maximizing one metric might lead to a significant drop in the other.

### **Example Calculation**

Given:
y_true = [1, 0, 1, 1, 0] 


y_pred = [1, 0, 0, 1, 1] 

1. **Calculate True Positives (TP), False Positives (FP), and False Negatives (FN)**:

    $$
    \text{TP} = 2, \quad \text{FP} = 1, \quad \text{FN} = 1
    $$

2. **Calculate Precision**:

    $$
    \text{Precision} = \frac{2}{2 + 1} = \frac{2}{3} \approx 0.667
    $$

3. **Calculate Recall**:

    $$
    \text{Recall} = \frac{2}{2 + 1} = \frac{2}{3} \approx 0.667
    $$

4. **Calculate F1 Score**:

    $$
    \text{F1 Score} = 2 \times \frac{0.667 \times 0.667}{0.667 + 0.667} = 0.667
    $$

### **Applications**

The F1 score is widely used in:
- Binary classification problems (e.g., spam detection, fraud detection).
- Multi-class classification problems (evaluated per class and averaged).
- Information retrieval tasks (e.g., search engines, recommendation systems).

Mastering the F1 score is essential for evaluating and comparing the performance of classification models.

### Verified Pure NumPy Solution (From Scratch)
```python
def calculate_f1_score(y_true, y_pred):
    """
    Calculate the F1 score based on true and predicted labels.

    Args:
        y_true (list): True labels (ground truth).
        y_pred (list): Predicted labels.

    Returns:
        float: The F1 score rounded to three decimal places.
    """
    if len(y_true) != len(y_pred):
        raise ValueError("Lengths of y_true and y_pred must be the same")

    tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp == 1)
    fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)
    fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 0)

    if tp + fp == 0 or tp + fn == 0:
        return 0.0

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    if precision + recall == 0:
        return 0.0

    f1_score = 2 * (precision * recall) / (precision + recall)
    return round(f1_score, 3)
```

### Verified Sample Trace Data
```json
{
  "test": "print(calculate_f1_score([1, 0, 1, 1, 0], [1, 0, 0, 1, 1]))",
  "expected_output": "0.667"
}
```

---

## Problem 92: Linear Regression - Power Grid Optimization
- **Difficulty**: `medium`
- **Category**: Machine Learning

### Description & Spec
It is the year 2157. Mars has its first thriving colony, and energy consumption is steadily on the rise. As the lead data scientist, you have daily power usage measurements (10 days) affected by both a growing linear trend and a daily fluctuation. The fluctuation follows the formula fᵢ = 10 x sin(2π x i / 10), where i is the day number (1 through 10). Your challenge is to remove this known fluctuation from each data point, fit a linear regression model to the detrended data, predict day 15's base consumption, add back the fluctuation for day 15, and finally include a 5% safety margin. The final answer must be an integer, ensuring you meet the colony's future needs.

### Mathematical Derivation & Intuition
## Balancing Trend and Fluctuation with Math

When dealing with time-series data, it's common to see both a long-term trend and periodic fluctuations. In this challenge, the daily fluctuation for day $i$ is given by:

$$
f_i = 10 \times \sin\left(\frac{2\pi \times i}{10}\right).
$$

### Steps to Solve

1. **Fluctuation Removal**: Subtract $f_i$ from each day's consumption to isolate the colony's base usage.

2. **Linear Regression**: Fit a linear model $y = mx + b$ using the detrended values. The slope $m$ and intercept $b$ are calculated using the least squares method:

$$
m = \frac{n \sum(x_i y_i) - (\sum x_i)(\sum y_i)}{n \sum(x_i^2) - (\sum x_i)^2}, \quad b = \frac{\sum y_i - m \sum x_i}{n}.
$$

Here, $n$ is the number of data points (10 in this case).

3. **Forecast**: Use the regression line to predict the base consumption for day 15, $x = 15$:

$$
\text{base}_{15} = m \times 15 + b.
$$

4. **Add Back Fluctuation**: Compute $f_{15} = 10 \times \sin\left(\frac{2\pi \times 15}{10}\right)$ and add it to the base prediction:

$$
\text{pred}_{15} = \text{base}_{15} + f_{15}.
$$

5. **Round and Add Safety Margin**: Round $\text{pred}_{15}$ to the nearest integer and then apply a 5% upward safety margin to ensure sufficient capacity:

$$
\text{final}_{15} = \lceil 1.05 \times \text{round}(\text{pred}_{15}) \rceil.
$$

### Summary

By following these steps **removing the fluctuation**, **fitting the linear model**, **predicting day 15**'s base consumption, **restoring the fluctuation**, and **applying a safety margin** you'll arrive at a robust energy requirement forecast for the colony's future needs.

### Verified Pure NumPy Solution (From Scratch)
```python
import math

PI = 3.14159

def power_grid_forecast(consumption_data):
    # consumption_data: list of 10 daily consumption values
    # days: 1 through 10
    days = list(range(1, 11))
    n = len(days)

    # 1) Remove daily fluctuation f_i = 10 * sin(2π * i / 10)
    detrended = []
    for i, cons in zip(days, consumption_data):
        fluctuation_i = 10 * math.sin((2 * PI * i) / 10)
        detrended_value = cons - fluctuation_i
        detrended.append(detrended_value)

    # 2) Perform linear regression on the detrended data
    sum_x = sum(days)
    sum_y = sum(detrended)
    sum_xy = sum(x * y for x, y in zip(days, detrended))
    sum_x2 = sum(x**2 for x in days)

    # slope (m) and intercept (b) for y = m*x + b
    m = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x**2)
    b = (sum_y - m * sum_x) / n

    # 3) Predict day 15's base usage
    day_15_base = m * 15 + b

    # 4) Add back daily fluctuation for day 15
    day_15_fluctuation = 10 * math.sin((2 * PI * 15) / 10)
    day_15_prediction = day_15_base + day_15_fluctuation

    # 5) Round and add 5% safety margin
    day_15_rounded = round(day_15_prediction)
    final_15 = math.ceil(day_15_rounded * 1.05)

    return final_15
```

### Verified Sample Trace Data
```json
{
  "test": "print(power_grid_forecast([150, 165, 185, 195, 210, 225, 240, 260, 275, 290]))",
  "expected_output": "404"
}
```

---

## Problem 93: Calculate Mean Absolute Error (MAE)
- **Difficulty**: `easy`
- **Category**: Machine Learning

### Description & Spec
Implement a function to calculate the Mean Absolute Error (MAE) between two arrays of actual and predicted values. The MAE is a metric used to measure the average magnitude of errors in a set of predictions without considering their direction.

### Mathematical Derivation & Intuition
## Mean Absolute Error (MAE)

The Mean Absolute Error (MAE) is a measure of the average magnitude of errors between predicted and actual values. Here's how to express it mathematically:

1. **Basic Formula**:
   - The MAE formula can be written as: $MAE = \frac{1}{n}\sum_{i=1}^n |y_i - \hat{y}_i|$

   Where:
   - $n$ is the number of observations
   - $y_i$ is the true value
   - $\hat{y}_i$ is the predicted value
   - $|...|$ represents the absolute value

2. **Example Calculation**:
   For the values:
   ```
   y_true = [3, -0.5, 2, 7]
   y_pred = [2.5, 0.0, 2, 8]
   ```

   The calculation would be:
   $$
   \begin{align*}
   MAE &= \frac{1}{4}(|3-2.5| + |-0.5-0.0| + |2-2| + |7-8|) \\
   &= \frac{1}{4}(0.5 + 0.5 + 0 + 1) \\
   &= \frac{2}{4} \\
   &= 0.5
   \end{align*}
   $$

3. **Properties**:
   - MAE is always non-negative: $MAE \geq 0$
   - Perfect predictions result in $MAE = 0$
   - MAE is measured in the same units as the original data
   - MAE treats all errors with equal weight (unlike Mean Squared Error)

4. **Comparison with Other Metrics**:
   The formula for Mean Squared Error (MSE) is:
   $$MSE = \frac{1}{n}\sum_{i=1}^n (y_i - \hat{y}_i)^2$$

   While MAE uses absolute values, MSE squares the differences, which:
   - Makes MSE more sensitive to outliers
   - Results in MSE values that are not in the original unit of measurement

### Verified Pure NumPy Solution (From Scratch)
```python
import numpy as np

def mae(y_true, y_pred):
    """
    Calculate Mean Absolute Error between two arrays.

    Parameters:
    y_true (numpy.ndarray): Array of true values
    y_pred (numpy.ndarray): Array of predicted values

    Returns:
    float: Mean Absolute Error rounded to 3 decimal places
    """
    if y_true.shape != y_pred.shape:
        raise ValueError("Arrays must have the same shape")
    if y_true.size == 0:
        raise ValueError("Arrays cannot be empty")

    return round(np.mean(np.abs(y_true - y_pred)), 3)
```

### Verified Sample Trace Data
```json
{
  "test": "print(mae(np.array([3, -0.5, 2, 7]), np.array([2.5, 0.0, 2, 8])))",
  "expected_output": "0.500"
}
```

---

## Problem 104: Binary Classification with Logistic Regression
- **Difficulty**: `easy`
- **Category**: Machine Learning

### Description & Spec
Implement the prediction function for binary classification using Logistic Regression. Your task is to compute class probabilities using the sigmoid function and return binary predictions based on a threshold of 0.5.

### Mathematical Derivation & Intuition
## Binary Classification with Logistic Regression

Logistic Regression is a fundamental algorithm for binary classification. Given input features and learned model parameters (weights and bias), your task is to implement the prediction function that computes class probabilities.

### Mathematical Background

The logistic regression model makes predictions using the sigmoid function:

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

where z is the linear combination of features and weights plus bias:

$$z = \mathbf{w}^T\mathbf{x} + b = \sum_{i=1}^{n} w_ix_i + b$$

### Implementation Requirements

Your task is to implement a function that:

- Takes a batch of samples $\mathbf{X}$ (shape: N x D), weights $\mathbf{w}$ (shape: D), and bias b
- Computes $z = \mathbf{X}\mathbf{w} + b$ for all samples
- Applies the sigmoid function to get probabilities
- Returns binary predictions i.e., 0 or 1 using a threshold of 0.5

### Important Considerations

- Handle numerical stability in sigmoid computation
- Ensure efficient vectorized operations using numpy
- Return binary predictions (0 or 1)

### Hint

To prevent overflow in the exponential calculation of the sigmoid function, use `np.clip` to limit z values:

```python
z = np.clip(z, -500, 500)
```
This ensures numerical stability when dealing with large input values.

### Verified Pure NumPy Solution (From Scratch)
```python
import numpy as np

def predict_logistic(X: np.ndarray, weights: np.ndarray, bias: float) -> np.ndarray:
    """
    Implements binary classification prediction using Logistic Regression.

    Args:
        X: Input feature matrix (shape: N × D)
        weights: Model weights (shape: D)
        bias: Model bias

    Returns:
        Binary predictions (0 or 1)
    """
    z = np.dot(X, weights) + bias
    z = np.clip(z, -500, 500)  # Prevent overflow in exp
    probabilities = 1 / (1 + np.exp(-z))
    return (probabilities >= 0.5).astype(int)
```

### Verified Sample Trace Data
```json
{
  "test": "print(predict_logistic(np.array([[1, 1], [2, 2], [-1, -1], [-2, -2]]), np.array([1, 1]), 0))",
  "expected_output": "[1 1 0 0]"
}
```

---

## Problem 105: Train Softmax Regression with Gradient Descent
- **Difficulty**: `hard`
- **Category**: Machine Learning

### Description & Spec
Implement a gradient descent-based training algorithm for Softmax regression. Your task is to compute model parameters using Cross Entropy loss and return the optimized coefficients along with collected loss values over iterations. Make sure to round your solution to 4 decimal places

### Mathematical Derivation & Intuition
## Overview
Softmax regression is a type of logistic regression that extends it to a multiclass problem by outputting a vector $P$ of probabilities for each distinct class and taking $argmax(P)$.

## Connection to a regular logistic regression
Recall that a standard logistic regression is aimed at approximating
$$
p = \frac{1}{e^{-X\beta}+1} = \\
= \frac{e^{X\beta}}{1+e^{X\beta}},
$$

which actually alignes with the definition of the softmax function:
$$
softmax(z_i)=\sigma(z_i)=\frac{e^{z_i}}{\sum_j^Ce^{z_j}},
$$

where $C$ is the number of classes and values of which sum up to $1$. Hence it simply extends the functionality of sigmoid to more than 2 classes and could be used for assigning probability values in a categorical distribution, i.e. softmax regression searches for the following vector-approximation:
$$
p^{(i)}=\frac{e^{x^{(i)}\beta}}{\sum_j^Ce^{x^{(i)}\beta_j}_j}
$$

## Loss in softmax regression
**tl;dr** key differences in the loss from logistic regression include replacing sigmoid with softmax and calculating several gradients for vectors $\beta_j$ corresponding to a particular class $j\in\{1,...,C\}$.

Recall that we use MLE in logistic regression. It is the same case with softmax regression, although instead of Bernoulli-distributed random variable we have categorical distribution, which is an extension of Bernoulli to more than 2 labels. Its PMF is defined as:
$$
f(y|p)=\prod_{i=1}^Kp_i^{[i=y]},
$$

Hence, our log-likelihood looks like:
$$
\sum_X \sum_j^C [y_i=j] \log \left[p\left(x_i\right)\right]
$$

Where we replace our probability function with softmax:
$$
\sum_X \sum_j^C [y_i=j] \log \frac{e^{x_i\beta_j}}{\sum_j^Ce^{x_i\beta_j}}
$$

where $[i=y]$ is a function, that returns $0$, if $i\neq y$, and $1$ otherwise and $C$ - number of distinct classes (labels). You can see that since we are expecting a $1\times C$ output of $y$, just like in the neuron backprop problem, we will be having separate vector $\beta_j$ for every $j$ class out of $C$. 

## Optimization objective
The optimization objective is the same as with logistic regression. The function, which we are optimizing, is also commonly refered as **Cross Entropy** (CE):

$$
argmin_\beta -[\sum_X \sum_j^C [y_i=j] \log \frac{e^{x_i\beta_j}}{\sum_j^Ce^{x_i\beta_j}}] \\
$$

Then we are yet again using a chain rule for calculating partial derivative of $CE$ with respect to $\beta$:


$$
\frac{\partial CE}{\partial\beta^{(j)}_i}=\frac{\partial CE}{\partial\sigma}\frac{\partial\sigma}{\partial[X\beta^{(j)}]}\frac{\partial[X\beta^{(j)}]}{\beta^{(j)}_i}
$$

Which is eventually reduced to a similar to logistic regression gradient matrix form:
$$
X^T(\sigma(X\beta^{(j)})-Y)
$$

Then we can finally use gradient descent in order to iteratively update our parameters with respect to a particular class:
$$
\beta^{(j)}_{t+1}=\beta^{(j)}_t - \eta [X^T(\sigma(X\beta^{(j)}_t)-Y)]
$$

### Verified Pure NumPy Solution (From Scratch)
```python
import numpy as np


def train_softmaxreg(X: np.ndarray, y: np.ndarray, 
                 learning_rate: float, iterations: int) -> tuple[list[float], ...]:
    '''        
    Gradient-descent training algorithm for softmax regression, that collects mean-reduced
    CE losses, accuracies.
    Returns
    -------
    B : list[float]
        CxM updated parameter vector rounded to 4 floating points
    losses : list[float]
        collected values of a Cross Entropy rounded to 4 floating points
    '''

    def softmax(z):
        return np.exp(z) / np.sum(np.exp(z), axis=1, keepdims=True)

    def accuracy(y_pred, y_true):
        return (np.argmax(y_true, axis=1) == np.argmax(y_pred, axis=1)).sum() / len(y_true)

    def ce_loss(y_pred, y_true):
        true_labels_idx = np.argmax(y_true, axis=1)
        return -np.sum(np.log(y_pred)[list(range(len(y_pred))),true_labels_idx])

    y = y.astype(int)
    C = y.max()+1 # we assume that classes start from 0
    y = np.eye(C)[y]
    X = np.hstack((np.ones((X.shape[0], 1)), X))
    B = np.zeros((X.shape[1], C))
    accuracies, losses = [], []

    for epoch in range(iterations):
        y_pred = softmax(X @ B)
        B -= learning_rate * X.T @ (y_pred - y)
        losses.append(round(ce_loss(y_pred, y), 4))
        accuracies.append(round(accuracy(y_pred, y), 4))

    return B.T.round(4).tolist(), losses
```

### Verified Sample Trace Data
```json
{
  "test": "print(train_softmaxreg(np.array([[2.5257, 2.3333, 1.7730, 0.4106, -1.6648], [1.5101, 1.3023, 1.3198, 1.3608, 0.4638], [-2.0969, -1.3596, -1.0403, -2.2548, -0.3235], [-0.9666, -0.6068, -0.7201, -1.7325, -1.1281], [-0.3809, -0.2485, 0.1878, 0.5235, 1.3072], [0.5482, 0.3315, 0.1067, 0.3069, -0.3755], [-3.0339, -2.0196, -0.6546, -0.9033, 2.8918], [0.2860, -0.1265, -0.5220, 0.2830, -0.5865], [-0.2626, 0.7601, 1.8409, -0.2324, 1.8071], [0.3028, -0.4023, -1.2955, -0.1422, -1.7812]]), np.array([2, 3, 0, 0, 1, 3, 0, 1, 2, 1]), 0.03, 10))",
  "expected_output": "([[-0.0841, -0.5693, -0.3651, -0.2423, -0.5344, 0.0339], [0.2566, 0.0535, -0.2103, -0.4004, 0.2709, -0.1461], [-0.1318, 0.211, 0.3998, 0.523, -0.1001, 0.0545], [-0.0407, 0.3049, 0.1757, 0.1197, 0.3637, 0.0576]], [13.8629, 10.7201, 9.3163, 8.4942, 7.9132, 7.4598, 7.0854, 6.7653, 6.4851, 6.2358])"
}
```

---

## Problem 106: Train Logistic Regression with Gradient Descent
- **Difficulty**: `hard`
- **Category**: Machine Learning

### Description & Spec
Implement a gradient descent-based training algorithm for logistic regression. Your task is to compute model parameters using Binary Cross Entropy loss and return the optimized coefficients along with collected loss values over iterations(round to the 4th decimal).

### Mathematical Derivation & Intuition
## Overview
Logistic regression is a model used for a binary classification poblem.

## Prerequisites for a regular logistic regression
Logistic regression is based on the concept of "logits of odds". **Odds** is measure of how frequent we encounter success. It also allows us to shift our probabilities domain of $[0, 1]$ to $[0,\infty]$ Consider a probability of scoring a goal $p=0.8$, then our $odds=\frac{0.8}{0.2}=4$. This means that every $4$ matches we could be expecting a goal followed by a miss. So the higher the odds, the more consistent is our streak of goals. **Logit** is an inverse of the standard logistic function, i.e. sigmoid: $logit(p)=\sigma^{-1}(p)=ln\frac{p}{1-p}$. In our case $p$ is a probability, therefore we call $\frac{p}{1-p}$ the "odds". The logit allows us to further expand our domain from $[0,\infty]$ to $[-\infty,\infty]$.

With this domain expansion we can treat our problem as a linear regression and try to approximate our logit function: $X\beta=logit(p)$. However what we really want for this approximation is to yield predictions for probabilities:
$$
X\beta=ln\frac{p}{1-p} \\
e^{-X\beta}=\frac{1-p}{p} \\ 
e^{-X\beta}+1 = \frac{1}{p} \\
p = \frac{1}{e^{-X\beta}+1}
$$

What we practically just did is taking an inverse of a logit function w.r.t. our approximation and go back to sigmoid. This is also the backbone of the regular logistic regression, which is commonly defined as:
$$
\pi=\frac{e^{\alpha+X\beta}}{1+e^{\alpha+X\beta}}=\frac{1}{1+e^{-(\alpha+X\beta)}}.
$$

## Loss in logistic regression
The loss function used for solving the logistic regression for $\beta$ is derived from MLE (Maximum Likelihood Estimation). This method allows us to search for $\beta$ that maximize our **likelihood function** $L(\beta)$. This function tells us how likely it is that $X$ has come from the distribution generated by $\beta$: $L(\beta)=L(\beta|X)=P(X|\beta)=\prod_{\{x\in X\}}f^{univar}_X(x;\beta)$, where $f$ is a PMF and $univar$ means univariate, i.e. applied to a single variable.

In the case of a regular logistic regression we expect our output to belong to a single Bernoulli-distributed random variable (hence the univariance), since our true label is either $y_i=0$ or $y_i=1$. The Bernoulli's PMF is defined as $P(Y=y)=p^y(1-p)^{(1-y)}$, where $y\in\{0, 1\}$. Also let's denote $\{x\in X\}$ simply as $X$ and refer to a single pair of vectors from the training set as $(x_i, y_i)$. Thus, our likelihood function would look like this:
$$
\prod_X p\left(x_i\right)^{y_i} \times\left[1-p\left(x_i\right)\right]^{1-y_i}
$$

Then we convert our function from likelihood to log-likelihood by taking $ln$ (or $log$) of it:
$$
\sum_X y_i \log \left[p\left(x_i\right)\right]+\left(1-y_i\right) \log \left[1-p\left(x_i\right)\right]
$$

And then we replace $p(x_i)$ with the sigmoid from previously defined equality to get a final version of our **loss function**:
$$
\sum_X y_i \log \left(\frac{1}{1+e^{-x_i\beta}}\right)+\left(1-y_i\right)\log \left(1-\frac{1}{1+e^{-x_i\beta}}\right)
$$

## Optimization objective
Recall that originally we wanted to search for $\beta$ that maximize the likelihood function. Since $log$ is a monotonic transformation, our maximization objective does not change and we can confindently say that now we can equally search for $\beta$ that maximize our log-likelihood. Hence we can finally write our actual objective as:

$$
argmax_\beta [\sum_X y_i \log\sigma(x_i\beta)+\left(1-y_i\right)\log (1-\sigma(x_i\beta))] = \\
= argmin_\beta -[\sum_X y_i \log\sigma(x_i\beta)+\left(1-y_i\right)\log (1-\sigma(x_i\beta))]
$$

where $\sigma$ is the sigmoid. This function we're trying to minimize is also called **Binary Cross Entropy** loss function (BCE). To find the minimum we would need to take the gradient of this LLF (Log-Likelihood Function), or find a vector of derivatives with respect to every individual $\beta_j$.

### Step 1
To do that we're going to use a chain rule, that describes relational change in variables that our original function is made of. In our case the log-likeligood function depends on sigmoid $\sigma$, $\sigma$ depends on $X\beta$ and $X\beta$ finally depends on $\beta_j$, hence:

$$
\frac{\partial LLF}{\partial\beta_j}=\frac{\partial LLF}{\partial\sigma}\frac{\partial\sigma}{\partial[X\beta]}\frac{\partial[X\beta]}{\beta_j}\\
=-\sum_{i=1}^n\left(y^{(i)} \frac{1}{\sigma\left(x^{(i)}\beta\right)}-(1-y^{(i)} ) \frac{1}{1-\sigma\left(x^{(i)}\beta\right)}\right) \frac{\partial\sigma}{\partial[x^{(i)}\beta]}
$$

### Step 2
Then we use a derivative of the sigmoid function, that is $\frac{\partial\sigma(x)}{\partial x}=\sigma(x)(1-\sigma(x))$: 
$$
-\sum_{i=1}^n\left(y^{(i)} \frac{1}{\sigma\left(x^{(i)}\beta\right)}-(1-y^{(i)} ) \frac{1}{1-\sigma\left(x^{(i)}\beta\right)}\right)\\
     \sigma\left(x^{(i)}\beta\right)\left(1-\sigma\left(x^{(i)}\beta\right)\right)^{(*)} \frac{\partial[x^{(i)}\beta]}{\partial\beta_j} \\
=-\sum_{i=1}^n\left(y^{(i)}\left(1-\sigma\left(x^{(i)}\beta\right)\right)-(1-y^{(i)} ) \sigma\left(x^{(i)}\beta\right)\right) x_j^{(i)} \\
=-\sum_{i=1}^n\left(y^{(i)}-\sigma\left(x^{(i)}\beta\right)\right) x_j^{(i)} \\
=\sum_{i=1}^n\left(\sigma\left(x^{(i)}\beta\right)-y^{(i)}\right) x_j^{(i)}.
$$

The result sum can be then rewritten in a more convenient gradient matrix form as:
$$
X^T(\sigma(X\beta)-Y)
$$

Then we can finally use gradient descent in order to iteratively update our parameters:
$$
\beta_{t+1}=\beta_t - \eta [X^T(\sigma(X\beta_t)-Y)]
$$

### Verified Pure NumPy Solution (From Scratch)
```python
import numpy as np

def train_logreg(X: np.ndarray, y: np.ndarray, learning_rate: float, iterations: int) -> tuple[list[float], ...]:
    """
    Gradient-descent training algorithm for logistic regression, optimizing parameters with Binary Cross Entropy loss.
    """
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    y = y.reshape(-1, 1)
    X = np.hstack((np.ones((X.shape[0], 1)), X))
    B = np.zeros((X.shape[1], 1))
    losses = []

    for _ in range(iterations):
        y_pred = sigmoid(X @ B)
        B -= learning_rate * X.T @ (y_pred - y)
        loss = -np.sum(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
        losses.append(round(loss, 4))

    return B.flatten().round(4).tolist(), losses
```

### Verified Sample Trace Data
```json
{
  "test": "print(train_logreg(np.array([[0.7674, -0.2341, -0.2341, 1.5792], [-1.4123, 0.3142, -1.0128, -0.9080], [-0.4657, 0.5425, -0.4694, -0.4634], [-0.5622, -1.9132, 0.2419, -1.7249], [-1.4247, -0.2257, 1.4656, 0.0675], [1.8522, -0.2916, -0.6006, -0.6017], [0.3756, 0.1109, -0.5443, -1.1509], [0.1968, -1.9596, 0.2088, -1.3281], [1.5230, -0.1382, 0.4967, 0.6476], [-1.2208, -1.0577, -0.0134, 0.8225]]), np.array([1, 0, 0, 0, 1, 1, 0, 0, 1, 0]), 0.001, 10))",
  "expected_output": "([-0.0097, 0.0286, 0.015, 0.0135, 0.0316], [6.9315, 6.9075, 6.8837, 6.8601, 6.8367, 6.8134, 6.7904, 6.7675, 6.7448, 6.7223])"
}
```

---

## Problem 108: Measure Disorder in Apple Colors
- **Difficulty**: `easy`
- **Category**: Machine Learning

### Description & Spec
Implement a function that calculates the disorder in a basket of apples based on their colors, where each apple color is represented by an integer. The disorder must be 0 if all apples are the same color and must increase as the variety of colors increases. In particular:
- [0,0,0,0] should yield 0.
- [1,1,0,0] should have a higher disorder than [0,0,0,0].
- [0,1,2,3] should have a higher disorder than [1,1,0,0].
- [0,0,1,1,2,2,3,3] should have a higher disorder than [0,0,0,0,0,1,2,3].

You may use any method to measure disorder as long as these properties are satisfied.

### Mathematical Derivation & Intuition
## Using Gini Impurity to Measure Disorder

One valid approach to measure disorder in a basket of apples is to use the **Gini impurity** metric. The Gini impurity is defined as:

$$
G = 1 - \sum_{i=1}^{k} p_i^2
$$

where:
- $p_i$ is the proportion of apples of the $i$-th color.
- $k$ is the total number of distinct colors.

### Key Properties

- **Single Color Case:** If all apples in the basket have the same color, then $p = 1$ and the Gini impurity is:
  $$
  G = 1 - 1^2 = 0
  $$
- **Increasing Disorder:** As the variety of colors increases, the impurity increases. For example:
  - Two equally frequent colors:  
    $$
    G = 1 - \left(0.5^2 + 0.5^2\right) = 0.5
    $$
  - Four equally frequent colors:  
    $$
    G = 1 - \left(4 \times 0.25^2\right) = 0.75
    $$

### Comparing Different Baskets

1. **Basket:** `[0,0,0,0]`  
   - Only one color -> $G = 0$
2. **Basket:** `[1,1,0,0]`  
   - Two colors, equal frequency -> $G = 0.5$
3. **Basket:** `[0,1,2,3]`  
   - Four equally frequent colors -> $G = 0.75$
4. **Basket:** `[0,0,1,1,2,2,3,3]`  
   - Equal distribution among four colors -> $G = 0.75$
5. **Basket:** `[0,0,0,0,0,1,2,3]`  
   - One dominant color, three others -> $G = 0.5625$

### Flexibility

While the Gini impurity is a suitable measure of disorder, any method that satisfies the following constraints is valid:
1. A basket with a single color must return a disorder score of **0**.
2. Baskets with more distinct colors must yield **higher disorder** scores.
3. The specific ordering constraints provided in the problem must be **maintained**.

By using this impurity measure, we can quantify how diverse a basket of apples is based on color distribution.

### Verified Pure NumPy Solution (From Scratch)
```python
def disorder(apples: list) -> float:
    """
    Calculates a measure of disorder in a basket of apples based on their colors.
    One valid approach is to use the Gini impurity, defined as:
      G = 1 - sum((count/total)^2 for each color)
    This method returns 0 for a basket with all apples of the same color and increases as the variety of colors increases.
    While this implementation uses the Gini impurity, any method that satisfies the following properties is acceptable:
      1. A single color results in a disorder of 0.
      2. Baskets with more distinct colors yield a higher disorder score.
      3. The ordering constraints are maintained.
    """
    if not apples:
        return 0.0
    total = len(apples)
    counts = {}
    for color in apples:
        counts[color] = counts.get(color, 0) + 1
    impurity = 1.0
    for count in counts.values():
        p = count / total
        impurity -= p * p
    return round(impurity, 4)
```

### Verified Sample Trace Data
```json
{
  "test": "print(disorder([0,0,0,0])<disorder([1,0,0,0]))",
  "expected_output": "True"
}
```

---

## Problem 109: Implement Layer Normalization for Sequence Data
- **Difficulty**: `medium`
- **Category**: Machine Learning

### Description & Spec
Implement a function to perform Layer Normalization on an input tensor. Given a 3D array representing batch_size, sequence length, and feature dimensions, normalize the data across the feature dimension for each sequence, then apply scaling and shifting parameters.

### Mathematical Derivation & Intuition
## Understanding Layer Normalization

Layer Normalization (LN) is a technique commonly used in natural language processing (NLP) tasks to normalize the activations within each individual sample (or instance) across all the features (or dimensions). Unlike **Batch Normalization**, which normalizes across the batch dimension, Layer Normalization normalizes across the feature (or channel) dimension. This makes it particularly useful for sequence-based tasks, where the sequence length varies or where the batch size is small, as is often the case in NLP applications.

### Concepts

Layer Normalization operates on the principle of normalizing each input sample independently, across its feature dimensions. This ensures that all the features in a single instance (or sequence) are scaled and shifted to have similar statistics, which helps stabilize the training process and improve the model's ability to learn.

The process of Layer Normalization consists of the following steps:

1. **Compute the Mean and Variance for Each Sample:** For each input sequence, compute the mean and variance across the feature dimensions.
2. **Normalize the Inputs:** Normalize the activations by subtracting the mean and dividing by the standard deviation.
3. **Apply Scale and Shift:** After normalization, apply a learned scale (gamma) and shift (beta) to allow the model to restore the original distribution of the data, if necessary.

### Structure of Layer Normalization for (Batch Size, Sequence Length, Feature Dimension) Input

For an input tensor with the shape **(batch_size, seq_len, d_model)** (where:
- **batch_size**: number of sequences in a batch,
- **seq_len**: sequence length (number of tokens or time steps),
- **d_model**: number of features (model dimension)),
Layer Normalization operates over the **d_model** (features) dimension, normalizing each sequence independently.

#### 1. Mean and Variance Calculation for Each Sample

- For each individual sequence in the batch (for each **b** in **batch_size**), the **mean** $\mu_b$ and **variance** $\sigma_b^2$ are computed over the features (or channels) for that particular sequence. Importantly, this computation does not involve the batch dimension, meaning each sequence is normalized independently.

  $$ 
  \mu_b = \frac{1}{d_{\text{model}}} \sum_{i=1}^{d_{\text{model}}} x_{b,i}
  $$

  $$
  \sigma_b^2 = \frac{1}{d_{\text{model}}} \sum_{i=1}^{d_{\text{model}}} (x_{b,i} - \mu_b)^2
  $$

  Where:
  - $x_{b,i}$ is the activation at batch index $b$ and feature index $i$ (across the sequence length).
  - $d_{\text{model}}$ is the model dimension (the number of features).

#### 2. Normalization

Once the mean $\mu_b$ and variance $\sigma_b^2$ have been computed for each sequence, the next step is to **normalize** the input by subtracting the mean and dividing by the standard deviation (square root of variance plus a small constant $\epsilon$ for numerical stability):

$$
\hat{x}_{b,i} = \frac{x_{b,i} - \mu_b}{\sqrt{\sigma_b^2 + \epsilon}}
$$

Where:
- $\hat{x}_{b,i}$ is the normalized value of the $i$-th feature for the $b$-th sequence.
- $\epsilon$ is a small constant added to the variance for numerical stability.

#### 3. Scale and Shift

After normalization, the next step is to apply a **scale** ($\gamma$) and **shift** ($\beta$) to the normalized activations for each sequence. These are learned parameters that allow the model to adjust the output distribution for each sequence:

$$
y_{b,i} = \gamma \hat{x}_{b,i} + \beta
$$

Where:
- $\gamma$ is the scaling factor for the feature $i$.
- $\beta$ is the shifting factor for the feature $i$.

#### 4. Training and Inference

- **During Training**: For each sequence, the mean and variance are computed over the feature dimensions and used for normalization.
- **During Inference**: The model uses the running averages of the statistics (mean and variance) computed during training to ensure consistent behavior during inference.

### Why Use Layer Normalization in NLP?

Layer Normalization is especially useful in NLP and sequence-based tasks because of the following reasons:

- **Independence from Batch Size**: Layer Normalization operates independently for each sample (sequence), which means it does not depend on the batch size. This is important in NLP tasks where the batch size can vary, or be small, which would make Batch Normalization less effective.

- **Variable Sequence Lengths**: NLP models often work with sequences of varying lengths. Layer Normalization normalizes over the feature dimension, making it easier to handle sequences of different lengths without the need for special adjustments.

- **Training Stability**: Layer Normalization helps stabilize the training process by ensuring that the activations within each sequence are normalized, preventing the network from becoming sensitive to the scale of the inputs and improving gradient flow.

### Why Not Use Batch Normalization in NLP?

Batch Normalization (BN) normalizes over the batch dimension, which works well when the batch size is large and fixed. However, in NLP tasks, there are a few reasons why **Batch Normalization** is less commonly used:

- **Batch Size Variability**: In NLP, the batch size can vary across training and inference steps. A smaller or variable batch size can lead to poor estimates of the mean and variance during Batch Normalization, which can degrade performance.

- **Sequence Length Variability**: In NLP, the length of input sequences can vary greatly (e.g., sentences of different lengths). Batch Normalization requires that the statistics be computed over the batch, which makes it difficult to apply across sequences of varying lengths without padding or truncation.

- **Dependence on Batch Statistics**: Since Batch Normalization relies on batch statistics, it can cause issues when used in tasks with smaller or highly variable batch sizes, such as in NLP, where each sequence may not represent a meaningful distribution of activations across the batch.

### Key Points

- **Normalization Over Features**: Layer Normalization normalizes across the feature dimensions (model dimension), rather than across the batch dimension, making it ideal for NLP tasks where the batch size may vary or be small.

- **Sequence-Based Normalization**: By normalizing each sequence independently, Layer Normalization ensures that the activations within a single sequence are normalized, without needing information from other sequences in the batch.

- **Stabilizing Training**: Layer Normalization improves the gradient flow and ensures that activations within each sequence are consistent, which stabilizes training and helps prevent vanishing or exploding gradients.

- **Better for Small or Variable Batch Sizes**: Layer Normalization works well with smaller batch sizes, which are often used in NLP tasks like language modeling, machine translation, and text classification.

### Summary

Layer Normalization is particularly effective in NLP tasks because it normalizes each sequence independently, ensuring that each sample has a consistent activation distribution. It is preferable over Batch Normalization in cases where the batch size is small or variable, and when sequences have different lengths, making it a popular choice for sequence-based models like transformers, BERT, and GPT.

### Verified Pure NumPy Solution (From Scratch)
```python
import numpy as np

def layer_normalization(X: np.ndarray, gamma: np.ndarray, beta: np.ndarray, epsilon: float = 1e-5) -> np.ndarray:
    """
    Perform Layer Normalization.
    """
    mean = np.mean(X, axis=-1, keepdims=True)
    variance = np.var(X, axis=-1, keepdims=True)
    X_norm = (X - mean) / np.sqrt(variance + epsilon)
    norm_X = gamma * X_norm + beta
    return norm_X
```

### Verified Sample Trace Data
```json
{
  "test": "np.random.seed(42); X = np.random.randn(2, 2, 3); gamma = np.ones(3).reshape(1, 1, -1); beta = np.zeros(3).reshape(1, 1, -1); print(np.round(layer_normalization(X, gamma, beta), 3))",
  "expected_output": "[[[ 0.474, -1.391,  0.917], [ 1.414, -0.707, -0.707]], [[ 1.132,  0.168, -1.3  ], [ 1.414, -0.705, -0.71 ]]]"
}
```

---

## Problem 131: Implement Efficient Sparse Window Attention
- **Difficulty**: `medium`
- **Category**: Machine Learning

### Description & Spec
Create a function named sparse_window_attention that computes sparse attention over long sequences by sliding a fixed-radius window across the sequence.

• The parameter window_size represents the radius w of the window.
- For a token at index i, attend only to tokens whose indices are within max(0, i - w) through min(seq_len - 1, i + w), inclusive.
- Tokens near the beginning or end of the sequence simply have smaller windows; no padding is added.

• Inputs
- Q, K, V: NumPy arrays with shapes (seq_len, d_k) for Q and K, and (seq_len, d_v) for V.
- window_size: integer window radius.
- scale_factor (optional): value used to scale dot-product scores; if None, default to sqrt(d_k).

• Output
- A NumPy array of shape (seq_len, d_v) containing the attention results.

### Mathematical Derivation & Intuition
## Sparse Window Attention

Sparse window attention is a technique used in sequence processing models to efficiently focus on relevant parts of the data. It limits the model's attention to a local neighborhood around each position, reducing computational demands while maintaining effectiveness for tasks involving long sequences.

### 1. Understanding Attention Mechanisms

Attention mechanisms enable a model to weigh the importance of different elements in a sequence when generating an output. At its core, attention computes a set of weights that indicate how much each element should contribute to the result for a given position. These weights are derived from the similarity between a query representing the current position and keys, which represent other positions. The final output is a combination of values associated with those positions, scaled by the weights.

For instance, imagine reading a sentence: your brain focuses more on nearby words to understand the current word, rather than scanning the entire sentence. Mathematically, this process involves calculating similarities and producing a weighted average of the values.

### 2. The Challenge with Full Attention

In traditional attention, every position in a sequence interacts with every other position, leading to high computational costs. This approach scales poorly for long sequences, as the number of interactions grows quadratically with the sequence length. To address this, sparse attention introduces restrictions, allowing the model to ignore distant or irrelevant positions.

By focusing only on a subset of the sequence, sparse attention maintains accuracy for local dependencies 2014such as in language where words often relate to their immediate neighbors while drastically reducing the resources needed.

### 3. Defining the Window in Sparse Attention

Sparse window attention defines a fixed neighborhood, or "window," around each position. For a given position, the model considers only the elements within a specified radius on either side. This radius, often called the window size, determines how far the attention extends.

For example, if the window size is 2, a position at index 5 would attend to positions 3, 4, 5, 6, and 7 (assuming those exist in the sequence). This sliding window approach ensures that attention is local and efficient, capturing patterns that are typically short-range while discarding long-range interactions that may not be necessary.

The key benefit here is efficiency: by limiting the scope, the overall process avoids examining the entire sequence, much like how a person might skim a text by focusing on paragraphs rather than every line.

### 4. Computing the Attention Scores

Once the window is defined, attention scores are calculated to measure the relevance of each element within that window. These scores are based on the dot product between the query and the keys in the window, which quantifies their similarity.

The formula for the scores is given by:

$$
\text{scores} = \frac{Q K^T}{\sqrt{d_k}}
$$

Here, $Q$ represents the query vector for the current position, $K$ is the matrix of key vectors within the window, and $K^T$ is its transpose. The term $d_k$ denotes the dimensionality of the keys, and dividing by $\sqrt{d_k}$ scales the scores to prevent them from becoming too large, which could destabilize the process.

This equation produces a set of numbers indicating how aligned the query is with each key. A higher score means greater similarity, reflecting a stronger influence on the output.

### 5. Applying the Softmax and Weighted Sum

After obtaining the scores, they are normalized to create probabilities using the softmax function. This step ensures that the weights sum to 1, turning the raw scores into a distribution.

The softmax operation is defined as:

$$
\text{attention weights} = \frac{\exp(\text{scores})}{\sum \exp(\text{scores})}
$$

Each element in the attention weights represents the relative importance of the corresponding key in the window. Finally, the output for the current position is computed as a weighted sum of the values in the window:

$$
\text{output} = \text{attention weights} \cdot V
$$

In this expression, $V$ is the matrix of value vectors within the window. The result is a single vector that combines the values based on their computed importance, effectively summarizing the relevant information from the local context.

### 6. Example Walkthrough

---

Consider a simple sequence of five numbers: [1, 2, 3, 4, 5]. Suppose the window size is 1, meaning each position attends to itself and its immediate neighbors.

For the position of the number 3 (at index 2), the window includes indices 1, 2, and 3 corresponding to the numbers 2, 3, and 4. The model would compute similarities between the query for index 2 and the keys for indices 1, 2, and 3. It then assigns weights to 2, 3, and 4 based on these similarities and produces an output as a weighted combination of these numbers.

This illustrates how sparse window attention efficiently captures local relationships, such as how 3 might relate more to 2 and 4 than to distant numbers like 1 or 5.

### Verified Pure NumPy Solution (From Scratch)
```python
import numpy as np

def sparse_window_attention(Q, K, V, window_size, scale_factor=None):
    """
    Computes sparse attention with a sliding window mask to efficiently handle longer context lengths.
    This implementation uses a loop over the sequence to compute attention only within the specified window,
    reducing memory usage compared to dense attention.

    Args:
        Q (np.ndarray): Query matrix of shape (seq_len, d_k)
        K (np.ndarray): Key matrix of shape (seq_len, d_k)
        V (np.ndarray): Value matrix of shape (seq_len, d_v)
        window_size (int): The radius of the attention window (attends to window_size positions on each side).
        scale_factor (float, optional): Scaling factor for the dot product. If None, uses sqrt(d_k).

    Returns:
        np.ndarray: Attention output of shape (seq_len, d_v)
    """
    seq_len = Q.shape[0]
    d_k = Q.shape[1]
    if scale_factor is None:
        scale_factor = np.sqrt(d_k).astype(float)
    output = np.zeros((seq_len, V.shape[1]), dtype=V.dtype)
    for i in range(seq_len):
        start = max(0, i - window_size)
        end = min(seq_len, i + window_size + 1)
        local_Q = Q[i:i+1]
        local_K = K[start:end]
        local_V = V[start:end]
        scores = np.dot(local_Q, local_K.T) / scale_factor
        max_score = np.max(scores, axis=1, keepdims=True)
        exp_scores = np.exp(scores - max_score)
        sum_exp = np.sum(exp_scores, axis=1, keepdims=True)
        attention_weights = exp_scores / sum_exp
        output[i] = np.dot(attention_weights, local_V)
    return output
```

### Verified Sample Trace Data
```json
{
  "test": "import numpy as np\nQ = np.array([[1.0], [1.0], [1.0]])\nK = np.array([[1.0], [1.0], [1.0]])\nV = np.array([[1.0], [2.0], [3.0]])\nprint(sparse_window_attention(Q, K, V, 1))",
  "expected_output": "[[1.5], [2.], [2.5]]"
}
```

---

## Problem 135: Implement Early Stopping Based on Validation Loss
- **Difficulty**: `easy`
- **Category**: Machine Learning

### Description & Spec
Create a function to decide when to stop training a model early based on a list of validation losses. The early stopping criterion should stop training if the validation loss hasn't improved for a specified number of epochs (patience), and only count as improvement if the loss decreases by more than a certain threshold (min_delta). Your function should return the epoch to stop at and the best epoch that achieved the lowest validation loss.

### Mathematical Derivation & Intuition
## Implementing Early Stopping Criterion

Early stopping is a regularization technique that helps prevent overfitting in machine learning models. Your task is to implement the early stopping decision logic based on the validation loss history.

### Problem Description

Given a sequence of validation losses from model training, determine if training should be stopped based on the following criteria:

- Training should stop if the validation loss hasn't improved (decreased) for a specified number of epochs (patience)
- An improvement is only counted if the loss decreases by more than a minimum threshold (min_delta)
- The best model is the one with the lowest validation loss

### Example

Consider the following validation losses: [0.9, 0.8, 0.75, 0.77, 0.76, 0.77, 0.78]

- With patience=2 and min_delta=0.01:
  - Best loss is 0.75 at epoch 2
  - No improvement > 0.01 for next 2 epochs
  - Should stop at epoch 4

### Function Requirements

- Return both the epoch to stop at and the best epoch
- If no stopping is needed, return the last epoch
- Epochs are 0-indexed

### Verified Pure NumPy Solution (From Scratch)
```python
from typing import Tuple

def early_stopping(val_losses: list[float], patience: int, min_delta: float) -> Tuple[int, int]:
    best_loss = float('inf')
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch, loss in enumerate(val_losses):
        if loss < best_loss - min_delta:
            best_loss = loss
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            return epoch, best_epoch

    return len(val_losses) - 1, best_epoch
```

### Verified Sample Trace Data
```json
{
  "test": "print(early_stopping([0.9, 0.8, 0.75, 0.77, 0.76, 0.77, 0.78], 2, 0.01))",
  "expected_output": "(4, 2)"
}
```

---

## Problem 138: Find the Best Gini-Based Split for a Binary Decision Tree
- **Difficulty**: `medium`
- **Category**: Machine Learning

### Description & Spec
Implement a function that scans every feature and threshold in a small data set, then returns the split that minimises the weighted Gini impurity. Your implementation should support binary class labels (0 or 1) and handle ties gracefully.  

You will write **one** function:

```python
find_best_split(X: np.ndarray, y: np.ndarray) -> tuple[int, float]
```

* **`X`** is an $n\times d$ NumPy array of numeric features.
* **`y`** is a length-$n$ NumPy array of 0/1 labels.
* The function returns `(best_feature_index, best_threshold)` for the split with the **lowest** weighted Gini impurity.
* If several splits share the same impurity, return the first that you encounter while scanning features and thresholds.

### Mathematical Derivation & Intuition
# Learn: Gini Impurity and Best Split in Decision Trees

## Overview

A core concept in Decision Trees (and by extension, Random Forests) is how the model chooses where to split the data at each node. One popular criterion used for splitting is **Gini Impurity**.

In this task, you will implement:
- Gini impurity computation
- Finding the best feature and threshold to split on based on impurity reduction

This helps build the foundation for how trees grow in a Random Forest.

---

## Gini Impurity

For a set of samples with class labels \( y \), the Gini Impurity is defined as:

$$
G(y) = 1 - \sum_{i=1}^{k} p_i^2
$$

Where \( p_i \) is the proportion of samples belonging to class \( i \).

A pure node (all one class) has \( G = 0 \), and higher values indicate more class diversity.

---

## Weighted Gini Impurity

Given a feature and a threshold to split the dataset into left and right subsets:

$$
G_{\text{split}} = \frac{n_{\text{left}}}{n} G(y_{\text{left}}) + \frac{n_{\text{right}}}{n} G(y_{\text{right}})
$$

We choose the split that **minimizes** $( G_{\text{split}} )$.

---

## Problem Statement

You are given a dataset $( X \in \mathbb{R}^{n \times d} )$ and labels $( y \in \{0, 1\}^n $). Implement the following functions:

### Functions to Implement

```python
def find_best_split(X: np.ndarray, y: np.ndarray) -> Tuple[int, float]:
    ...
```

### Verified Pure NumPy Solution (From Scratch)
```python
import numpy as np
from typing import Tuple

def find_best_split(X: np.ndarray, y: np.ndarray) -> Tuple[int, float]:
    def gini(y_subset: np.ndarray) -> float:
        if y_subset.size == 0:
            return 0.0
        p = y_subset.mean()
        return 1.0 - (p**2 + (1 - p)**2)

    n_samples, n_features = X.shape
    best_feature, best_threshold = -1, float('inf')
    best_gini = float('inf')

    for f in range(n_features):
        for threshold in np.unique(X[:, f]):
            left = y[X[:, f] <= threshold]
            right = y[X[:, f] > threshold]
            g_left, g_right = gini(left), gini(right)
            weighted = (len(left) * g_left + len(right) * g_right) / n_samples
            if weighted < best_gini:
                best_gini, best_feature, best_threshold = weighted, f, threshold

    return best_feature, best_threshold
```

### Verified Sample Trace Data
```json
{
  "test": "import numpy as np\nX1 = np.array([[2.5], [3.5], [1.0], [4.0]])\ny1 = np.array([0, 1, 0, 1])\nf1, t1 = find_best_split(X1, y1)\nprint(f1, round(t1, 4))",
  "expected_output": "0, 2.5"
}
```

---

## Problem 139: Elastic Net Regression via Gradient Descent
- **Difficulty**: `medium`
- **Category**: Machine Learning

### Description & Spec
Implement Elastic Net Regression using gradient descent, combining L1 and L2 penalties to handle multicollinearity and encourage sparsity in the feature weights.

### Mathematical Derivation & Intuition
# Elastic Net Regression Using Gradient Descent

Elastic Net Regression combines both L1 (Lasso) and L2 (Ridge) regularization techniques to overcome the limitations of using either regularization method alone. It's particularly useful when dealing with datasets that have many correlated features.

## What is Elastic Net?

Elastic Net addresses two main issues:
- **Lasso's limitation**: When features are highly correlated, Lasso tends to select only one feature from a group of correlated features arbitrarily
- **Ridge's limitation**: Ridge regression doesn't perform feature selection (coefficients approach zero but never become exactly zero)

The goal of Elastic Net is to minimize the objective function:

$$J(w, b) = \underbrace{\frac{1}{2n} \sum_{i=1}^n\left( y_i - \left(\sum_{j=1}^pX_{ij}w_j+b\right)\right)^2}_{\text{MSE Loss}} + \underbrace{\alpha_1 \sum_{j=1}^p |w_j|}_{\text{L1 Regularization}} + \underbrace{\alpha_2 \sum_{j=1}^p w_j^2}_{\text{L2 Regularization}}$$

Where:
* The first term is the **Mean Squared Error (MSE) Loss**: $\frac{1}{2n} \sum_{i=1}^n\left( y_i - \left(\sum_{j=1}^pX_{ij}w_j+b\right)\right)^2$
* The second term is the **L1 Regularization** (Lasso penalty): $\alpha_1 \sum_{j=1}^p |w_j|$
* The third term is the **L2 Regularization** (Ridge penalty): $\alpha_2 \sum_{j=1}^p w_j^2$
* $\alpha_1$ controls the strength of L1 regularization
* $\alpha_2$ controls the strength of L2 regularization

## Step-by-Step Implementation Guide

### 1. Initialize weights $w_j$ and bias $b$ to 0

### 2. Make Predictions
At each iteration, calculate predictions using:
$$\hat{y}_i = \sum_{j=1}^pX_{ij}w_j + b$$

Where:
- $\hat{y}_i$ is the predicted value for the $i$-th sample
- $X_{ij}$ is the value of the $i$-th sample's $j$-th feature
- $w_j$ is the weight associated with the $j$-th feature

### 3. Calculate Residuals
Find the difference between actual and predicted values: $error_i = \hat{y}_i - y_i$

### 4. Update Weights and Bias Using Gradients

**Gradient with respect to weights:**
$$\frac{\partial J}{\partial w_j} = \frac{1}{n} \sum_{i=1}^nX_{ij}(\hat{y}_i - y_i) + \alpha_1 \cdot \text{sign}(w_j) + 2\alpha_2 \cdot w_j$$

**Gradient with respect to bias:**
$$\frac{\partial J}{\partial b} = \frac{1}{n} \sum_{i=1}^n(\hat{y}_i - y_i)$$

**Update rules:**
$$w_j = w_j - \eta \cdot \frac{\partial J}{\partial w_j}$$
$$b = b - \eta \cdot \frac{\partial J}{\partial b}$$

Where $\eta$ is the learning rate.

### 5. Check for Convergence
Repeat steps 2-4 until convergence. Convergence is determined by evaluating the L1 norm of the weight gradients:

$$||\nabla w||_1 = \sum_{j=1}^p \left|\frac{\partial J}{\partial w_j}\right|$$

If $||\nabla w||_1 < \text{tolerance}$, stop the algorithm.

### 6. Return the Final Weights and Bias

## Key Parameters

- **alpha1**: L1 regularization strength (promotes sparsity)
- **alpha2**: L2 regularization strength (handles correlated features)
- **learning_rate**: Step size for gradient descent
- **max_iter**: Maximum number of iterations
- **tol**: Convergence tolerance
Path
## Key Differences from Lasso and Ridge

1. **Lasso (L1 only)**: Tends to select one feature from correlated groups, can be unstable with small sample sizes
2. **Ridge (L2 only)**: Keeps all features but shrinks coefficients, doesn't perform feature selection
3. **Elastic Net (L1 + L2)**: Combines benefits of both - performs feature selection while handling correlated features better than Lasso alone

The balance between L1 and L2 regularization is controlled by the `alpha1` and `alpha2` parameters, allowing you to tune the model for your specific dataset characteristics.

### Verified Pure NumPy Solution (From Scratch)
```python
import numpy as np

def elastic_net_gradient_descent(
    X: np.ndarray,
    y: np.ndarray,
    alpha1: float = 0.1,
    alpha2: float = 0.1,
    learning_rate: float = 0.01,
    max_iter: int = 1000,
    tol: float = 1e-4,
) -> tuple:
    n_samples, n_features = X.shape
    weights = np.zeros(n_features)
    bias = 0

    for _ in range(max_iter):
        y_pred = np.dot(X, weights) + bias
        error = y_pred - y
        grad_w = (1 / n_samples) * np.dot(X.T, error) + alpha1 * np.sign(weights) + 2 * alpha2 * weights
        grad_b = (1 / n_samples) * np.sum(error)
        weights -= learning_rate * grad_w
        bias -= learning_rate * grad_b
        if np.linalg.norm(grad_w, ord=1) < tol:
            break

    return weights, bias
```

### Verified Sample Trace Data
```json
{
  "test": "X = np.array([[0, 0], [1, 1], [2, 2]])\ny = np.array([0, 1, 2])\nw, b = elastic_net_gradient_descent(X, y, alpha1=0.1, alpha2=0.1, learning_rate=0.01, max_iter=1000)\nprint(np.round(w,2), round(b,2))",
  "expected_output": "[0.37, 0.37],0.25"
}
```

---

## Problem 140: Bernoulli Naive Bayes Classifier
- **Difficulty**: `medium`
- **Category**: Machine Learning

### Description & Spec
Write a Python class to implement the Bernoulli Naive Bayes classifier for binary (0/1) feature data. Your class should have two methods: `forward(self, X, y)` to train on the input data (X: 2D NumPy array of binary features, y: 1D NumPy array of class labels) and `predict(self, X)` to output predicted labels for a 2D test matrix X. Use Laplace smoothing (parameter: smoothing=1.0). Return predictions as a NumPy array. Only use NumPy. Predictions must be binary (0 or 1) and you must handle cases where the training data contains only one class. All log/likelihood calculations should use log probabilities for numerical stability.

### Mathematical Derivation & Intuition
# **Naive Bayes Classifier**

## **1. Definition**

Naive Bayes is a **probabilistic machine learning algorithm** used for **classification tasks**. It is based on **Bayes' Theorem**, which describes the probability of an event based on prior knowledge of related events.

The algorithm assumes that:
- **Features are conditionally independent** given the class label (the "naive" assumption).
- It calculates the posterior probability for each class and assigns the class with the **highest posterior** to the sample.

---

## **2. Bayes' Theorem**

Bayes' Theorem is given by:

$$
P(C | X) = \frac{P(X | C) \times P(C)}{P(X)}
$$

Where:
- $P(C | X)$ **Posterior** probability: the probability of class $C $ given the feature vector $X$
- $P(X | C)$ → **Likelihood**: the probability of the data $X$ given the class
- $P(C)$ → **Prior** probability: the initial probability of class $C$ before observing any data
- $ P(X)$ → **Evidence**: the total probability of the data across all classes (acts as a normalizing constant)

Since $P(X)$ is the same for all classes during comparison, it can be ignored, simplifying the formula to:

$$
P(C | X) \propto P(X | C) \times P(C)
$$
---

### 3 **Bernoulli Naive Bayes**
- Used for **binary data** (features take only 0 or 1 values).
- The likelihood is given by:

$$
P(X | C) = \prod_{i=1}^{n} P(x_i | C)^{x_i} \cdot (1 - P(x_i | C))^{1 - x_i}
$$

---

## **4. Applications of Naive Bayes**

- **Text Classification:** Spam detection, sentiment analysis, and news categorization.
- **Document Categorization:** Sorting documents by topic.
- **Fraud Detection:** Identifying fraudulent transactions or behaviors.
- **Recommender Systems:** Classifying users into preference groups.

---

### Verified Pure NumPy Solution (From Scratch)
```python
import numpy as np

class NaiveBayes():
    def __init__(self, smoothing=1.0):
        self.smoothing = smoothing
        self.classes = None
        self.priors = None
        self.likelihoods = None

    def forward(self, X, y):
        self.classes, class_counts = np.unique(y, return_counts=True)
        self.priors = {cls: np.log(class_counts[i] / len(y)) for i, cls in enumerate(self.classes)}
        self.likelihoods = {}
        for cls in self.classes:
            X_cls = X[y == cls]
            prob = (np.sum(X_cls, axis=0) + self.smoothing) / (X_cls.shape[0] + 2 * self.smoothing)
            self.likelihoods[cls] = (np.log(prob), np.log(1 - prob))

    def _compute_posterior(self, sample):
        posteriors = {}
        for cls in self.classes:
            posterior = self.priors[cls]
            prob_1, prob_0 = self.likelihoods[cls]
            likelihood = np.sum(sample * prob_1 + (1 - sample) * prob_0)
            posterior += likelihood
            posteriors[cls] = posterior
        return max(posteriors, key=posteriors.get)

    def predict(self, X):
        return np.array([self._compute_posterior(sample) for sample in X])
```

### Verified Sample Trace Data
```json
{
  "test": "import numpy as np\nmodel = NaiveBayes(smoothing=1.0)\nX = np.array([[1, 0, 1], [1, 1, 0], [0, 0, 1], [0, 1, 0], [1, 1, 1]])\ny = np.array([1, 1, 0, 0, 1])\nmodel.forward(X, y)\nprint(model.predict(np.array([[1, 0, 1]])))",
  "expected_output": "[1]"
}
```

---

## Problem 141: Shift and Scale Array to Target Range
- **Difficulty**: `easy`
- **Category**: Machine Learning

### Description & Spec
Write a Python function `convert_range` that shifts and scales the values of a NumPy array from their original range $[a, b]$ (where $a=\min(x)$ and $b=\max(x)$) to a new target range $[c, d]$. Your function should work for both 1D and 2D arrays, returning an array of the same shape, and only use NumPy. Return floating-point results, and ensure you use the correct formula to map the input interval to the output interval.

### Mathematical Derivation & Intuition
# **Shifting and Scaling a Range (Rescaling Data)**

## **1. Motivation**

Rescaling (or shifting and scaling) is a common preprocessing step in data analysis and machine learning. It's often necessary to map data from an original range (e.g., test scores, pixel values, GPA) to a new range suitable for downstream tasks or compatibility between datasets. For example, you might want to shift a GPA from $[0, 10]$ to $[0, 4]$ for comparison or model input.

---

## **2. The General Mapping Formula**

Suppose you have input values in the range $[a, b]$ and you want to map them to the interval $[c, d]$.

- First, shift the lower bound to $0$ by applying $x \mapsto x - a$, so $[a, b] \rightarrow [0, b-a]$.
- Next, scale to unit interval: $t \mapsto \frac{1}{b-a} \cdot t$, yielding $[0, 1]$.
- Now, scale to $[0, d-c]$ with $t \mapsto (d-c)t$, and shift to $[c, d]$ with $t \mapsto c + t$.
- Combining all steps, the complete formula is:

$$
    f(x) = c + \left(\frac{d-c}{b-a}\right)(x-a)
$$

- $x$ = the input value
- $a = \min(x)$ and $b = \max(x)$
- $c$, $d$ = target interval endpoints

---

## **3. Applications**
- **Image Processing**: Rescale pixel intensities
- **Feature Engineering**: Normalize features to a common range
- **Score Conversion**: Convert test scores or grades between systems

---

## **4. Practical Considerations**
- Be aware of the case when $a = b$ (constant input); this may require special handling (e.g., output all $c$).
- For multidimensional arrays, use NumPy’s `.min()` and `.max()` to determine the full input range.

---

This formula gives a **simple, mathematically justified way to shift and scale data to any target range**—a core tool for robust machine learning pipelines.

### Verified Pure NumPy Solution (From Scratch)
```python
import numpy as np

def convert_range(values: np.ndarray, c: float, d: float) -> np.ndarray:
    """
    Shift and scale values from their original range [min, max] to a target [c, d] range.

    Parameters
    ----------
    values : np.ndarray
        Input array (1D or 2D) to be rescaled.
    c : float
        New range lower bound.
    d : float
        New range upper bound.

    Returns
    -------
    np.ndarray
        Scaled array with the same shape as the input.
    """
    a, b = values.min(), values.max()
    return c + (d - c) / (b - a) * (values - a)
```

### Verified Sample Trace Data
```json
{
  "test": "import numpy as np\nseq = np.array([388, 242, 124, 384, 313, 277, 339, 302, 268, 392])\nc, d = 0, 1\nout = convert_range(seq, c, d)\nprint(np.round(out, 6))",
  "expected_output": "[0.985075, 0.440299, 0.,       0.970149, 0.705224, 0.570896, 0.802239, 0.664179, 0.537313, 1.      ]"
}
```

---

## Problem 152: Implementing ROUGE Score
- **Difficulty**: `medium`
- **Category**: Machine Learning

### Description & Spec
## Problem

Implement the ROUGE-1 (Recall-Oriented Understudy for Gisting Evaluation) score to evaluate the quality of a generated summary by comparing it to a reference summary. ROUGE-1 focuses on unigram (single word) overlaps between the candidate and reference texts.

Your task is to write a function that computes the ROUGE-1 recall, precision, and F1 score based on the number of overlapping unigrams.

### Mathematical Derivation & Intuition
# ROUGE-1 Score Learning Guide

## Solution Explanation

ROUGE-1 (Recall-Oriented Understudy for Gisting Evaluation) is a fundamental metric for evaluating the quality of automatically generated summaries by comparing them to reference summaries. The "1" in ROUGE-1 refers to unigrams (single words), making it the most basic but widely used variant of ROUGE metrics.

### Intuition

Imagine you're a teacher grading a student's book summary. You have a reference summary (the "gold standard") and want to measure how well the student's summary captures the key information. ROUGE-1 essentially counts how many important words from the reference summary appear in the student's summary.

The core idea is simple: **if a generated summary contains many of the same words as a high-quality reference summary, it's likely capturing similar content and therefore of good quality.**

### Mathematical Foundation

ROUGE-1 is built on three fundamental components:

**1. Precision (P)**
$$P = \frac{\text{Number of overlapping unigrams}}{\text{Total unigrams in generated summary}}$$

**2. Recall (R)**
$$R = \frac{\text{Number of overlapping unigrams}}{\text{Total unigrams in reference summary}}$$

**3. F1-Score (F)**
$$F = \frac{2 \times P \times R}{P + R}$$

Where an "overlapping unigram" is a word that appears in both the generated summary and the reference summary.

### Step-by-Step Calculation Process

Let's work through a concrete example:

**Reference Summary:** "The quick brown fox jumps over the lazy dog"
**Generated Summary:** "A quick fox jumps over a lazy cat"

**Step 1: Tokenization**
- Reference tokens: ["The", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog"]
- Generated tokens: ["A", "quick", "fox", "jumps", "over", "a", "lazy", "cat"]

**Step 2: Identify Overlapping Unigrams**
Overlapping words (case-insensitive): ["quick", "fox", "jumps", "over", "lazy"]
- Count of overlapping unigrams: 5

**Step 3: Calculate Precision**
$$P = \frac{5}{8} = 0.625$$
*Interpretation: 62.5% of words in the generated summary appear in the reference*

**Step 4: Calculate Recall**
$$R = \frac{5}{9} = 0.556$$
*Interpretation: 55.6% of words in the reference summary are captured in the generated summary*

**Step 5: Calculate F1-Score**
$$F = \frac{2 \times 0.625 \times 0.556}{0.625 + 0.556} = \frac{0.695}{1.181} = 0.588$$

### Understanding the Components

**Precision answers:** "Of all the words in my generated summary, how many are actually relevant (appear in the reference)?"
- High precision means the generated summary doesn't contain many irrelevant words
- Low precision suggests the summary is verbose or off-topic

**Recall answers:** "Of all the important words in the reference, how many did my generated summary capture?"
- High recall means the generated summary covers most key information
- Low recall suggests the summary misses important content

**F1-Score provides:** A balanced measure that penalizes both missing important information (low recall) and including irrelevant information (low precision)

### Advanced Considerations

**Preprocessing Steps:**
1. **Case normalization:** Convert all text to lowercase
2. **Tokenization:** Split text into individual words
3. **Stop word handling:** Optionally remove common words like "the", "and", "is"
4. **Stemming/Lemmatization:** Optionally reduce words to their root forms

**Mathematical Variants:**
- **ROUGE-1 Precision:** $P = \frac{\sum_{i} \text{Count}_{\text{match}}(unigram_i)}{\sum_{i} \text{Count}(unigram_i)}$
- **ROUGE-1 Recall:** $R = \frac{\sum_{i} \text{Count}_{\text{match}}(unigram_i)}{\sum_{i} \text{Count}_{\text{ref}}(unigram_i)}$

Where $\text{Count}_{\text{match}}(unigram_i)$ is the minimum of the counts of $unigram_i$ in the generated and reference summaries.

### Practical Implementation Insights

**Handling Multiple References:**
When multiple reference summaries exist, ROUGE-1 can be calculated against each reference separately, then the maximum score is typically taken:

$$\text{ROUGE-1} = \max_{j} \text{ROUGE-1}(\text{generated}, \text{reference}_j)$$

**Limitations to Consider:**
- **Word order independence:** ROUGE-1 ignores sentence structure and word order
- **Semantic blindness:** Synonyms and paraphrases aren't recognized
- **Length bias:** Longer summaries may achieve higher recall simply by including more words

### Real-World Applications

ROUGE-1 is extensively used in:
- **Automatic summarization evaluation** (news articles, scientific papers)
- **Machine translation quality assessment** (as a secondary metric)
- **Question answering systems** (evaluating answer quality)
- **Chatbot response evaluation** (measuring relevance to expected responses)

### Verified Pure NumPy Solution (From Scratch)
```python
from collections import Counter

def rouge_1_score(reference: str, candidate: str) -> dict:
    """
    Compute ROUGE-1 score between reference and candidate texts.
    
    Returns a dictionary with precision, recall, and f1.
    """
    ref_tokens = reference.lower().split()
    cand_tokens = candidate.lower().split()

    ref_counter = Counter(ref_tokens)
    cand_counter = Counter(cand_tokens)

    # Count overlapping unigrams
    overlap = sum(min(ref_counter[w], cand_counter[w]) for w in cand_counter)

    precision = overlap / len(cand_tokens) if cand_tokens else 0.0
    recall = overlap / len(ref_tokens) if ref_tokens else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0.0

    return {"precision": precision, "recall": recall, "f1": f1}
```

### Verified Sample Trace Data
```json
{
  "test": "print(rouge_1_score('the cat sat on the mat', 'the cat is on the mat'))",
  "expected_output": "{'precision': 0.8333333333333334, 'recall': 0.8333333333333334, 'f1': 0.8333333333333334}"
}
```

---

## Problem 153: StepLR Learning Rate Scheduler
- **Difficulty**: `easy`
- **Category**: Machine Learning

### Description & Spec
## Problem

Write a Python class StepLRScheduler to implement a learning rate scheduler based on the StepLR strategy. Your class should have an __init__ method implemented to initialize with an initial_lr (float), step_size (int), and gamma (float) parameter. It should also have a **get_lr(self, epoch)** method implemented that returns the current learning rate for a given epoch (int). The learning rate should be decreased by gamma every step_size epochs. The answer should be rounded to 4 decimal places. Only use standard Python.

### Mathematical Derivation & Intuition
# **Learning Rate Schedulers: StepLR**

## **1. Definition**
A **learning rate scheduler** is a component used in machine learning, especially in neural network training, to adjust the learning rate during the training process. The **learning rate** is a hyperparameter that determines the step size at each iteration while moving towards a minimum of a loss function.

**StepLR (Step Learning Rate)** is a common type of learning rate scheduler that multiplicatively decays the learning rate by a fixed factor at predefined intervals (epochs). It is simple yet effective in stabilizing training and improving model performance.

## **2. Why Use Learning Rate Schedulers?**
* **Faster Convergence:** A higher initial learning rate can help quickly move through the loss landscape.
* **Improved Performance:** A smaller learning rate towards the end of training allows for finer adjustments and helps in converging to a better local minimum, avoiding oscillations around the minimum.
* **Stability:** Reducing the learning rate prevents large updates that could lead to divergence or instability.

## **3. StepLR Mechanism**
The learning rate is reduced by a factor γ (gamma) every step size epochs.

The formula for the learning rate at a given epoch is:

$$LR_e = LR_{\text{initial}} \times \gamma^{\lfloor e / \text{step size} \rfloor}$$

Where:
* $LR_e$: The learning rate at epoch e.
* $LR_{\text{initial}}$: The initial learning rate.
* γ (gamma): The multiplicative factor by which the learning rate is reduced (usually between 0 and 1, e.g., 0.1, 0.5).
* step size: The interval (in epochs) after which the learning rate is decayed.
* ⌊ · ⌋: The floor function, which rounds down to the nearest integer. This determines how many times the learning rate has been decayed.

**Example:**
If initial learning rate = 0.1, step size = 5, and γ = 0.5:
* Epoch 0-4: $LR_e = 0.1 \times 0.5^{\lfloor 0/5 \rfloor} = 0.1 \times 0.5^0 = 0.1$
* Epoch 5-9: $LR_e = 0.1 \times 0.5^{\lfloor 5/5 \rfloor} = 0.1 \times 0.5^1 = 0.05$
* Epoch 10-14: $LR_e = 0.1 \times 0.5^{\lfloor 10/5 \rfloor} = 0.1 \times 0.5^2 = 0.025$

## **4. Applications of Learning Rate Schedulers**
Learning rate schedulers, including StepLR, are widely used in training various machine learning models, especially deep neural networks, across diverse applications such as:
* **Image Classification:** Training Convolutional Neural Networks (CNNs) for tasks like object recognition.
* **Natural Language Processing (NLP):** Training Recurrent Neural Networks (RNNs) and Transformers for tasks like machine translation, text generation, and sentiment analysis.
* **Speech Recognition:** Training models for converting spoken language to text.
* **Reinforcement Learning:** Optimizing policies in reinforcement learning agents.
* **Any optimization problem:** where gradient descent or its variants are used.

### Verified Pure NumPy Solution (From Scratch)
```python
class StepLRScheduler:
    def __init__(self, initial_lr, step_size, gamma):
        """
        Initializes the StepLR scheduler.

        Args:
            initial_lr (float): The initial learning rate.
            step_size (int): The period of learning rate decay.
                             Learning rate is decayed every `step_size` epochs.
            gamma (float): The multiplicative factor of learning rate decay.
                           (e.g., 0.1 for reducing LR by 10x).
        """
        self.initial_lr = initial_lr
        self.step_size = step_size
        self.gamma = gamma

    def get_lr(self, epoch):
        """
        Calculates and returns the current learning rate for a given epoch.

        Args:
            epoch (int): The current epoch number (0-indexed).

        Returns:
            float: The calculated learning rate for the current epoch.
        """
        # Calculate the number of decays that have occurred.
        # Integer division (//) in Python automatically performs the floor operation.
        num_decays = epoch // self.step_size
        
        # Apply the decay factor 'gamma' raised to the power of 'num_decays'
        # to the initial learning rate.
        current_lr = self.initial_lr * (self.gamma ** num_decays)
        
        return round(current_lr, 4)
```

### Verified Sample Trace Data
```json
{
  "test": "scheduler = StepLRScheduler(initial_lr=0.1, step_size=5, gamma=0.5)\nprint(scheduler.get_lr(epoch=0))",
  "expected_output": "0.1"
}
```

---

## Problem 154: ExponentialLR Learning Rate Scheduler
- **Difficulty**: `easy`
- **Category**: Machine Learning

### Description & Spec
## Problem

Write a Python class ExponentialLRScheduler to implement a learning rate scheduler based on the ExponentialLR strategy. Your class should have an __init__ method to initialize with an initial_lr (float) and gamma (float) parameter. It should also have a get_lr(self, epoch) method that returns the current learning rate for a given epoch (int). The learning rate should be decreased by gamma every epoch. The returned learning rate should be rounded to 4 decimal places. Only use standard Python.

### Mathematical Derivation & Intuition
# **Learning Rate Schedulers: ExponentialLR**

## **1. Definition**
A **learning rate scheduler** is a component used in machine learning, especially in neural network training, to adjust the learning rate during the training process. The **learning rate** is a hyperparameter that determines the step size at each iteration while moving towards a minimum of a loss function.

**ExponentialLR (Exponential Learning Rate)** is a common type of learning rate scheduler that decays the learning rate by a fixed multiplicative factor γ (gamma) at *every* epoch. This results in an exponential decrease of the learning rate over time. It's often used when a rapid and continuous reduction of the learning rate is desired.

## **2. Why Use Learning Rate Schedulers?**
* **Faster Convergence:** A higher initial learning rate can help quickly move through the loss landscape.
* **Improved Performance:** A smaller learning rate towards the end of training allows for finer adjustments and helps in converging to a better local minimum, avoiding oscillations around the minimum.
* **Stability:** Reducing the learning rate prevents large updates that could lead to divergence or instability.

## **3. ExponentialLR Mechanism**
The learning rate is reduced by a factor γ (gamma) every epoch.

The formula for the learning rate at a given epoch e is:

$$LR_e = LR_{\text{initial}} \times \gamma^e$$

Where:
* $LR_e$: The learning rate at epoch e.
* $LR_{\text{initial}}$: The initial learning rate.
* γ (gamma): The multiplicative factor by which the learning rate is reduced per epoch (usually between 0 and 1, e.g., 0.9, 0.99).
* e: The current epoch number (0-indexed).

**Example:**
If initial learning rate = 0.1, and γ = 0.9:
* Epoch 0: $LR_0 = 0.1 \times 0.9^0 = 0.1 \times 1 = 0.1$
* Epoch 1: $LR_1 = 0.1 \times 0.9^1 = 0.1 \times 0.9 = 0.09$
* Epoch 2: $LR_2 = 0.1 \times 0.9^2 = 0.1 \times 0.81 = 0.081$
* Epoch 3: $LR_3 = 0.1 \times 0.9^3 = 0.1 \times 0.729 = 0.0729$

## **4. Applications of Learning Rate Schedulers**
Learning rate schedulers, including ExponentialLR, are widely used in training various machine learning models, especially deep neural networks, across diverse applications such as:
* **Image Classification:** Training Convolutional Neural Networks (CNNs) for tasks like object recognition.
* **Natural Language Processing (NLP):** Training Recurrent Neural Networks (RNNs) and Transformers for tasks like machine translation, text generation, and sentiment analysis.
* **Speech Recognition:** Training models for converting spoken language to text.
* **Reinforcement Learning:** Optimizing policies in reinforcement learning agents.
* **Any optimization problem** where gradient descent or its variants are used.

### Verified Pure NumPy Solution (From Scratch)
```python
class ExponentialLRScheduler:
    def __init__(self, initial_lr, gamma):
        """
        Initializes the ExponentialLR scheduler.

        Args:
            initial_lr (float): The initial learning rate.
            gamma (float): The multiplicative factor of learning rate decay per epoch.
                           (e.g., 0.9 for reducing LR by 10% each epoch).
        """
        self.initial_lr = initial_lr
        self.gamma = gamma

    def get_lr(self, epoch):
        """
        Calculates and returns the current learning rate for a given epoch,
        rounded to 4 decimal places.

        Args:
            epoch (int): The current epoch number (0-indexed).

        Returns:
            float: The calculated learning rate for the current epoch, rounded to 4 decimal places.
        """
        # Apply the decay factor 'gamma' raised to the power of the current 'epoch'
        # to the initial learning rate.
        current_lr = self.initial_lr * (self.gamma ** epoch)
        
        # Round the learning rate to 4 decimal places
        return round(current_lr, 4)
```

### Verified Sample Trace Data
```json
{
  "test": "scheduler = ExponentialLRScheduler(initial_lr=0.1, gamma=0.9)\nprint(f\"{scheduler.get_lr(epoch=0):.4f}\")",
  "expected_output": "0.1000"
}
```

---

## Problem 155: CosineAnnealingLR Learning Rate Scheduler
- **Difficulty**: `medium`
- **Category**: Machine Learning

### Description & Spec
## Problem

Write a Python class CosineAnnealingLRScheduler to implement a learning rate scheduler based on the Cosine Annealing LR strategy. Your class should have an __init__ method to initialize with an initial_lr (float), T_max (int, the maximum number of iterations/epochs), and min_lr (float, the minimum learning rate) parameters. It should also have a **get_lr(self, epoch)** method that returns the current learning rate for a given epoch (int). The learning rate should follow a cosine annealing schedule. The returned learning rate should be rounded to 4 decimal places. Only use standard Python and the math module for trigonometric functions.

### Mathematical Derivation & Intuition
# **Learning Rate Schedulers: CosineAnnealingLR**

## **1. Definition**
A **learning rate scheduler** is a technique used in machine learning to adjust the learning rate during the training of a model. The **learning rate** dictates the step size taken in the direction of the negative gradient of the loss function.

**CosineAnnealingLR (Cosine Annealing Learning Rate)** is a scheduler that aims to decrease the learning rate from a maximum value to a minimum value following the shape of a cosine curve. This approach helps in achieving faster convergence while also allowing the model to explore flatter regions of the loss landscape towards the end of training. It is particularly effective for deep neural networks.

## **2. Why Use Learning Rate Schedulers?**
* **Faster Convergence:** A higher initial learning rate allows for quicker movement through the loss landscape.
* **Improved Performance:** A smaller learning rate towards the end of training enables finer adjustments, helping the model converge to a better local minimum and preventing oscillations.
* **Avoiding Local Minima:** The cyclical nature (or a part of it, as often seen in restarts) of cosine annealing can help the optimizer escape shallow local minima.
* **Stability:** Gradual reduction in learning rate promotes training stability.

## **3. CosineAnnealingLR Mechanism**
The learning rate is scheduled according to a cosine function. Over a cycle of $T_{\text{max}}$ epochs, the learning rate decreases from an initial learning rate (often considered the maximum $LR_{\text{max}}$) to a minimum learning rate ($LR_{\text{min}}$).

The formula for the learning rate at a given epoch e is:

$$LR_e = LR_{\text{min}} + 0.5 \times (LR_{\text{initial}} - LR_{\text{min}}) \times \left(1 + \cos\left(\frac{e}{T_{\text{max}}} \times \pi\right)\right)$$

Where:
* $LR_e$: The learning rate at epoch e.
* $LR_{\text{initial}}$: The initial (maximum) learning rate.
* $LR_{\text{min}}$: The minimum learning rate that the schedule will reach.
* $T_{\text{max}}$: The maximum number of epochs in the cosine annealing cycle. The learning rate will reach $LR_{\text{min}}$ at epoch $T_{\text{max}}$.
* e: The current epoch number (0-indexed), clamped between 0 and $T_{\text{max}}$.
* π: The mathematical constant pi (approximately 3.14159).
* $\cos(\cdot)$: The cosine function.

**Example:**
If $LR_{\text{initial}} = 0.1$, $T_{\text{max}} = 10$, and $LR_{\text{min}} = 0.001$:

* **Epoch 0:** 
  $LR_0 = 0.001 + 0.5 \times (0.1 - 0.001) \times (1 + \cos(0)) = 0.001 + 0.0495 \times 2 = 0.1$

* **Epoch 5 (mid-point):** 
  $LR_5 = 0.001 + 0.5 \times (0.1 - 0.001) \times (1 + \cos(\pi/2)) = 0.001 + 0.0495 \times 1 = 0.0505$

* **Epoch 10 (end of cycle):** 
  $LR_{10} = 0.001 + 0.5 \times (0.1 - 0.001) \times (1 + \cos(\pi)) = 0.001 + 0.0495 \times 0 = 0.001$

## **4. Applications of Learning Rate Schedulers**
Learning rate schedulers, including CosineAnnealingLR, are widely used in training various machine learning models, especially deep neural networks, across diverse applications such as:
* **Image Classification:** Training Convolutional Neural Networks (CNNs) for tasks like object recognition.
* **Natural Language Processing (NLP):** Training Recurrent Neural Networks (RNNs) and Transformers for tasks like machine translation, text generation, and sentiment analysis.
* **Speech Recognition:** Training models for converting spoken language to text.
* **Reinforcement Learning:** Optimizing policies in reinforcement learning agents.
* **Any optimization problem** where gradient descent or its variants are used.

### Verified Pure NumPy Solution (From Scratch)
```python
import math

class CosineAnnealingLRScheduler:
    def __init__(self, initial_lr, T_max, min_lr):
        """
        Initializes the CosineAnnealingLR scheduler.

        Args:
            initial_lr (float): The initial (maximum) learning rate.
            T_max (int): The maximum number of epochs in the cosine annealing cycle.
                         The learning rate will reach min_lr at this epoch.
            min_lr (float): The minimum learning rate.
        """
        self.initial_lr = initial_lr
        self.T_max = T_max
        self.min_lr = min_lr

    def get_lr(self, epoch):
        """
        Calculates and returns the current learning rate for a given epoch,
        following a cosine annealing schedule and rounded to 4 decimal places.

        Args:
            epoch (int): The current epoch number (0-indexed).

        Returns:
            float: The calculated learning rate for the current epoch, rounded to 4 decimal places.
        """
        # Ensure epoch does not exceed T_max for the calculation cycle,
        # as the cosine formula is typically defined for e from 0 to T_max.
        # Although in practice, schedulers might restart or hold LR after T_max.
        # For this problem, we'll clamp it to T_max if it goes over.
        current_epoch = min(epoch, self.T_max)

        # Calculate the learning rate using the Cosine Annealing formula
        # LR_e = LR_min + 0.5 * (LR_initial - LR_min) * (1 + cos(e / T_max * pi))
        lr = self.min_lr + 0.5 * (self.initial_lr - self.min_lr) * \
             (1 + math.cos(current_epoch / self.T_max * math.pi))
        
        # Round the learning rate to 4 decimal places
        return round(lr, 4)
```

### Verified Sample Trace Data
```json
{
  "test": "import math\nscheduler = CosineAnnealingLRScheduler(initial_lr=0.1, T_max=10, min_lr=0.001)\nprint(f\"{scheduler.get_lr(epoch=0):.4f}\")",
  "expected_output": "0.1000"
}
```

---

## Problem 160: Mixed Precision Training
- **Difficulty**: `medium`
- **Category**: Machine Learning

### Description & Spec
Write a Python class to implement Mixed Precision Training that uses both float32 and float16 data types to optimize memory usage and speed. Your class should have an `__init__(self, loss_scale=1024.0)` method to initialize with loss scaling factor. Implement `forward(self, weights, inputs, targets)` to perform forward pass with float16 computation and return Mean Squared Error (MSE) loss (scaled) in float32, and `backward(self, gradients)` to unscale gradients and check for overflow. Use float16 for computations but float32 for gradient accumulation. Return gradients as float32 and set them to zero if overflow is detected. Only use NumPy.

### Mathematical Derivation & Intuition
# **Mixed Precision Training**
## **1. Definition**
Mixed Precision Training is a **deep learning optimization technique** that uses both **float16** (half precision) and **float32** (single precision) data types during training to reduce memory usage and increase training speed while maintaining model accuracy.
The technique works by:
- **Using float16 for forward pass computations** to save memory and increase speed
- **Using float32 for gradient accumulation** to maintain numerical precision
- **Applying loss scaling** to prevent gradient underflow in float16
---
## **2. Key Components**
### **Mean Squared Error (MSE) Loss**
The loss function must be computed as Mean Squared Error:
$$
\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
$$
where $y_i$ is the target and $\hat{y}_i$ is the prediction for sample $i$.

### **Loss Scaling**
To prevent gradient underflow in float16, gradients are scaled up during the forward pass:
$$
\text{scaled\_loss} = \text{MSE} \times \text{scale\_factor}
$$
Then unscaled during backward pass:
$$
\text{gradient} = \frac{\text{scaled\_gradient}}{\text{scale\_factor}}
$$
### **Overflow Detection**
Check for invalid gradients (NaN or Inf) that indicate numerical overflow:
$$
\text{overflow} = \text{any}(\text{isnan}(\text{gradients}) \text{ or } \text{isinf}(\text{gradients}))
$$
---
## **3. Precision Usage**
- **float16**: Forward pass computations, activations, temporary calculations
- **float32**: Gradient accumulation, parameter updates, loss scaling
- **Automatic casting**: Convert between precisions as needed
- **Loss computation**: Use MSE as the loss function before scaling
---
## **4. Benefits and Applications**
- **Memory Efficiency**: Reduces memory usage by ~50% for activations
- **Speed Improvement**: Faster computation on modern GPUs with Tensor Cores
- **Training Stability**: Loss scaling prevents gradient underflow
- **Model Accuracy**: Maintains comparable accuracy to full precision training
Common in training large neural networks where memory is a constraint and speed is critical.
---

### Verified Pure NumPy Solution (From Scratch)
```python
import numpy as np

class MixedPrecision:
    def __init__(self, loss_scale=1024.0):
        self.loss_scale = loss_scale

    def forward(self, weights, inputs, targets):
        # Convert ALL inputs to float16 for computation (regardless of input dtype)
        weights_fp16 = weights.astype(np.float16)
        inputs_fp16 = inputs.astype(np.float16)
        targets_fp16 = targets.astype(np.float16)

        # Simple forward pass: linear model + MSE loss
        predictions = np.dot(inputs_fp16, weights_fp16)
        loss = np.mean((targets_fp16 - predictions) ** 2)

        # Scale loss and convert back to float32 (Python float)
        scaled_loss = float(loss) * self.loss_scale
        return scaled_loss

    def backward(self, gradients):
        # Convert gradients to float32 for precision (regardless of input dtype)
        gradients_fp32 = gradients.astype(np.float32)

        # Check for overflow (NaN or Inf)
        overflow = np.any(np.isnan(gradients_fp32)) or np.any(np.isinf(gradients_fp32))

        if overflow:
            # Return zero gradients if overflow detected (must be float32)
            return np.zeros_like(gradients_fp32, dtype=np.float32)

        # Unscale gradients (ensure result is float32)
        unscaled_gradients = gradients_fp32 / self.loss_scale
        return unscaled_gradients.astype(np.float32)
```

### Verified Sample Trace Data
```json
{
  "test": "import numpy as np\nmp = MixedPrecision(loss_scale=1024.0)\nweights = np.array([0.5, -0.3], dtype=np.float32)\ninputs = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)\ntargets = np.array([1.0, 0.0], dtype=np.float32)\nloss = mp.forward(weights, inputs, targets)\nprint(f\"Loss: {loss:.4f}\")\nprint(f\"Loss dtype: {type(loss).__name__}\")",
  "expected_output": "Loss: 665.0000\nLoss dtype: float"
}
```

---

## Problem 173: Implement K-Nearest Neighbors
- **Difficulty**: `medium`
- **Category**: Machine Learning

### Description & Spec
## Problem

Given a list of points in n-dimensional space represented as tuples and a query point, implement a function to find the k nearest neighbors to the query point using Euclidean distance.

### Mathematical Derivation & Intuition
## Solution Explanation

The key insight is to use numpy's vectorized operations to efficiently calculate distances between the query point and all data points simultaneously, then select the k smallest distances.

## Algorithm Steps:

Convert to numpy arrays - Transform the input tuples into numpy arrays for vectorized operations
Calculate distances - Use broadcasting to compute Euclidean distance from query point to all points at once
Find k nearest - Use np.argsort() to get indices of points sorted by distance, then take first k
Return as tuples - Convert the selected points back to tuple format

### Verified Pure NumPy Solution (From Scratch)
```python
def k_nearest_neighbors(points, query_point, k):
    """
    Find k nearest neighbors to a query point
    
    Args:
        points: List of tuples representing points [(x1, y1), (x2, y2), ...]
        query_point: Tuple representing query point (x, y)
        k: Number of nearest neighbors to return
    
    Returns:
        List of k nearest neighbor points as tuples
    """
    if not points or k <= 0:
        return []
    
    if k > len(points):
        k = len(points)
    
    # Convert to numpy arrays for vectorized operations
    points_array = np.array(points)
    query_array = np.array(query_point)
    
    # Calculate Euclidean distances using broadcasting
    distances = np.sqrt(np.sum((points_array - query_array) ** 2, axis=1))
    
    # Get indices of k smallest distances
    k_nearest_indices = np.argsort(distances)[:k]
    
    # Return the k nearest points as tuples
    return [tuple(points_array[i]) for i in k_nearest_indices]
```

### Verified Sample Trace Data
```json
{
  "test": "print(k_nearest_neighbors([(1, 2), (3, 4), (1, 1), (5, 6), (2, 3)], (2, 2), 3))",
  "expected_output": "[(1, 2), (2, 3), (1, 1)]"
}
```

---

## Problem 186: Gaussian Process for Regression
- **Difficulty**: `medium`
- **Category**: Machine Learning

### Description & Spec
## Problem

Problem Statement: Task is to implement GaussianProcessRegression class which is a guassian process model for prediction regression problems.

### Mathematical Derivation & Intuition
# **Gaussian Processes (GP): From-Scratch Regression Example**

## **1. What’s a Gaussian Process?**

A **Gaussian Process** defines a distribution over functions $f(\cdot)$.
For any finite set of inputs $( X = {x_i}_{i=1}^n )$, the function values $f(X)$ follow a multivariate normal:

$$
f(X) \sim \mathcal{N}\big(0,; K(X,X)\big)
$$

where ( K ) is a **kernel** (covariance) function encoding similarity between inputs.
With noisy targets $( y = f(X) + \varepsilon, \varepsilon \sim \mathcal{N}(0,\sigma_n^2 I) )$,
GP regression yields a closed-form posterior predictive mean and variance at new points $( X_* )$.

---

## **2. The Implementation at a Glance**

The provided code builds a minimal yet complete GP regression stack:

* **Kernels implemented**

  * Radial Basis Function (RBF / Squared Exponential)
  * Matérn $(( \nu = 0.5, 1.5, 2.5 ), or general ( \nu ))$
  * Periodic
  * Linear
  * Rational Quadratic

* **Core GP classes**

  * `_GaussianProcessBase`: kernel selection & covariance matrix computation
  * `GaussianProcessRegression`:

    * `fit`: $builds ( K )$, does **Cholesky decomposition**, $solves ( \alpha )$
    * `predict`: returns posterior mean & variance
    * `log_marginal_likelihood`: computes GP evidence
    * `optimize_hyperparameters`: basic optimizer (for RBF hyperparams)

---

## **3. Kernel Cheat-Sheet**

Let $( x, x' \in \mathbb{R}^d ), ( r = \lVert x - x' \rVert )$.

* **RBF (SE):**
  $$
  k_{\text{RBF}}(x,x') = \sigma^2 \exp!\left(-\tfrac{1}{2}\tfrac{r^2}{\ell^2}\right)
  $$

* **Matérn (( \nu = 1.5 )):**
  $$
  k(x,x') = \Big(1 + \tfrac{\sqrt{3},r}{\ell}\Big)\exp!\Big(-\tfrac{\sqrt{3},r}{\ell}\Big)
  $$

* **Periodic:**
  $$
  k(x,x') = \sigma^2 \exp!\left(-\tfrac{2}{\ell^2}\sin^2!\Big(\tfrac{\pi r}{p}\Big)\right)
  $$

* **Linear:**
  $$
  k(x,x') = \sigma_b^2 + \sigma_v^2,x^\top x'
  $$

* **Rational Quadratic:**
  $$
  k(x,x') = \sigma^2\Big(1 + \tfrac{r^2}{2\alpha \ell^2}\Big)^{-\alpha}
  $$

---

## **4. GP Regression Mechanics**

### **Training**

1. Build covariance:
   $$
   K = K(X,X) + \sigma_n^2 I
   $$

2. Cholesky factorization:
   $$
   K = L L^\top
   $$

3. Solve ( \alpha ):
   $$
   L L^\top \alpha = y
   $$

### **Prediction**

At new inputs ( X_* ):

* $( K_* = K(X, X_*) ), ( K_{**} = K(X_*, X_*) )$

* **Mean:**
  $$
  \mu_* = K_*^\top \alpha
  $$

* **Covariance:**
  $$
  \Sigma_* = K_{**} - V^\top V, \quad V = L^{-1} K_*
  $$

### **Model Selection**

* **Log Marginal Likelihood (LML):**
  $$
  \log p(y \mid X) = -\tfrac{1}{2} y^\top \alpha - \sum\nolimits_i \log L_{ii} - \tfrac{n}{2}\log(2\pi)
  $$

---

## **5. Worked Example (Linear Kernel)**

```python
import numpy as np
gp = GaussianProcessRegression(kernel='linear',
                               kernel_params={'sigma_b': 0.0, 'sigma_v': 1.0},
                               noise=1e-8)

X_train = np.array([[1], [2], [4]])
y_train = np.array([3, 5, 9])   # y = 2x + 1
gp.fit(X_train, y_train)

X_test = np.array([[3.0]])
mu = gp.predict(X_test)
print(f"{mu[0]:.4f}")   # -> 7.0000
```

---

## **6. When to Use GP Regression**

* **Small-to-medium datasets** where uncertainty estimates are valuable
* Cases requiring **predictive intervals** (not just point predictions)
* **Nonparametric modeling** with kernel priors
* Automatic hyperparameter tuning via **marginal likelihood**

---

## **7. Practical Tips**

* Always add **jitter** $10^{-6}$ to the diagonal for numerical stability
* **Standardize inputs/outputs** before training
* Be aware: Exact GP has complexity **$\mathcal{O}(n^3)$** in time and **$\mathcal{O}(n^2)$** in memory
* Choose kernels to match problem structure:

  * **RBF:** smooth functions
  * **Matérn:** rougher functions
  * **Periodic:** seasonal/cyclical data
  * **Linear:** global linear trends

### Verified Pure NumPy Solution (From Scratch)
```python
# ---------------------------------------- utf-8 encoding ---------------------------------
# This file contains Gaussian Process implementation.
import numpy as np
import math
from scipy.spatial.distance import euclidean
from scipy.special import kv as bessel_kv
from scipy.special import gamma
from scipy.linalg import cholesky, solve_triangular
from scipy.optimize import minimize
from scipy.special import expit, softmax


# --- KERNEL FUNCTIONS --------------------------------------------------------
def matern_kernel(x: np.ndarray, x_prime: np.ndarray, length_scale=1.0, nu=1.5):
    d = euclidean(x, x_prime)
    if d == 0:
        return 1.0  # Covariance with self is 1 before scaling
    if nu == 0.5:
        return np.exp(-d / length_scale)
    elif nu == 1.5:
        return (1 + np.sqrt(3) * d / length_scale) * np.exp(
            -np.sqrt(3) * d / length_scale
        )
    elif nu == 2.5:
        return (
            1 + np.sqrt(5) * d / length_scale + 5 * d**2 / (3 * length_scale**2)
        ) * np.exp(-np.sqrt(5) * d / length_scale)
    else:
        factor = (2 ** (1 - nu)) / gamma(nu)
        scaled_d = np.sqrt(2 * nu) * d / length_scale
        return factor * (scaled_d**nu) * bessel_kv(nu, scaled_d)


def rbf_kernel(x: np.ndarray, x_prime, sigma=1.0, length_scale=1.0):
    # This is a squared exponential kernel

    # Calculate the squared euclidean distance
    sq_norm = np.linalg.norm(x - x_prime) ** 2

    # Correctly implement the formula
    return sigma**2 * np.exp(-sq_norm / (2 * length_scale**2))


def periodic_kernel(
    x: np.ndarray, x_prime: np.ndarray, sigma=1.0, length_scale=1.0, period=1.0
):
    return sigma**2 * np.exp(
        -2 * np.sin(np.pi * np.linalg.norm(x - x_prime) / period) ** 2 / length_scale**2
    )


def linear_kernel(x: np.ndarray, x_prime: np.ndarray, sigma_b=1.0, sigma_v=1.0):
    return sigma_b**2 + sigma_v**2 * np.dot(x, x_prime)


def rational_quadratic_kernel(
    x: np.ndarray, x_prime: np.ndarray, sigma=1.0, length_scale=1.0, alpha=1.0
):
    return sigma**2 * (
        1 + np.linalg.norm(x - x_prime) ** 2 / (2 * alpha * length_scale**2)
    ) ** (-alpha)


# --- BASE CLASS -------------------------------------------------------------


class _GaussianProcessBase:
    def __init__(self, kernel="rbf", noise=1e-5, kernel_params=None):
        self.kernel_name = kernel
        self.noise = noise
        self.kernel_params = kernel_params if kernel_params else {}
        self.X_train = None
        self.y_train = None
        self.K = None

    def _select_kernel(self, x1, x2):
        """Selects and computes the kernel value for two single data points."""
        if self.kernel_name == "rbf":
            return rbf_kernel(x1, x2, **self.kernel_params)
        elif self.kernel_name == "matern":
            return matern_kernel(x1, x2, **self.kernel_params)
        elif self.kernel_name == "periodic":
            return periodic_kernel(x1, x2, **self.kernel_params)
        elif self.kernel_name == "linear":
            return linear_kernel(x1, x2, **self.kernel_params)
        elif self.kernel_name == "rational_quadratic":
            return rational_quadratic_kernel(x1, x2, **self.kernel_params)
        else:
            raise ValueError(
                "Unsupported kernel. Choose from ['rbf', 'matern', 'periodic', 'linear', 'rational_quadratic']."
            )

    def _compute_covariance(self, X1, X2):
        """
        Computes the covariance matrix between two sets of points.
        This method fixes the vectorization bug from the original code.
        """
        # Ensuring X1 and X2 are 2D arrays
        X1 = np.atleast_2d(X1)
        X2 = np.atleast_2d(X2)

        n1, _ = X1.shape
        n2, _ = X2.shape
        K = np.zeros((n1, n2))
        for i in range(n1):
            for j in range(n2):
                K[i, j] = self._select_kernel(X1[i], X2[j])
        return K


# --- REGRESSION MODEL -------------------------------------------------------
class GaussianProcessRegression(_GaussianProcessBase):
    def fit(self, X, y):
        self.X_train = np.asarray(X)
        self.y_train = np.asarray(y)
        self.K = self._compute_covariance(
            self.X_train, self.X_train
        ) + self.noise * np.eye(len(self.X_train))

        # Compute Cholesky decomposition for stable inversion
        self.L = cholesky(self.K, lower=True)
        # alpha = K_inv * y
        self.alpha = solve_triangular(
            self.L.T, solve_triangular(self.L, self.y_train, lower=True)
        )

    def predict(self, X_test, return_std=False):
        X_test = np.atleast_2d(X_test)
        K_s = self._compute_covariance(self.X_train, X_test)
        K_ss = self._compute_covariance(X_test, X_test)

        # Compute predictive mean
        mu = K_s.T @ self.alpha

        # Compute predictive variance
        v = solve_triangular(self.L, K_s, lower=True)
        cov = K_ss - v.T @ v

        if return_std:
            return mu, np.sqrt(np.diag(cov))
        return mu

    def log_marginal_likelihood(self):
        return (
            -0.5 * (self.y_train.T @ self.alpha)
            - np.sum(np.log(np.diag(self.L)))
            - len(self.X_train) / 2 * np.log(2 * np.pi)
        )

    def optimize_hyperparameters(self):
        # NOTE: This is a simplified optimizer for 'rbf' kernel's params.
        def objective(params):
            self.kernel_params = {
                "length_scale": np.exp(params[0]),
                "sigma": np.exp(params[1]),
            }
            self.fit(self.X_train, self.y_train)
            return -self.log_marginal_likelihood()

        init_params = np.log(
            [
                self.kernel_params.get("length_scale", 1.0),
                self.kernel_params.get("sigma", 1.0),
            ]
        )
        res = minimize(
            objective, init_params, method="L-BFGS-B", bounds=[(-5, 5), (-5, 5)]
        )

        self.kernel_params = {
            "length_scale": np.exp(res.x[0]),
            "sigma": np.exp(res.x[1]),
        }
        # Re-fit with optimal hyperparameters
        self.fit(self.X_train, self.y_train)


if __name__ == "__main__":
    gp = GaussianProcessRegression(
        kernel="linear", kernel_params={"sigma_b": 0.0, "sigma_v": 1.0}, noise=1e-8
    )
    X_train = np.array([[1], [2], [4]])
    y_train = np.array([3, 5, 9])
    gp.fit(X_train, y_train)
    X_test = np.array([[3.0]])
    mu = gp.predict(X_test)
```

### Verified Sample Trace Data
```json
{
  "test": "import numpy as np\ngp = GaussianProcessRegression(kernel='rbf', kernel_params={'sigma': 1.0, 'length_scale': 1.0}, noise=1e-8)\nX_train = np.array([[0], [2.5], [5.0], [7.5], [10.0]])\ny_train = np.sin(X_train).ravel()\ngp.fit(X_train, y_train)\nX_test = np.array([[1.25]])\nmu = gp.predict(X_test)\nprint(f\"{mu[0]:.4f}\")",
  "expected_output": "0.2814"
}
```

---

## Problem 188: Gradient Checkpointing
- **Difficulty**: `easy`
- **Category**: Machine Learning

### Description & Spec
## Problem

Write a Python function `checkpoint_forward` that takes a list of numpy functions (each representing a layer or operation) and an input numpy array, and returns the final output by applying each function in sequence. To simulate gradient checkpointing, the function should not store intermediate activations; instead, it should recompute them as needed (for this problem, just apply the functions in sequence as usual). Only use standard Python and numpy. The returned array should be of type float and have the same shape as the output of the last function.

### Mathematical Derivation & Intuition
# **Gradient Checkpointing**

## **1. Definition**
Gradient checkpointing is a technique used in deep learning to reduce memory usage during training by selectively storing only a subset of intermediate activations (checkpoints) and recomputing the others as needed during the backward pass. This allows training of larger models or using larger batch sizes without exceeding memory limits.

## **2. Why Use Gradient Checkpointing?**
* **Reduce Memory Usage:** By storing fewer activations, memory requirements are reduced, enabling training of deeper or larger models.
* **Enable Larger Batches/Models:** Makes it possible to fit larger models or use larger batch sizes on limited hardware.
* **Tradeoff:** The main tradeoff is increased computation time, as some activations must be recomputed during the backward pass.

## **3. Gradient Checkpointing Mechanism**
Suppose a model consists of $N$ layers, each represented by a function $f_i$. Normally, the forward pass stores all intermediate activations:

$$
A_0 = x \\
A_1 = f_1(A_0) \\
A_2 = f_2(A_1) \\
\ldots \\
A_N = f_N(A_{N-1})
$$

With gradient checkpointing, only a subset of $A_i$ are stored (the checkpoints). The others are recomputed as needed during backpropagation. In the simplest case, you can store only the input and output, and recompute all intermediates when needed.

**Example:**
If you have three functions $f_1, f_2, f_3$ and input $x$:
* Forward: $A_1 = f_1(x)$, $A_2 = f_2(A_1)$, $A_3 = f_3(A_2)$
* With checkpointing, you might only store $x$ and $A_3$, and recompute $A_1$ and $A_2$ as needed.

## **4. Applications of Gradient Checkpointing**
Gradient checkpointing is widely used in training:
* **Very Deep Neural Networks:** Transformers, ResNets, and other architectures with many layers.
* **Large-Scale Models:** Language models, vision models, and more.
* **Memory-Constrained Environments:** When hardware cannot fit all activations in memory.
* **Any optimization problem** where memory is a bottleneck during training.

Gradient checkpointing is a powerful tool to enable training of large models on limited hardware, at the cost of extra computation.

### Verified Pure NumPy Solution (From Scratch)
```python
import numpy as np

def checkpoint_forward(funcs, input_arr):
    """
    Applies a list of functions in sequence to the input array, simulating gradient checkpointing by not storing intermediates.

    Args:
        funcs (list of callables): List of functions to apply in sequence.
        input_arr (np.ndarray): Input numpy array.

    Returns:
        np.ndarray: The output after applying all functions, same shape as output of last function.
    """
    x = input_arr
    for f in funcs:
        x = f(x)
    return x.astype(float)
```

### Verified Sample Trace Data
```json
{
  "test": "import numpy as np\ndef f1(x): return x + 1\ndef f2(x): return x * 2\ndef f3(x): return x - 3\nfuncs = [f1, f2, f3]\ninput_arr = np.array([1.0, 2.0])\nprint(checkpoint_forward(funcs, input_arr))",
  "expected_output": "[1. 3.]"
}
```

---

## Problem XXX: TITLE GOES HERE
- **Difficulty**: `medium`
- **Category**: Machine Learning

### Description & Spec
## Problem

Write a concise problem description here.

### Mathematical Derivation & Intuition
## Solution Explanation

Add intuition, math, and step-by-step reasoning here.

### Writing Mathematical Expressions with LaTeX

This editor supports LaTeX for rendering mathematical equations and expressions. Here's how you can use it:

1. **Inline Math**: 
   - Wrap your expression with single `$` symbols.
   - Example: `$E = mc^2$` → Renders as: ( $E = mc^2$ )

2. **Block Math**:
   - Wrap your expression with double `$$` symbols.
   - Example: 
     ```
     $$
     \int_a^b f(x) \, dx
     $$
     ```
     Renders as:
     $$
     \int_a^b f(x) \, dx
     $$

3. **Math Functions**:
   - Use standard LaTeX functions like `\frac`, `\sqrt`, `\sum`, etc.
   - Examples:
     - `$\frac{a}{b}$` → ( $\frac{a}{b}$ )
     - `$\sqrt{x}$` → ( $\sqrt{x}$ )

4. **Greek Letters and Symbols**:
   - Use commands like `\alpha`, `\beta`, etc., for Greek letters.
   - Example: `$\alpha + \beta = \gamma$` → ( $\alpha + \beta = \gamma$ )

5. **Subscripts and Superscripts**:
   - Use `_{}` for subscripts and `^{}` for superscripts.
   - Examples:
     - `$x_i$` → ( $x_i$ )
     - `$x^2$` → ( $x^2$ )

6. **Combined Examples**:
   - `$\sum_{i=1}^n i^2 = \frac{n(n+1)(2n+1)}{6}$`
     Renders as:
     $\sum_{i=1}^n i^2 = \frac{n(n+1)(2n+1)}{6}$

Feel free to write your own mathematical expressions, and they will be rendered beautifully in the preview!

### Verified Pure NumPy Solution (From Scratch)
```python
def your_function(...):
    # reference implementation
    ...
```

### Verified Sample Trace Data
```json
{
  "test": "print(your_function(...))",
  "expected_output": "..."
}
```

---

