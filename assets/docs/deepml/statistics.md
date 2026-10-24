# Deep-ML Knowledge Base: Statistics

Total verified problems: 4

Source: [Deep-ML (Open-Deep-ML)](https://www.deep-ml.com/)

---

## Problem 10: Calculate Covariance Matrix
- **Difficulty**: `easy`
- **Category**: Statistics

### Description & Spec
Write a Python function to calculate the covariance matrix for a given set of vectors. The function should take a list of lists, where each inner list represents a feature with its observations, and return a covariance matrix as a list of lists. Additionally, provide test cases to verify the correctness of your implementation.

### Mathematical Derivation & Intuition
## Understanding Covariance Matrix

The covariance matrix is a fundamental concept in statistics and machine learning, used to understand the relationship between multiple variables (features) in a dataset. It quantifies the degree to which two variables change together.

### Key Concepts

- **Covariance**: Measures the directional relationship between two random variables. A positive covariance indicates that the variables increase together, while a negative covariance indicates that one variable increases as the other decreases.
- **Covariance Matrix**: For a dataset with $n$ features, the covariance matrix is an $n \times n$ matrix where each element $(i, j)$ represents the covariance between the $i^{th}$ and $j^{th}$ features.

### Covariance Formula

The covariance between two variables $X$ and $Y$ is calculated as:

$$
\text{cov}(X, Y) = \frac{\sum_{k=1}^{m} (X_k - \bar{X})(Y_k - \bar{Y})}{m - 1}
$$

Where:

- $X_k$ and $Y_k$ are the individual observations of variables $X$ and $Y$.
- $\bar{X}$ and $\bar{Y}$ are the means of $X$ and $Y$.
- $m$ is the number of observations.

### Constructing the Covariance Matrix

Given a dataset with $n$ features, the covariance matrix is constructed as follows:

1. **Calculate the Mean**: Compute the mean of each feature.
2. **Compute Covariance**: For each pair of features, calculate the covariance using the formula above.
3. **Populate the Matrix**: Place the computed covariance values in the corresponding positions in the matrix. The diagonal elements represent the variance of each feature.

$$
\text{Covariance Matrix} =
\begin{bmatrix}
\text{cov}(X_1, X_1) & \text{cov}(X_1, X_2) & \cdots & \text{cov}(X_1, X_n) \\
\text{cov}(X_2, X_1) & \text{cov}(X_2, X_2) & \cdots & \text{cov}(X_2, X_n) \\
\vdots & \vdots & \ddots & \vdots \\
\text{cov}(X_n, X_1) & \text{cov}(X_n, X_2) & \cdots & \text{cov}(X_n, X_n) \\
\end{bmatrix}
$$

### Example Calculation

Consider the following dataset with two features:

$$
\begin{align*}
\text{Feature 1} &: [1, 2, 3] \\
\text{Feature 2} &: [4, 5, 6]
\end{align*}
$$

1. **Calculate Means**:
   $$
   \bar{X}_1 = \frac{1 + 2 + 3}{3} = 2.0 \\
   \bar{X}_2 = \frac{4 + 5 + 6}{3} = 5.0
   $$

2. **Compute Covariances**:
   $$
   \text{cov}(X_1, X_1) = \frac{(1-2)^2 + (2-2)^2 + (3-2)^2}{3-1} = 1.0 \\
   \text{cov}(X_1, X_2) = \frac{(1-2)(4-5) + (2-2)(5-5) + (3-2)(6-5)}{3-1} = 1.0 \\
   \text{cov}(X_2, X_2) = \frac{(4-5)^2 + (5-5)^2 + (6-5)^2}{3-1} = 1.0
   $$

3. **Covariance Matrix**:
   $$
   \begin{bmatrix}
   1.0 & 1.0 \\
   1.0 & 1.0 
   \end{bmatrix}
   $$

### Applications

Covariance matrices are widely used in various fields, including:

- **Principal Component Analysis (PCA)**: Reducing the dimensionality of datasets while preserving variance.
- **Portfolio Optimization**: Understanding the variance and covariance between different financial assets.
- **Multivariate Statistics**: Analyzing the relationships between multiple variables simultaneously.

Understanding the covariance matrix is crucial for interpreting the relationships in multivariate data and for performing advanced statistical analyses.

### Verified Pure NumPy Solution (From Scratch)
```python
import numpy as np

def calculate_covariance_matrix(vectors: list[list[float]]) -> list[list[float]]:
    n_observations = len(vectors)
    n_features = len(vectors[0])
    covariance_matrix = [[0 for _ in range(n_observations)] for _ in range(n_observations)]

    means = [sum(feature) / n_features for feature in vectors]

    for i in range(n_observations):
        for j in range(i, n_observations):
            covariance = sum((vectors[i][k] - means[i]) * (vectors[j][k] - means[j]) for k in range(n_features)) / (n_features - 1)
            covariance_matrix[i][j] = covariance_matrix[j][i] = covariance

    return covariance_matrix
```

### Verified Sample Trace Data
```json
{
  "test": "print(calculate_covariance_matrix([[1, 2, 3], [4, 5, 6]]))",
  "expected_output": "[[1.0, 1.0], [1.0, 1.0]]"
}
```

---

## Problem 78: Descriptive Statistics Calculator
- **Difficulty**: `easy`
- **Category**: Statistics

### Description & Spec
Write a Python function to calculate various descriptive statistics metrics for a given dataset. The function should take a list or NumPy array of numerical values and return a dictionary containing mean, median, mode, variance, standard deviation, percentiles (25th, 50th, 75th), and interquartile range (IQR).

### Mathematical Derivation & Intuition
## Understanding Descriptive Statistics

Descriptive statistics provide a summary of data through various measures and help understand the basic structure and distribution of data.

### Key Metrics

Descriptive statistics cover several key metrics, including mean, median, mode, variance, standard deviation, percentiles, quartiles, and interquartile range.

- **Mean**: The average of all values in the dataset, calculated by summing all the values and dividing by the count of values.
- **Median**: The middle value when the data is sorted. If there is an even number of values, the median is the average of the two middle values.
- **Mode**: The value that occurs most frequently in the dataset.
- **Variance**: A measure of how much the values in the dataset deviate from the mean, calculated as:
  $$
  \text{Variance} = \frac{\sum (x_i - \bar{x})^2}{N}
  $$
  where \(x_i\) are the data points, \(\bar{x}\) is the mean, and \(N\) is the number of data points.
- **Standard Deviation**: The square root of the variance, calculated as:
  $$
  \text{Standard Deviation} = \sqrt{\frac{\sum (x_i - \bar{x})^2}{N}}
  $$
- **Percentiles and Quartiles**: Percentiles divide the data into 100 equal parts, while quartiles divide the data into four equal parts. The 25th, 50th (median), and 75th percentiles are the common quartiles.
- **Interquartile Range (IQR)**: The difference between the 75th and 25th percentiles, calculated as:
  $$
  \text{IQR} = Q_3 - Q_1
  $$
  where \(Q_3\) is the 75th percentile and \(Q_1\) is the 25th percentile.

### Example Calculation

Given data:

Data set: \([12, 15, 12, 18, 19, 17, 15, 14, 16, 18]\)

- **Mean**:
  $$
  \text{Mean} = \frac{\sum x_i}{N} = \frac{156}{10} = 15.6
  $$
- **Median**:
  $$
  \text{Median} = 15.5
  $$
- **Mode**:
  $$
  \text{Mode} = 12, 15, 18
  $$
- **Variance**:
  $$
  \text{Variance} = \frac{\sum (x_i - 15.6)^2}{9} \approx 6.93
  $$
- **Standard Deviation**:
  $$
  \text{Standard Deviation} = \sqrt{6.93} \approx 2.63
  $$
- **Percentiles**:
  $$
  25\% \text{ Percentile} = 13.5, \quad 50\% \text{ Percentile} = 15.5 ,\quad 75\% \text{ Percentile} = 17.5
  $$
- **Interquartile Range**:
  $$
  \text{IQR} = 17.5 - 13.5 = 4
  $$

### Applications

Descriptive statistics are widely used in:

- Data Analysis
- Exploratory Data Analysis (EDA)
- Understanding Data Distributions
- Feature Engineering
- Identifying Outliers

These metrics are foundational in data science, helping to interpret data and prepare it for further analysis in machine learning.

### Verified Pure NumPy Solution (From Scratch)
```python
import numpy as np

def descriptive_statistics(data):
    """
    Calculate various descriptive statistics metrics for a given dataset.
    :param data: List or numpy array of numerical values
    :return: Dictionary containing mean, median, mode, variance, standard deviation,
             percentiles (25th, 50th, 75th), and interquartile range (IQR)
    """
    # Ensure data is a numpy array for easier calculations
    data = np.array(data)

    # Mean
    mean = np.mean(data)

    # Median
    median = np.median(data)

    # Mode
    unique, counts = np.unique(data, return_counts=True)
    mode = unique[np.argmax(counts)] if len(data) > 0 else None

    # Variance
    variance = np.var(data)

    # Standard Deviation
    std_dev = np.sqrt(variance)

    # Percentiles (25th, 50th, 75th)
    percentiles = np.percentile(data, [25, 50, 75])

    # Interquartile Range (IQR)
    iqr = percentiles[2] - percentiles[0]

    # Compile results into a dictionary
    stats_dict = {
        "mean": mean,
        "median": median,
        "mode": mode,
        "variance": np.round(variance,4),
        "standard_deviation": np.round(std_dev,4),
        "25th_percentile": percentiles[0],
        "50th_percentile": percentiles[1],
        "75th_percentile": percentiles[2],
        "interquartile_range": iqr
    }

    return stats_dict
```

### Verified Sample Trace Data
```json
{
  "test": "print(descriptive_statistics([10, 20, 30, 40, 50]))",
  "expected_output": "{'mean': 30.0, 'median': 30.0, 'mode': 10, 'variance': 200.0, 'standard_deviation': 14.1421, '25th_percentile': 20.0, '50th_percentile': 30.0, '75th_percentile': 40.0, 'interquartile_range': 20.0}"
}
```

---

## Problem 95: Calculate the Phi Coefficient
- **Difficulty**: `easy`
- **Category**: Statistics

### Description & Spec
Implement a function to calculate the Phi coefficient, a measure of the correlation between two binary variables. The function should take two lists of integers (0s and 1s) as input and return the Phi coefficient rounded to 4 decimal places.

### Mathematical Derivation & Intuition
# The Phi coefficient

The Phi coefficient is a type of correlation coefficient , which is used when we need to find the correlation between two binary variables.
For example when we have two variables x and y , x being gender and y signifying the presence of heart disease. 
Both variables are binary and if we need to find a correlation between x and y then we can use the formula below :


$\phi = \frac{(x_{00} \cdot x_{11}) - (x_{01} \cdot x_{10})}{\sqrt{(x_{00} + x_{01})(x_{10} + x_{11})(x_{00} + x_{10})(x_{01} + x_{11})}}$




### Explanation of Terms:
- \(x_00\): The number of cases where \(x = 0\) and \(y = 0\).
- \(x_01\): The number of cases where \(x = 0\) and \(y = 1\).
- \(x_10\): The number of cases where \(x = 1\) and \(y = 0\).
- \(x_11\): The number of cases where \(x = 1\) and \(y = 1\).

### Verified Pure NumPy Solution (From Scratch)
```python
def phi_corr(x: list[int], y: list[int]) -> float:
    """
    Calculate the Phi coefficient between two binary variables.

    Args:
    x (list[int]): A list of binary values (0 or 1).
    y (list[int]): A list of binary values (0 or 1).

    Returns:
    float: The Phi coefficient rounded to 4 decimal places.
    """
    x1y1 = x1y0 = x0y1 = x0y0 = 0

    # Count occurrences
    for i in range(len(x)):
        if x[i] == 1:
            if y[i] == 1:
                x1y1 += 1
            else:
                x1y0 += 1
        elif x[i] == 0:
            if y[i] == 1:
                x0y1 += 1
            else:
                x0y0 += 1

    # Calculate numerator and denominator
    numerator = (x0y0 * x1y1) - (x0y1 * x1y0)
    denominator = ((x0y0 + x0y1) * (x1y0 + x1y1) * (x0y0 + x1y0) * (x0y1 + x1y1)) ** 0.5

    if denominator == 0:
        return 0.0

    phi = numerator / denominator
    return round(phi, 4)
```

### Verified Sample Trace Data
```json
{
  "test": "print(phi_corr([1, 1, 0, 0], [0, 0, 1, 1]))",
  "expected_output": "-1.0"
}
```

---

## Problem 120: Bhattacharyya Distance Between Two Distributions
- **Difficulty**: `easy`
- **Category**: Statistics

### Description & Spec
Implement a function to calculate the Bhattacharyya distance between two probability distributions. The function should take two lists representing discrete probability distributions `p` and `q`, and return the Bhattacharyya distance rounded to 4 decimal places. If the inputs have different lengths or are empty, return 0.0.

### Mathematical Derivation & Intuition
## Understanding Bhattacharyya Distance

**Bhattacharyya Distance (BD)** is a concept in statistics used to measure the **similarity** or **overlap** between two probability distributions **P(x)** and **Q(x)** on the same domain **x**.  

This differs from **KL Divergence**, which measures the **loss of information** when projecting one probability distribution onto another (reference distribution).  

### **Bhattacharyya Distance Formula**
The Bhattacharyya distance is defined as:  

$$
BC (P, Q) = \sum \sqrt{P(X) \cdot Q(X)}
$$

$$
BD (P, Q) = -\ln(BC (P, Q))
$$

where **BC (P, Q)** is the **Bhattacharyya coefficient**.  

### **Key Properties**
1. **BD is always non-negative**:  
   $$ BD \geq 0 $$
2. **Symmetric in nature**:  
   $$ BD (P, Q) = BD (Q, P) $$
3. **Applications**:  
   - Risk assessment  
   - Stock predictions  
   - Feature scaling  
   - Classification problems  

### **Example Calculation**
Consider two probability distributions **P(x)** and **Q(x)**:  

$$
P(x) = [0.1, 0.2, 0.3, 0.4], \quad Q(x) = [0.4, 0.3, 0.2, 0.1]
$$

1. **Bhattacharyya Coefficient**:  

$$
BC (P, Q) = \sum \sqrt{P(X) \cdot Q(X)} = 0.8898
$$

2. **Bhattacharyya Distance**:  

$$
BD (P, Q) = -\ln(BC (P, Q)) = -\ln(0.8898) = 0.1166
$$

This illustrates how BD quantifies the **overlap** between two probability distributions.

### Verified Pure NumPy Solution (From Scratch)
```python
import numpy as np

def bhattacharyya_distance(p : list[float], q : list[float]) -> float:
    if len(p) != len(q):
        return 0.0

    p, q = np.array(p), np.array(q)
    BC = np.sum(np.sqrt(p * q))
    DB = -np.log(BC)
    return round(DB, 4)
```

### Verified Sample Trace Data
```json
{
  "test": "print(round(bhattacharyya_distance([0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1]),4))",
  "expected_output": "0.1166"
}
```

---

