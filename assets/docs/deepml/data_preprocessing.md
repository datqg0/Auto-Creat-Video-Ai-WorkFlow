# Deep-ML Knowledge Base: Data Preprocessing

Total verified problems: 1

Source: [Deep-ML (Open-Deep-ML)](https://www.deep-ml.com/)

---

## Problem 112: Min-Max Normalization of Feature Values
- **Difficulty**: `easy`
- **Category**: Data Preprocessing

### Description & Spec
Implement a function that performs Min-Max Normalization on a list of integers, scaling all values to the range [0, 1]. Min-Max normalization helps ensure that all features contribute equally to a model by scaling them to a common range.

### Mathematical Derivation & Intuition
## Understanding Min-Max Normalization

Min-Max Normalization is a technique used to rescale numerical data to the range $[0, 1]$.

The formula used is:

$$
X' = \frac{X - X_{\min}}{X_{\max} - X_{\min}}
$$

### Why Normalize?

- Ensures all features have equal importance regardless of their original scale.
- Commonly used in preprocessing for machine learning algorithms such as k-nearest neighbors, neural networks, and gradient descent-based models.

### Special Case

If all the elements in the input are identical, then $X_{\max} = X_{\min}$. In that case, return an array of zeros.

### Example

Given the input list `[1, 2, 3, 4, 5]`:

- Minimum: $1$
- Maximum: $5$
- The normalized values are:

$$
\begin{aligned}
&\frac{1 - 1}{4} = 0.0 \\
&\frac{2 - 1}{4} = 0.25 \\
&\frac{3 - 1}{4} = 0.5 \\
&\frac{4 - 1}{4} = 0.75 \\
&\frac{5 - 1}{4} = 1.0
\end{aligned}
$$

The result is `[0.0, 0.25, 0.5, 0.75, 1.0]`.

Remember to round the result to **4 decimal places**.

### Verified Pure NumPy Solution (From Scratch)
```python
def min_max(x: list[int]) -> list[float]:
    largest = max(x)
    smallest = min(x)
    if largest == smallest:
        return [0.0] * len(x)
    for i in range(len(x)):
        x[i] = round((x[i] - smallest) / (largest - smallest), 4)
    return x
```

### Verified Sample Trace Data
```json
{
  "test": "print(min_max([1, 2, 3, 4, 5]))",
  "expected_output": "[0.0, 0.25, 0.5, 0.75, 1.0]"
}
```

---

