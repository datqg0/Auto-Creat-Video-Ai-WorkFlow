# Deep-ML Knowledge Base: Calculus

Total verified problems: 2

Source: [Deep-ML (Open-Deep-ML)](https://www.deep-ml.com/)

---

## Problem 116: Derivative of a Polynomial
- **Difficulty**: `easy`
- **Category**: calculus

### Description & Spec
Implement a function that computes the derivative of a polynomial term of the form `c * x^n` at a given point `x`, where `c` is a coefficient and `n` is the exponent. The function should return the value of the derivative, accounting for the coefficient in the power rule. This is useful for understanding how polynomials change at specific points in machine learning optimization problems.

### Mathematical Derivation & Intuition
## Derivative of a Polynomial

A function's derivative is a way of quantifying the function's slope at a given point. It allows us to understand whether the function is increasing, decreasing or flat at specific input. 

Taking the derivative of a polynomial at a single point follows a straight-forward rule. This question will show the rule and the edge case you should be on the look-out for.

### Mathematical Definition

When calculating the slope of a function $f(x)$, we usually require two points $x_{1}$ and $x_{2}$ and use the following formula:

$$
\frac{f(x_{2}) - f(x_{1})}{x_{2} - x_{1}}
$$

A derivative generalizes that notion by calculating the slope of a function at a specific point.
A derivative of a function $f(x)$ is mathematically defined as:

$$
\frac{d f(x)}{d x} = \lim_{h \to 0} \frac{f(x+h) - f(x)}{h}
$$

Where:
- $x$ is the input to the function
- $h$ is the "step", which is equivalent to the difference $x_{2} - x_{1}$ in the two-point slope-formula

Taking the limit as the step grows smaller and smaller, allow us to quantify the slope at a certain point, instead of having to consider two points as in other methods of finding the slope.

When taking the derivative of a polynomial function $x^{n}$, where $n \neq 0$, then the derivative is: $n x^{n-1}$. In the special case where $n = 0$ then the derivative is zero. This is because $x^{0} = 1$ if $x \neq 0$.

A positive derivative indicates that the function is increasing in that point, a negative derivative indicates that the function is decreasing at that point. A derivative equal to zero indicates that the function is flat, which could potentially indicate a function's minimum or maximum.

### Verified Pure NumPy Solution (From Scratch)
```python
def poly_term_derivative(c: float, x: float, n: float) -> float:
    if n == 0.0:
        return 0.0
    return round(c * n * (x ** (n - 1)), 4)
```

### Verified Sample Trace Data
```json
{
  "test": "print(poly_term_derivative(2.0, 3.0, 2.0))",
  "expected_output": "12.0"
}
```

---

## Problem 127: Find Captain Redbeard's Hidden Treasure
- **Difficulty**: `medium`
- **Category**: Calculus

### Description & Spec
Captain Redbeard, the most daring pirate of the seven seas, has uncovered a mysterious ancient map. Instead of islands, it shows a strange wavy curve, and the treasure lies at the lowest point of the land! (watch out for those tricky local mins)

The land's height at any point $x$ is given by:

f(x) = x^4 - 3x^3 + 2


**Your Mission:**
Implement a Python function that finds the value of $x$ where $f(x)$ reaches its minimum, starting from any random initial position.

### Mathematical Derivation & Intuition
## How to Find the Minimum of a Function

To find the minimum of a function like

$$
f(x) = x^4 - 3x^3 + 2
$$

we can use a technique called **gradient descent**.

### Steps:

1. **Find the Derivative**
   - The derivative (slope) tells us which direction the function is increasing or decreasing.
   - For this problem, the derivative is:
     $$
     f'(x) = 4x^3 - 9x^2
     $$

2. **Move Opposite the Slope**
   - If the slope is positive, move left.
   - If the slope is negative, move right.
   - Update the position by:
     $$
     x_{new} = x_{old} - \text{learning rate} \times f'(x_{old})
     $$

3. **Repeat**
   - Keep updating $x$ until the change is very small (below a tolerance).

### Why Does This Work?
- If you always move downhill along the slope, you eventually reach a bottom a local minimum.

### Important Terms
- **Learning Rate**: How big a step to take each update.
- **Tolerance**: How close successive steps must be to stop.
- **Local Minimum**: A point where the function value is lower than nearby points.

In this problem, Captain Redbeard finds the hidden treasure by moving downhill until he reaches the lowest point!

### Verified Pure NumPy Solution (From Scratch)
```python
def find_treasure(start_x: float) -> float:
	learning_rate = 0.1, tolerance = 1e-6, max_iters = 10000)    def gradient(x):
        return 4 * x**3 - 9 * x**2  # derivative of x^4 - 3x^3 + 2

    x = start_x
    for _ in range(max_iters):
        grad = gradient(x)
        new_x = x - learning_rate * grad
        if abs(new_x - x) < tolerance:
            break
        x = new_x
    return round(x, 4)
```

### Verified Sample Trace Data
```json
{
  "test": "print(abs(find_treasure(-1) - 2.25) < 1e-2)",
  "expected_output": "True"
}
```

---

