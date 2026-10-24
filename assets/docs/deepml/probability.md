# Deep-ML Knowledge Base: Probability

Total verified problems: 5

Source: [Deep-ML (Open-Deep-ML)](https://www.deep-ml.com/)

---

## Problem 79: Binomial Distribution Probability
- **Difficulty**: `medium`
- **Category**: Probability

### Description & Spec
Write a Python function to calculate the probability of achieving exactly k successes in n independent Bernoulli trials, each with probability p of success, using the Binomial distribution formula.

### Mathematical Derivation & Intuition
## Understanding the Binomial Distribution

The Binomial distribution is a discrete probability distribution that models the number of successes in a fixed number of independent Bernoulli trials, each with the same probability of success.

### Mathematical Formulation

The probability of achieving exactly $k$ successes in $n$ trials is given by the formula:

$$
P(X = k) = \binom{n}{k} \cdot p^k \cdot (1-p)^{n-k}
$$

- **$n$**: Total number of trials  
- **$k$**: Number of successes  
- **$p$**: Probability of success on each trial  
- $\binom{n}{k}$: The number of ways to choose $k$ successes from $n$ trials, calculated as:

$$
\binom{n}{k} = \frac{n!}{k!(n-k)!}
$$

### Implementation Steps

1. Calculate $\binom{n}{k}$ using factorials.  
2. Raise $p$ to the power of $k$ and $(1-p)$ to the power of $(n-k)$.  
3. Multiply these results to get the probability.

### Example Calculation

Given:

- $n = 5$  
- $k = 2$  
- $p = 0.4$  

Step-by-step:

1. Calculate $\binom{n}{k}$:

$$
\binom{5}{2} = \frac{5!}{2!(5-2)!} = \frac{5 \cdot 4}{2 \cdot 1} = 10
$$

2. Calculate $p^k \cdot (1-p)^{n-k}$:

$$
0.4^2 \cdot (1-0.4)^3 = 0.16 \cdot 0.216 = 0.03456
$$

3. Multiply results:

$$
P(X = 2) = 10 \cdot 0.03456 = 0.3456
$$

The probability of exactly 2 successes is $0.3456$.

### Applications

The Binomial distribution is widely used in:

- Quality control and defect analysis  
- Survey analysis  
- Medical trials  
- Modeling success/failure experiments  

It provides insights into the likelihood of various outcomes in scenarios with two possible results (e.g., success or failure).

### Verified Pure NumPy Solution (From Scratch)
```python
import math

def binomial_probability(n, k, p):
    """
    Calculate the probability of achieving exactly k successes in n independent Bernoulli trials,
    each with probability p of success, using the Binomial distribution formula.
    :param n: Total number of trials
    :param k: Number of successes
    :param p: Probability of success on each trial
    :return: Probability of k successes in n trials
    """
    # Calculate binomial coefficient (n choose k)
    binomial_coeff = math.comb(n, k)
    # Calculate the probability using the binomial formula
    probability = binomial_coeff * (p ** k) * ((1 - p) ** (n - k))
    # Return the probability, rounded to five decimal places
    return round(probability, 5)
```

### Verified Sample Trace Data
```json
{
  "test": "print(binomial_probability(6, 2, 0.5))",
  "expected_output": "0.23438"
}
```

---

## Problem 80: Normal Distribution PDF Calculator
- **Difficulty**: `medium`
- **Category**: Probability

### Description & Spec
Write a Python function to calculate the probability density function (PDF) of the normal distribution for a given value, mean, and standard deviation. The function should use the mathematical formula of the normal distribution to return the PDF value rounded to 5 decimal places.

### Mathematical Derivation & Intuition
## Understanding Normal Distribution

The Normal Distribution, also known as the Gaussian Distribution, is a continuous probability distribution that is symmetrical and bell-shaped, representing the distribution of data around the mean.

### Key Characteristics

- **Symmetry**: The distribution is symmetric around the mean, which means the left and right halves of the graph are mirror images.
- **Mean, Median, and Mode**: In a perfectly normal distribution, the mean, median, and mode are all equal.
- **Shape**: The bell-shaped curve is defined by its mean ($\mu$) and standard deviation ($\sigma$).
- **Empirical Rule**: Approximately:
  - $68\%$ of data falls within 1 standard deviation ($\mu \pm \sigma$).
  - $95\%$ of data falls within 2 standard deviations ($\mu \pm 2\sigma$).
  - $99.7\%$ of data falls within 3 standard deviations ($\mu \pm 3\sigma$).

### Mathematical Formula

The probability density function (PDF) of a normal distribution is given by:

$$
f(x) = \frac{1}{\sqrt{2\pi\sigma^2}} e^{-\frac{(x-\mu)^2}{2\sigma^2}}
$$

- **$x$**: Random variable
- **$\mu$**: Mean of the distribution
- **$\sigma$**: Standard deviation of the distribution

### Implementation Steps

1. **Calculate the mean ($\mu$) and standard deviation ($\sigma$):**
   - $\mu = \frac{\sum x_i}{N}$
   - $\sigma = \sqrt{\frac{\sum (x_i - \mu)^2}{N}}$

2. **Use the normal distribution formula to calculate the probability density for each value:**
   - $\frac{1}{\sqrt{2\pi\sigma^2}} e^{-\frac{(x-\mu)^2}{2\sigma^2}}$

3. **Visualize the curve**:
   - Plot the calculated PDF values for a range of $x$ to visualize the bell-shaped curve.

### Example Calculation

Given:

- Data: [10, 12, 14, 16, 18, 20]

1. **Mean ($\mu$):**
   $$
   \mu = \frac{10 + 12 + 14 + 16 + 18 + 20}{6} = 15
   $$

2. **Standard Deviation ($\sigma$):**
   $$
   \sigma = \sqrt{\frac{(10-15)^2 + \dots + (20-15)^2}{6}} = \sqrt{\frac{25}{6}} \approx 2.04
   $$

3. **PDF for $x = 16$:**
   $$
   f(16) = \frac{1}{\sqrt{2\pi(2.04)^2}} e^{-\frac{(16-15)^2}{2(2.04)^2}} \approx 0.176
   $$

### Applications

The Normal Distribution is widely used in:

- Data Analysis
- Statistical Inference
- Machine Learning Algorithms
- Quality Control
- Risk Management

This distribution is crucial in fields like economics, biology, psychology, and engineering to model natural phenomena and make predictions.

### Verified Pure NumPy Solution (From Scratch)
```python
import math

def normal_pdf(x, mean, std_dev):
    """
    Calculate the probability density function (PDF) of the normal distribution.
    :param x: The value at which the PDF is evaluated.
    :param mean: The mean (μ) of the distribution.
    :param std_dev: The standard deviation (σ) of the distribution.
    :return: The PDF value for the given x.
    """
    coefficient = 1 / (math.sqrt(2 * math.pi) * std_dev)
    exponent = math.exp(-((x - mean) ** 2) / (2 * std_dev ** 2))
    return round(coefficient * exponent, 5)
```

### Verified Sample Trace Data
```json
{
  "test": "print(normal_pdf(0, 0, 1))",
  "expected_output": "0.39894"
}
```

---

## Problem 81: Poisson Distribution Probability Calculator
- **Difficulty**: `easy`
- **Category**: Probability

### Description & Spec
Write a Python function to calculate the probability of observing exactly k events in a fixed interval using the Poisson distribution formula. The function should take k (number of events) and lam (mean rate of occurrences) as inputs and return the probability rounded to 5 decimal places.

### Mathematical Derivation & Intuition
## Understanding Poisson Distribution

The Poisson distribution is a discrete probability distribution that expresses the probability of a given number of events occurring in a fixed interval of time or space, provided these events occur with a known constant mean rate and independently of the time since the last event.

### Mathematical Definition

The probability of observing \( k \) events in a given interval is defined as:

$$
P(k; \lambda) = \frac{\lambda^k e^{-\lambda}}{k!}
$$

- **\( k \)**: Number of events (non-negative integer)
- **\( \lambda \)**: The mean number of events in the given interval (rate parameter)
- **\( e \)**: Euler's number, approximately 2.718

### Key Properties

- **Mean**: \( \lambda \)
- **Variance**: \( \lambda \)
- The Poisson distribution is used for modeling rare or random events.

### Example Calculation

Suppose the mean number of calls received in an hour (\( \lambda \)) is 5. Calculate the probability of receiving exactly 3 calls in an hour:

1. **Substitute into the formula**:
   $$
   P(3; 5) = \frac{5^3 e^{-5}}{3!}
   $$

2. **Calculate step-by-step**:
   $$
   P(3; 5) = \frac{125 \cdot e^{-5}}{6} \approx 0.14037
   $$

### Applications

The Poisson distribution is widely used in:

- Modeling the number of arrivals at a queue (e.g., calls at a call center)
- Counting occurrences over time (e.g., number of emails received per hour)
- Biology (e.g., distribution of mutations in a DNA strand)
- Traffic flow analysis (e.g., number of cars passing through an intersection)

This distribution is essential for understanding and predicting rare events in real-world scenarios.

### Verified Pure NumPy Solution (From Scratch)
```python
import math

def poisson_probability(k, lam):
    """
    Calculate the probability of observing exactly k events in a fixed interval,
    given the mean rate of events lam, using the Poisson distribution formula.
    :param k: Number of events (non-negative integer)
    :param lam: The average rate (mean) of occurrences in a fixed interval
    :return: Probability of k events occurring
    """
    # Calculate the Poisson probability using the formula
    probability = (lam ** k) * math.exp(-lam) / math.factorial(k)
    # Return the probability, rounded to five decimal places
    return round(probability, 5)
```

### Verified Sample Trace Data
```json
{
  "test": "print(poisson_probability(3, 5))",
  "expected_output": "0.14037"
}
```

---

## Problem 132: Simulate Markov Chain Transitions
- **Difficulty**: `medium`
- **Category**: Probability

### Description & Spec
Implement a function to simulate a Markov Chain. The function should take a 2D numpy array representing the transition matrix (where each row sums to 1), an integer for the initial state index, and an integer for the number of steps to simulate. It should return a numpy array containing the sequence of state indices over time, including the initial state. Use numpy for array operations and random selections.

### Mathematical Derivation & Intuition
## Markov Chains: A Stochastic Process

Markov Chains are a fundamental concept in probability theory, used to model systems that transition between different states over time. In this explanation, we will explore the key ideas behind Markov Chains, focusing on their mathematical foundations and intuitive meaning.

### 1. Definition of a Markov Chain

A Markov Chain is a sequence of events or states where the probability of moving to the next state depends only on the current state, not on any previous states. This property is known as the "memoryless" or Markov property.

To illustrate, imagine a system with a set of possible states, such as weather conditions (e.g., sunny or rainy). At any given moment, the system occupies one state, and the likelihood of transitioning to another state is determined solely by the current one. Mathematically, if we denote the states as $S_1, S_2, \dots, S_n$, the process evolves according to probabilities that satisfy the Markov property.

For example, the equation for the probability of being in state $S_j$ at time $t+1$, given the current state $S_i$ at time $t$, can be expressed as:

$
P(S_{t+1} = S_j \mid S_t = S_i, S_{t-1}, \dots) = P(S_{t+1} = S_j \mid S_t = S_i)
$

Here, $P$ represents probability, $S_t$ is the state at time $t$, and the right side shows that only the current state $S_t$ matters. This equation highlights how the process simplifies decision-making by ignoring historical data, making it useful for modeling random phenomena like random walks or population dynamics.

### 2. Transition Probabilities and the Matrix

At the heart of a Markov Chain is the concept of transition probabilities, which quantify the likelihood of moving from one state to another. These probabilities are organized into a structure called a transition matrix.

A transition matrix is a square array where each entry represents the probability of transitioning from a specific row state to a specific column state. For a system with $n$ states, the matrix is an $n \times n$ grid, and each row sums to 1, ensuring that the probabilities for all possible outcomes from a given state add up to certainty.

The general form of a transition matrix $P$ is:

$
P = \begin{pmatrix}
p_{11} & p_{12} & \cdots & p_{1n} \\
p_{21} & p_{22} & \cdots & p_{2n} \\
\vdots & \vdots & \ddots & \vdots \\
p_{n1} & p_{n2} & \cdots & p_{nn}
\end{pmatrix}
$

In this matrix, $p_{ij}$ is the probability of transitioning from state $i$ to state $j$. For instance, $p_{11}$ might represent the probability of staying in state 1, while $p_{12}$ represents the probability of moving from state 1 to state 2. Each $p_{ij}$ is a value between 0 and 1, and the sum of each row equals 1, as expressed by:

$
\sum_{j=1}^{n} p_{ij} = 1 \quad \text{for each row } i
$

This equation ensures that the matrix reflects a complete set of possibilities for each starting state, providing a clear framework for predicting future behavior based on the current position.

### 3. Evolution of States Over Time

Once the transition matrix is defined, we can describe how the states of a Markov Chain evolve through successive steps. Starting from an initial state, the process generates a sequence of states by applying the transition probabilities repeatedly.

At each step, the next state is determined by the probabilities associated with the current state. Over multiple steps, this leads to a sequence that can be analyzed to understand long-term patterns, such as whether the system tends to settle into certain states or remains unpredictable.

Mathematically, if we begin in state $i$ at time 0, the probability of being in state $j$ after one step is given by the entry $p_{ij}$ in the matrix. After multiple steps, the overall probability distribution can be computed by multiplying the initial state probabilities by the transition matrix raised to the power of the number of steps. For a probability vector $\mathbf{v}_t$ representing the likelihood of being in each state at time $t$, the evolution is:

$
\mathbf{v}_{t+1} = \mathbf{v}_t \cdot P
$

Here, $\mathbf{v}_t$ is a row vector of probabilities summing to 1, and $P$ is the transition matrix. This operation shows how the distribution shifts over time, with each multiplication reflecting the application of transition rules. In the long run, many Markov Chains reach a steady-state distribution, where the probabilities no longer change, offering insights into stable behaviors of the system.

---

### Example Walkthrough

To make the concept more concrete, consider a simple two-state system modeling weather patterns: State 1 as "Sunny" and State 2 as "Rainy." Suppose the transition matrix is:

$
P = \begin{pmatrix}
0.7 & 0.3 \\
0.4 & 0.6
\end{pmatrix}
$

In this matrix, the entry 0.7 means there is a 70% chance of staying Sunny if it is currently Sunny, while 0.3 means a 30% chance of becoming Rainy. Similarly, 0.4 indicates a 40% chance of becoming Sunny if it is currently Rainy, and 0.6 means a 60% chance of staying Rainy.

Starting from State 1 (Sunny), after one step, there is a 70% chance of remaining Sunny and a 30% chance of moving to Rainy. If it becomes Rainy, the next step would follow the second row of the matrix. Over several steps, this process might fluctuate, but eventually, it could approach a balance where the probabilities stabilize, reflecting typical weather patterns in this model. This example demonstrates how transition probabilities guide the system's behavior in a predictable yet random manner.

### Verified Pure NumPy Solution (From Scratch)
```python
import numpy as np

def simulate_markov_chain(transition_matrix, initial_state, num_steps):
    """
    Simulates a Markov Chain given a transition matrix, initial state, and number of steps.

    Parameters:
    transition_matrix : 2D numpy.ndarray, transition probabilities where each row sums to 1.
    initial_state : int, starting state index.
    num_steps : int, number of steps to simulate.

    Returns:
    numpy.ndarray, array of state indices over time, including the initial state.
    """
    states = np.zeros(num_steps + 1, dtype=int)
    states[0] = initial_state
    current_state = initial_state
    for t in range(num_steps):
        probabilities = transition_matrix[current_state]
        next_state = np.random.choice(transition_matrix.shape[1], p=probabilities)
        states[t + 1] = next_state
        current_state = next_state
    return states
```

### Verified Sample Trace Data
```json
{
  "test": "import numpy as np; np.random.seed(42); transition_matrix = np.array([[0.8, 0.2], [0.3, 0.7]]); print(simulate_markov_chain(transition_matrix, 0, 3))",
  "expected_output": "[0 0 1 1]"
}
```

---

## Problem 136: Calculate KL Divergence Between Two Multivariate Gaussian Distributions
- **Difficulty**: `medium`
- **Category**: Probability

### Description & Spec
KL divergence measures the dissimilarity between two probability distributions. In this problem, you'll implement a function to compute the KL divergence between two multivariate Gaussian distributions given their means and covariance matrices. Use the provided mathematical formulas and numerical considerations to ensure accuracy.

### Mathematical Derivation & Intuition
## KL divergence and its properties
KL divergence is used as a measure of dissimilarity between two distributions. It is defined by the following formula:
$$
D_{KL}(P || Q) = \mathbb{E}_{x\sim P(X)}log\frac{P(X)}{Q(X)},
$$
where $P(X)$ observed distribution we compare everything else with and $Q(X)$ is usually the varying one; $P(X)$ and $Q(X)$ are PMF (but could also be denoted as PDFs $f(x)$ and $q(x)$ in continuos case). The function has following properties:
* $D_{KL}\geq0$
* assymetry: $D_{KL}(P || Q) \neq D_{KL}(Q || P)$

## Finding $D_{KL}$ between two multivariate Gaussians
Consider two multivariate Normal distributions:
$$
p(x)\sim \mathbb{N}(\mu_1,\Sigma_1), \\
q(x)\sim \mathbb{N}(\mu_2,\Sigma_2)
$$

PDF of a multivariate Normal distribution is defined as:
$$
f(x)=\frac{1}{(2\pi)^\frac{p}{2}|\Sigma|^\frac{1}{2}}exp(-\frac{1}{2}(x-\mu)^T\Sigma^{-1}(x-\mu)),
$$

where $\Sigma$ - covariance matrix, $|\cdot|$ - determinant, $p$ - size of the random vector, i.e. number of different normally distributed features inside $P$ and $Q$ and $x$ usually denotes $x^T$, which is a random vector of size $p\times1$.

Now we can move onto calculating KL divergence for these two distributions, skipping the division part of two PDFs:
$$
\frac{1}{2}[\mathbb{E_p}log\frac{|\Sigma_q|}{|\Sigma_p|} ^ \textbf{[1]} - \mathbb{E_p}(x-\mu_p)^T\Sigma_p^{-1}(x-\mu_p) ^ \textbf{[2]} + \\
+ \mathbb{E_p}(x-\mu_q)^T\Sigma_q^{-1}(x-\mu_q) ^ \textbf{[3]}]= \\
= \frac{1}{2}[log\frac{|\Sigma_q|}{|\Sigma_p|}-p+(\mu_p-\mu_q)^T\Sigma^{-1}_q(\mu_p-\mu_q) + \\
+ tr(\Sigma^{-1}_q\Sigma_p)],
$$
where in order to achieve an equality we proceed to do $\textbf{[1]}:$
$$
log\frac{|\Sigma_q|}{|\Sigma_p|}=const\implies \text{EV equals to the value itself;}
$$

then $\textbf{[2]}:$
$$
\underset{N \times p}{(x-\mu_p)^T} * \sum_{p \times p} * \underset{N \times p}{(x-\mu_p)^T} = \underset{N\times N}{A}\text {, where } N=1 \implies \\
\implies A=\operatorname{tr}(A)
$$

Recall that:
$$
\operatorname{tr}(A B C)=\operatorname{tr}(B C A)=\operatorname{tr}(C B A)
$$

Then:
$$
\operatorname{tr}(A)=\operatorname{tr}\left(\left(x-\mu_p\right)^{\top}\left(x-\mu_p\right) \Sigma_p^{-1}\right)\\ =\operatorname{tr}\left(\Sigma_p \Sigma_p^{-1}\right)=\operatorname{tr}(I)=p  
$$

and finally $\textbf{[3]}$, where we should recall, that for multivariate Normal distributions this is true ($x\sim\mathbb{N}(\mu_2, \Sigma_2)$):
$$
\mathbb{E}(x-\mu_1)^TA(x-\mu_1)= \\
= (\mu_2-\mu_1)^TA(\mu_2-\mu_1)+tr(A\Sigma_2)
$$

### Verified Pure NumPy Solution (From Scratch)
```python
import numpy as np

def multivariate_kl_divergence(mu_p:np.ndarray, Cov_p:np.ndarray, 
                               mu_q:np.ndarray, Cov_q:np.ndarray) -> float:

    def trace(x: np.ndarray) -> float:
        return np.diag(x).sum()

    p = Cov_p.shape[0]
    return float(1/2 * (
        np.log(np.linalg.det(Cov_q)/np.linalg.det(Cov_p))         - p + (mu_p-mu_q).T @ np.linalg.inv(Cov_q) @ (mu_p-mu_q)         + trace(np.linalg.inv(Cov_q) @ Cov_p)
    ))
```

### Verified Sample Trace Data
```json
{
  "test": "import numpy as np\nnp.random.seed(42)\nPx = np.random.randn(4, 10)\nQx = np.random.randn(4, 10)\nmu1, cov1, mu2, cov2 = np.mean(Px, axis=1), np.cov(Px), np.mean(Qx, axis=1), np.cov(Qx)\nprint(round(multivariate_kl_divergence(mu1, cov1, mu2, cov2),4))",
  "expected_output": "2.193"
}
```

---

