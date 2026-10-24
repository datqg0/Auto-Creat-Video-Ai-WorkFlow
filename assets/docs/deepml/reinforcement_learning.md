# Deep-ML Knowledge Base: Reinforcement Learning

Total verified problems: 8

Source: [Deep-ML (Open-Deep-ML)](https://www.deep-ml.com/)

---

## Problem 101: Implement the GRPO Objective Function
- **Difficulty**: `hard`
- **Category**: Reinforcement Learning

### Description & Spec
Implement the GRPO (Generalized Relative Policy Optimization) objective function used to optimize policy parameters in reinforcement learning. Your task is to compute the GRPO objective given the likelihood ratios, advantage estimates, old policy probabilities, reference policy probabilities, and apply the clipping mechanism and KL divergence penalty correctly to maintain training stability.

### Mathematical Derivation & Intuition
### Understanding GRPO (Generalized Relative Policy Optimization)

GRPO is an advanced policy optimization algorithm in reinforcement learning that updates policy parameters while ensuring training stability. It builds upon Proximal Policy Optimization (PPO) by incorporating a KL divergence penalty to keep the new policy close to a reference policy.

### Mathematical Definition

The GRPO objective function is defined as:

$$
J_{GRPO}(\theta) = \mathbb{E}_{q \sim P(Q), \{o_i\}_{i=1}^G \sim \pi_{\theta_{old}}(O|q)} \left[ \frac{1}{G} \sum_{i=1}^G \min\left( \rho_i A_i, \text{clip}(\rho_i, 1-\epsilon, 1+\epsilon) A_i \right) - \beta D_{KL}(\pi_{\theta} \| \pi_{ref}) \right]
$$

Where:

- $\rho_i = \frac{\pi_{\theta}(o_i | q)}{\pi_{\theta_{old}}(o_i | q)}$ is the likelihood ratio.
- $A_i$ is the advantage estimate for the $i$-th action.
- $\epsilon$ is the clipping parameter.
- $\beta$ controls the influence of the KL divergence penalty.
- $D_{KL}$ is the Kullback-Leibler divergence between the new policy $\pi_{\theta}$ and the reference policy $\pi_{ref}$.

### Key Components

#### Likelihood Ratio $\rho_i$
- Measures how much more likely the new policy $\pi_{\theta}$ is to produce an output $o_i$ compared to the old policy $\pi_{\theta_{old}}$.
- $$\rho_i = \frac{\pi_{\theta}(o_i | q)}{\pi_{\theta_{old}}(o_i | q)}$$

#### Advantage Function $A_i$
- Evaluates the benefit of taking action $o_i$ compared to the average action.
- $$A_i = \frac{r_i - \text{mean}(r_1, \ldots, r_G)}{\text{std}(r_1, \ldots, r_G)}$$
- Where $r_i$ is the reward for the $i$-th action.

#### Clipping Mechanism
- Restricts the likelihood ratio to the range $[1 - \epsilon, 1 + \epsilon]$ to prevent large updates.
- $$\text{clip}(\rho_i, 1 - \epsilon, 1 + \epsilon)$$

#### KL Divergence Penalty
- Ensures the new policy $\pi_{\theta}$ does not deviate significantly from the reference policy $\pi_{ref}$.
- $$-\beta D_{KL}(\pi_{\theta} \| \pi_{ref})$$

### Benefits of GRPO

#### Stability
- The clipping mechanism prevents drastic policy updates, ensuring stable training.

#### Controlled Exploration
- The KL divergence penalty maintains a balance between exploring new policies and sticking close to a reliable reference policy.

#### Improved Performance
- By carefully managing policy updates, GRPO can lead to more effective learning and better policy performance.

### Use Cases

#### Reinforcement Learning Tasks
- Suitable for environments requiring stable and efficient policy updates.
- also a key component used for the DeepSeek-R1 model

#### Complex Decision-Making Problems
- Effective in scenarios with high-dimensional action spaces where maintaining policy stability is crucial.

### Conclusion

GRPO enhances policy optimization in reinforcement learning by combining the benefits of PPO with an additional KL divergence penalty. This ensures that policy updates are both effective and stable, leading to more reliable and performant learning agents.

### Verified Pure NumPy Solution (From Scratch)
```python
import numpy as np

def grpo_objective(rhos, A, pi_theta_old, pi_theta_ref, epsilon=0.2, beta=0.01) -> float:
    """
    Compute the GRPO objective function.

    Args:
        rhos: List of likelihood ratios (ρ_i) = π_theta(o_i | q) / π_theta_old(o_i | q).
        A: List of advantage estimates (A_i).
        pi_theta_old: List representing the old policy probabilities π_theta_old(o_i | q).
        pi_theta_ref: List representing the reference policy probabilities π_ref(o_i | q).
        epsilon: Clipping parameter (ϵ).
        beta: KL divergence penalty coefficient (β).

    Returns:
        The computed GRPO objective value.
    """
    G = len(rhos)
    if not (len(A) == len(pi_theta_old) == len(pi_theta_ref) == G):
        raise ValueError("All input lists must have the same length.")
    
    # Compute clipped likelihood ratios
    clipped_rhos = np.clip(rhos, 1 - epsilon, 1 + epsilon)
    
    # Compute the minimum terms for the objective
    unclipped = np.array(rhos) * np.array(A)
    clipped = clipped_rhos * np.array(A)
    min_terms = np.minimum(unclipped, clipped)
    average_min = np.mean(min_terms)
    
    # Compute pi_theta from rhos and pi_theta_old
    pi_theta = np.array(rhos) * np.array(pi_theta_old)
    
    # Normalize pi_theta and pi_theta_ref to ensure they are valid probability distributions
    pi_theta /= np.sum(pi_theta)
    pi_theta_ref /= np.sum(pi_theta_ref)
    
    # Compute KL divergence D_KL(pi_theta || pi_theta_ref)
    kl_divergence = np.sum(pi_theta * np.log(pi_theta / pi_theta_ref + 1e-10))  # Added epsilon to avoid log(0)
    
    # Compute the final objective
    objective = average_min - beta * kl_divergence
    
    return objective
```

### Verified Sample Trace Data
```json
{
  "test": "print(round(grpo_objective([1.2, 0.8, 1.1], [1.0, 1.0, 1.0], [0.9, 1.1, 1.0], [1.0, 0.5, 1.5], epsilon=0.2, beta=0.01),6))",
  "expected_output": "1.032749"
}
```

---

## Problem 122: Policy Gradient with REINFORCE
- **Difficulty**: `hard`
- **Category**: Reinforcement Learning

### Description & Spec
Implement the policy gradient estimator using the REINFORCE algorithm. The policy is parameterized by a 2D NumPy array `theta` of shape `(num_states, num_actions)`. The policy for each state is computed via softmax over `theta[s, :]`. Given a list of episodes (each a list of (state, action, reward) tuples), compute the average gradient of the log-policy multiplied by the return at each time step.

### Mathematical Derivation & Intuition
## REINFORCE and Policy Gradient Estimation

The REINFORCE algorithm computes gradients of the expected return with respect to policy parameters using Monte Carlo samples of episodes.

### Softmax Policy
Given $\theta$ with shape (num_states, num_actions), we define the probability of action $a$ in state $s$ as:

$$
\pi(a \mid s; \theta) = \frac{\exp(\theta[s, a])}{\sum_{a'} \exp(\theta[s, a'])}
$$

### REINFORCE Gradient
For an episode with transitions $(s_t, a_t, r_t)$ and returns $G_t = \sum_{k=t}^T r_k$:

$$
\nabla_\theta J(\theta) \approx \sum_t \nabla_\theta \log \pi(a_t \mid s_t) \cdot G_t
$$

### Log-Policy Gradient
To compute $\nabla_\theta \log \pi(a_t \mid s_t)$:

- For $\theta[s_t, a_t]$: $1 - \pi(a_t \mid s_t)$
- For $\theta[s_t, a']$, where $a' \neq a_t$: $-\pi(a' \mid s_t)$
- All other entries: 0

### Final Estimate
For multiple episodes:

$$
\hat{\nabla}_\theta J(\theta) = \frac{1}{N} \sum_{i=1}^N \sum_{t=0}^{T_i} \nabla_\theta \log \pi(a_t^i \mid s_t^i) \cdot G_t^i
$$

This algorithm works even without value function estimation, making it a foundational method in policy-based reinforcement learning.

### Verified Pure NumPy Solution (From Scratch)
```python
import numpy as np

def compute_policy_gradient(theta, episodes):
    def softmax(x):
        x = x - np.max(x)
        exps = np.exp(x)
        return exps / np.sum(exps)

    grad = np.zeros_like(theta)
    for episode in episodes:
        rewards = [step[2] for step in episode]
        returns = np.cumsum(rewards[::-1])[::-1]
        for t, (s, a, _), G in zip(range(len(episode)), episode, returns):
            probs = softmax(theta[s])
            grad_log_pi = np.zeros_like(theta)
            grad_log_pi[s, :] = -probs
            grad_log_pi[s, a] += 1
            grad += grad_log_pi * G
    return grad / len(episodes)
```

### Verified Sample Trace Data
```json
{
  "test": "theta = np.zeros((2,2)); episodes = [[(0,1,0), (1,0,1)], [(0,0,0)]]; print(np.round(compute_policy_gradient(theta, episodes), 4))",
  "expected_output": "[[-0.25  0.25]\n [ 0.25 -0.25]]"
}
```

---

## Problem 133: Implement Q-Learning Algorithm for MDPs
- **Difficulty**: `medium`
- **Category**: Reinforcement Learning

### Description & Spec
Write a function that implements the Q-Learning algorithm to learn the optimal Q-table for a given Markov Decision Process (MDP). The function should take the number of states, number of actions, transition probabilities matrix, rewards matrix, list of terminal states, learning rate, discount factor, epsilon for exploration, and the number of episodes as inputs. Use these parameters to iteratively update the Q-table based on the Q-Learning update rule, employing an epsilon-greedy strategy for action selection. Ensure the function handles starting from non-terminal states and stops episodes upon reaching a terminal state.

Constraints:
- num_states: Integer greater than or equal to 1.
- num_actions: Integer greater than or equal to 1.
- P: A 3D NumPy array of shape (num_states, num_actions, num_states) where each element is a probability between 0 and 1, and each sub-array sums to 1.
- R: A 2D NumPy array of shape (num_states, num_actions) with float or integer values.
- terminal_states: A list or NumPy array of integers, each between 0 and num_states - 1, with no duplicates.
- alpha: A float between 0 and 1.
- gamma: A float between 0 and 1.
- epsilon: A float between 0 and 1.
- num_episodes: An integer greater than or equal to 1.
The function should return a 2D NumPy array of shape (num_states, num_actions) representing the learned Q-table.

### Mathematical Derivation & Intuition
## Q-Learning: Learning Optimal Actions in Markov Decision Processes

Q-Learning is a method in reinforcement learning used to estimate the value of taking specific actions in different states within a Markov Decision Process (MDP). An MDP models decision-making scenarios where the outcomes of actions depend on the current state, and the goal is to maximize long-term rewards. This section breaks down the key concepts step by step, focusing on the underlying mathematics.

### 1. Understanding Markov Decision Processes
A Markov Decision Process is a framework for sequential decision-making. It consists of states, actions, transition probabilities, and rewards. In an MDP, the future state depends only on the current state and the chosen action, not on the history of previous states.

- States represent the situations an agent might encounter.
- Actions are the choices available in each state.
- Transition probabilities describe the likelihood of moving from one state to another after an action.
- Rewards are numerical values that quantify the immediate benefit of taking an action in a state.

For example, imagine navigating a simple grid where each cell is a state, moving right or left is an action, and reaching a goal gives a reward.

### 2. The Q-Value Function
At the heart of Q-Learning is the Q-value, which estimates the total expected reward of taking a specific action in a given state and then following the best possible strategy afterward.

Mathematically, the Q-value for a state $s$ and action $a$ is denoted as $Q(s, a)$. It is defined by the equation:

$$
Q(s, a) = r(s, a) + \gamma \sum_{s'} P(s' | s, a) \max_{a'} Q(s', a')
$$

Here:
- $r(s, a)$ is the immediate reward received for taking action $a$ in state $s$.
- $\gamma$ (gamma) is the discount factor, a number between 0 and 1 that reduces the importance of future rewards over time (e.g., if $\gamma = 0.9$, rewards in the near future are valued more than those far ahead).
- $P(s' | s, a)$ is the transition probability, representing the likelihood of ending up in state $s'$ after action $a$ in state $s$.
- $\max_{a'} Q(s', a')$ is the maximum Q-value of all possible actions in the next state $s'$, indicating the best future choice.

This equation captures the idea that the Q-value balances immediate rewards with the discounted value of future rewards, helping to identify the most valuable actions over time.

### 3. The Q-Learning Update Rule
Q-Learning updates the Q-value estimates iteratively based on experience, using a simple iterative formula. This process allows the agent to learn from trials without needing to know the full transition probabilities in advance.

The update rule is:

$$
Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r + \gamma \max_{a'} Q(s', a') - Q(s, a) \right]
$$

In this equation:
- $\alpha$ (alpha) is the learning rate, a value between 0 and 1 that controls how much new information overrides old estimates (e.g., if $\alpha = 0.1$, updates are gradual).
- $r$ is the reward observed after taking action $a$ in state $s$.
- $s'$ is the next state that results from the action.
- The term inside the brackets, $r + \gamma \max_{a'} Q(s', a') - Q(s, a)$, is the difference between the estimated Q-value and the actual experienced value, known as the temporal difference error.

This rule refines Q-values over multiple episodes, gradually converging to the optimal values that maximize long-term rewards.

### 4. Balancing Exploration and Exploitation
To learn effectively, Q-Learning must balance exploring new actions (to discover potential rewards) and exploiting known high-value actions.

This is achieved through an epsilon-greedy strategy, where:
- With probability $\epsilon$ (epsilon, a small number like 0.1), a random action is selected to encourage exploration.
- With probability $1 - \epsilon$, the action with the highest Q-value is chosen to exploit current knowledge.

For instance, if $\epsilon = 0.2$, in 20% of decisions, the agent tries something random, while in 80%, it picks the best-known option. Over time, $\epsilon$ can be reduced to favor exploitation as learning progresses.

---

### Example Walkthrough
Consider a simple two-state MDP: State A and State B, with two actions in each (Action 1 and Action 2). Suppose:
- From State A, Action 1 leads to State B with probability 1 and a reward of 1.
- From State B, any action ends the process with a reward of 0 (State B is terminal).
- Let $\gamma = 0.9$ and $\alpha = 0.5$.

Initially, assume all Q-values are 0. In the first episode:
- Start in State A and choose Action 1 (greedily, since all Q-values are equal).
- Move to State B, receive reward 1, and since State B is terminal, the update is:  
  $$
  Q(\text{A}, \text{Action 1}) \leftarrow 0 + 0.5 \left[ 1 + 0.9 \cdot 0 - 0 \right] = 0.5
  $$
- Now, Q(A, Action 1) is 0.5, so in future episodes, Action 1 is more likely in State A.

Through repeated episodes, Q-values adjust to reflect the best long-term rewards, such as prioritizing paths that lead to higher cumulative rewards.

### Verified Pure NumPy Solution (From Scratch)
```python
import numpy as np

def q_learning(num_states, num_actions, P, R, terminal_states, alpha, gamma, epsilon, num_episodes):
    """
    Implements Q-Learning algorithm to learn the optimal Q-table for a given MDP.

    Parameters:
    - num_states: int, number of states in the MDP
    - num_actions: int, number of actions in the MDP
    - P: NumPy array of shape (num_states, num_actions, num_states), transition probabilities
    - R: NumPy array of shape (num_states, num_actions), rewards for each state-action pair
    - terminal_states: list or NumPy array of integers, indices of terminal states
    - alpha: float, learning rate
    - gamma: float, discount factor
    - epsilon: float, probability of choosing a random action in epsilon-greedy policy
    - num_episodes: int, number of episodes to train

    Returns:
    - Q: NumPy array of shape (num_states, num_actions), the learned Q-table
    """
    Q = np.zeros((num_states, num_actions))
    
    for episode in range(num_episodes):
        # Start from a random non-terminal state
        state = np.random.choice([s for s in range(num_states) if s not in set(terminal_states)])
        
        while state not in terminal_states:
            # Epsilon-greedy action selection
            if np.random.rand() < epsilon:
                action = np.random.randint(num_actions)
            else:
                action = np.argmax(Q[state])
            
            # Sample next state based on transition probabilities
            next_state = np.random.choice(num_states, p=P[state, action])
            
            # Get reward
            reward = R[state, action]
            
            # Compute target Q-value
            if next_state in terminal_states:
                target = reward
            else:
                target = reward + gamma * np.max(Q[next_state])
            
            # Update Q-table
            Q[state, action] += alpha * (target - Q[state, action])
            
            # Transition to next state
            state = next_state
    
    return Q
```

### Verified Sample Trace Data
```json
{
  "test": "import numpy as np; np.random.seed(42); P = np.array([[[0, 1], [1, 0]], [[1, 0], [1, 0]]]); R = np.array([[1, 0], [0, 0]]); terminal_states = [1]; print(q_learning(2, 2, P, R, terminal_states, 0.1, 0.9, 0.1, 10))",
  "expected_output": "[[0.65132156, 0.052902  ],[0., 0.        ]]"
}
```

---

## Problem 142: Gridworld Policy Evaluation
- **Difficulty**: `medium`
- **Category**: Reinforcement Learning

### Description & Spec
Implement policy evaluation for a 5x5 gridworld. Given a policy (mapping each state to action probabilities), compute the state-value function $V(s)$ for each cell using the Bellman expectation equation. The agent can move up, down, left, or right, receiving a constant reward of -1 for each move. Terminal states (the four corners) are fixed at 0. Iterate until the largest change in $V$ is less than a given threshold. Only use Python built-ins and no external RL libraries.

### Mathematical Derivation & Intuition
# Gridworld Policy Evaluation

In reinforcement learning, **policy evaluation** is the process of computing the state-value function for a given policy. For a gridworld environment, this involves iteratively updating the value of each state based on the expected return following the policy.

## Key Concepts

- **State-Value Function (V):**  
  The expected return when starting from a state and following a given policy.

- **Policy:**  
  A mapping from states to probabilities of selecting each available action.

- **Bellman Expectation Equation:**  
  For each state $s$:
  $$
  V(s) = \sum_{a} \pi(a|s) \sum_{s'} P(s'|s,a) [R(s,a,s') + \gamma V(s')]
  $$
  where:
  - $ \pi(a|s) $ is the probability of taking action $ a $ in state $ s $,
  - $ P(s'|s,a) $ is the probability of transitioning to state $ s' $,
  - $ R(s,a,s') $ is the reward for that transition,
  - $ \gamma $ is the discount factor.

## Algorithm Overview

1. **Initialization:**  
   Start with an initial guess (commonly zeros) for the state-value function $ V(s) $.

2. **Iterative Update:**  
   For each non-terminal state, update the state value using the Bellman expectation equation. Continue updating until the maximum change in value (delta) is less than a given threshold.

3. **Terminal States:**  
   For this example, the four corners of the grid are considered terminal, so their values remain unchanged.

This evaluation method is essential for understanding how "good" each state is under a specific policy, and it forms the basis for more advanced reinforcement learning algorithms.

### Verified Pure NumPy Solution (From Scratch)
```python
def gridworld_policy_evaluation(policy: dict, gamma: float, threshold: float) -> list[list[float]]:
    grid_size = 5
    V = [[0.0 for _ in range(grid_size)] for _ in range(grid_size)]
    actions = {'up': (-1, 0), 'down': (1, 0), 'left': (0, -1), 'right': (0, 1)}
    reward = -1
    while True:
        delta = 0.0
        new_V = [row[:] for row in V]
        for i in range(grid_size):
            for j in range(grid_size):
                if (i, j) in [(0, 0), (0, grid_size-1), (grid_size-1, 0), (grid_size-1, grid_size-1)]:
                    continue
                v = 0.0
                for action, prob in policy[(i, j)].items():
                    di, dj = actions[action]
                    ni = i + di if 0 <= i + di < grid_size else i
                    nj = j + dj if 0 <= j + dj < grid_size else j
                    v += prob * (reward + gamma * V[ni][nj])
                new_V[i][j] = v
                delta = max(delta, abs(V[i][j] - new_V[i][j]))
        V = new_V
        if delta < threshold:
            break
    return V
```

### Verified Sample Trace Data
```json
{
  "test": "grid_size = 5\ngamma = 0.9\nthreshold = 0.001\npolicy = {(i, j): {'up': 0.25, 'down': 0.25, 'left': 0.25, 'right': 0.25} for i in range(grid_size) for j in range(grid_size)}\nV = gridworld_policy_evaluation(policy, gamma, threshold)\nprint([round(V[2][2], 4), V[0][0], V[0][4], V[4][0], V[4][4]])",
  "expected_output": "[-7.0902, 0.0, 0.0, 0.0, 0.0]"
}
```

---

## Problem 157: Implement the Bellman Equation for Value Iteration
- **Difficulty**: `medium`
- **Category**: Reinforcement Learning

### Description & Spec
Write a function that performs one step of value iteration for a given Markov Decision Process (MDP) using the Bellman equation. The function should update the state-value function V(s) for each state based on possible actions, transition probabilities, rewards, and the discount factor gamma. Only use NumPy.

### Mathematical Derivation & Intuition
# **The Bellman Equation**

The **Bellman equation** is a fundamental recursive equation in reinforcement learning that relates the value of a state to the values of possible next states. It provides the mathematical foundation for key RL algorithms such as value iteration and Q-learning.

---

## **Key Idea**
For each state $s$, the value $V(s)$ is the maximum expected return obtainable by choosing the best action $a$ and then following the optimal policy:

$$
V(s) = \max_{a} \sum_{s'} P(s'|s, a) \left[ R(s, a, s') + \gamma V(s') \right]
$$

Where:
- $V(s)$: value of state $s$
- $a$: possible actions
- $P(s'|s, a)$: probability of moving to state $s'$ from $s$ via $a$
- $R(s, a, s')$: reward for this transition
- $\gamma$: discount factor ($0 \leq \gamma \leq 1$)
- $V(s')$: value of next state

---

## **How to Use**
1. **For each state:**
   - For each possible action, sum over possible next states, weighting by transition probability.
   - Add the immediate reward and the discounted value of the next state.
   - Choose the action with the highest expected value (for control).
2. **Repeat until values converge** (value iteration) or as part of other RL updates.

---

## **Applications**
- **Value Iteration** and **Policy Iteration** in Markov Decision Processes (MDP)
- **Q-learning** and other RL algorithms
- Calculating the optimal value function and policy in gridworlds, games, and general MDPs

---

## **Why It Matters**
- The Bellman equation formalizes the notion of **optimality** in sequential decision-making.
- It is a backbone for teaching agents to solve environments with rewards, uncertainty, and long-term planning.

### Verified Pure NumPy Solution (From Scratch)
```python
import numpy as np

def bellman_update(V, transitions, gamma):
    n_states = len(V)
    new_V = np.zeros_like(V)
    for s in range(n_states):
        action_values = []
        for a in transitions[s]:
            total = 0
            for prob, next_s, reward, done in transitions[s][a]:
                total += prob * (reward + gamma * (0 if done else V[next_s]))
            action_values.append(total)
        new_V[s] = max(action_values)
    return new_V
```

### Verified Sample Trace Data
```json
{
  "test": "import numpy as np\ntransitions = [\n  # For state 0\n  {0: [(1.0, 0, 0.0, False)], 1: [(1.0, 1, 1.0, False)]},\n  # For state 1\n  {0: [(1.0, 0, 0.0, False)], 1: [(1.0, 1, 1.0, True)]}\n]\nV = np.array([0.0, 0.0])\ngamma = 0.9\nnew_V = bellman_update(V, transitions, gamma)\nprint(np.round(new_V, 2))",
  "expected_output": "[1., 1.]"
}
```

---

## Problem 158: Epsilon-Greedy Action Selection for n-Armed Bandit
- **Difficulty**: `easy`
- **Category**: Reinforcement Learning

### Description & Spec
Implement the epsilon-greedy method for action selection in an n-armed bandit problem. Given a set of estimated action values (Q-values), select an action using the epsilon-greedy policy: with probability epsilon, choose a random action; with probability 1 - epsilon, choose the action with the highest estimated value.

### Mathematical Derivation & Intuition
### Epsilon-Greedy Policy

The epsilon-greedy method is a fundamental action selection strategy used in reinforcement learning, especially for solving the n-armed bandit problem. The key idea is to balance **exploration** (trying new actions) and **exploitation** (choosing the best-known action):

- With probability $\varepsilon$ (epsilon), the agent explores by selecting an action at random.
- With probability $1-\varepsilon$, it exploits by choosing the action with the highest estimated value (greedy choice).

The epsilon-greedy policy is simple to implement and provides a way to avoid getting stuck with suboptimal actions due to insufficient exploration.

### Verified Pure NumPy Solution (From Scratch)
```python
import numpy as np

def epsilon_greedy(Q, epsilon=0.1):
    if np.random.rand() < epsilon:
        return np.random.randint(len(Q))
    else:
        return int(np.argmax(Q))
```

### Verified Sample Trace Data
```json
{
  "test": "import numpy as np\nnp.random.seed(0)\nprint([epsilon_greedy(np.array([1, 2, 3]), epsilon=0.0) for _ in range(5)])",
  "expected_output": "[2, 2, 2, 2, 2]"
}
```

---

## Problem 158: Incremental Mean for Online Reward Estimation
- **Difficulty**: `easy`
- **Category**: Reinforcement Learning

### Description & Spec
Implement an efficient method to update the mean reward for a k-armed bandit action after receiving each new reward, **without storing the full history of rewards**. Given the previous mean estimate (Q_prev), the number of times the action has been selected (k), and a new reward (R), compute the updated mean using the incremental formula.

**Note:** Using a regular mean that stores all past rewards will eventually run out of memory. Your solution should use only the previous mean, the count, and the new reward.

### Mathematical Derivation & Intuition
### Incremental Mean Update Rule

The incremental mean formula lets you update your estimate of the mean after each new observation, **without keeping all previous rewards in memory**. For the k-th reward $R_k$ and previous estimate $Q_{k}$:

$$
Q_{k+1} = Q_k + \frac{1}{k} (R_k - Q_k)
$$

This saves memory compared to the regular mean, which requires storing all past rewards and recalculating each time. The incremental rule is crucial for online learning and large-scale problems where storing all data is impractical.

### Verified Pure NumPy Solution (From Scratch)
```python
def incremental_mean(Q_prev, k, R):
    return Q_prev + (1 / k) * (R - Q_prev)
```

### Verified Sample Trace Data
```json
{
  "test": "Q = 0.0\nk = 1\nR = 5.0\nprint(round(incremental_mean(Q, k, R), 4))",
  "expected_output": "5.0"
}
```

---

## Problem 161: Exponential Weighted Average of Rewards
- **Difficulty**: `medium`
- **Category**: Reinforcement Learning

### Description & Spec
Given an initial value $Q_1$, a list of $k$ observed rewards $R_1, R_2, \ldots, R_k$, and a step size $\alpha$, implement a function to compute the exponentially weighted average as:

$$(1-\alpha)^k Q_1 + \sum_{i=1}^k \alpha (1-\alpha)^{k-i} R_i$$

This weighting gives more importance to recent rewards, while the influence of the initial estimate $Q_1$ decays over time. Do **not** use running/incremental updates; instead, compute directly from the formula. (This is called the *exponential recency-weighted average*.)

### Mathematical Derivation & Intuition
### Exponential Recency-Weighted Average

When the environment is nonstationary, it is better to give more weight to recent rewards. The formula $$(1-\alpha)^k Q_1 + \sum_{i=1}^k \alpha (1-\alpha)^{k-i} R_i$$ computes the expected value by exponentially decaying the influence of old rewards and the initial estimate. The parameter $\alpha$ controls how quickly old information is forgotten: higher $\alpha$ gives more weight to new rewards.

### Verified Pure NumPy Solution (From Scratch)
```python
def exp_weighted_average(Q1, rewards, alpha):
    k = len(rewards)
    value = (1 - alpha) ** k * Q1
    for i, Ri in enumerate(rewards):
        value += alpha * (1 - alpha) ** (k - i - 1) * Ri
    return value
```

### Verified Sample Trace Data
```json
{
  "test": "Q1 = 10.0\nrewards = [4.0, 7.0, 13.0]\nalpha = 0.5\nprint(round(exp_weighted_average(Q1, rewards, alpha), 4))",
  "expected_output": "10.0"
}
```

---

