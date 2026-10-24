# Deep-ML Knowledge Base: Deep Learning

Total verified problems: 48

Source: [Deep-ML (Open-Deep-ML)](https://www.deep-ml.com/)

---

## Problem 22: Sigmoid Activation Function Understanding
- **Difficulty**: `easy`
- **Category**: Deep Learning

### Description & Spec
Write a Python function that computes the output of the sigmoid activation function given an input value z. The function should return the output rounded to four decimal places.

### Mathematical Derivation & Intuition
## Understanding the Sigmoid Activation Function

The sigmoid activation function is crucial in neural networks, especially for binary classification tasks. It maps any real-valued number into the interval \( (0, 1) \), making it useful for modeling probability as an output.

### Mathematical Definition
The sigmoid function is mathematically defined as:
$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$
where \( z \) is the input to the function.

### Characteristics
- **Output Range**: The output is always between 0 and 1.
- **Shape**: The function has an "S" shaped curve.
- **Gradient**: The gradient is highest near \( z = 0 \) and decreases as \( z \) moves away from 0 in either direction.

The sigmoid function is particularly useful for turning logits (raw prediction values) into probabilities in binary classification models.

### Verified Pure NumPy Solution (From Scratch)
```python
import math
def sigmoid(z: float) -> float:
   result = 1 / (1 + math.exp(-z))
   return round(result, 4)
```

### Verified Sample Trace Data
```json
{
  "test": "print(sigmoid(0))",
  "expected_output": "0.5"
}
```

---

## Problem 23: Softmax Activation Function Implementation 
- **Difficulty**: `easy`
- **Category**: Deep Learning

### Description & Spec
Write a Python function that computes the softmax activation for a given list of scores. The function should return the softmax values as a list, each rounded to four decimal places.

### Mathematical Derivation & Intuition
## Understanding the Softmax Activation Function

The softmax function is a generalization of the sigmoid function and is used in the output layer of a neural network model that handles multi-class classification tasks.

### Mathematical Definition
The softmax function is mathematically represented as:
$$
\text{softmax}(z_i) = \frac{e^{z_i}}{\sum_{j} e^{z_j}}
$$

### Characteristics
- **Output Range**: Each output value is between 0 and 1, and the sum of all outputs is 1.
- **Probability Distribution**: It transforms scores into probabilities, making them easier to interpret and useful for classification tasks.

The softmax function is essential for models where the output needs to represent a probability distribution across multiple classes.

### Verified Pure NumPy Solution (From Scratch)
```python
import math
def softmax(scores: list[float]) -> list[float]:
    exp_scores = [math.exp(score) for score in scores]
    sum_exp_scores = sum(exp_scores)
    probabilities = [round(score / sum_exp_scores, 4) for score in exp_scores]
    return probabilities
```

### Verified Sample Trace Data
```json
{
  "test": "print(softmax([1, 2, 3]))",
  "expected_output": "[0.09, 0.2447, 0.6652]"
}
```

---

## Problem 24: Single Neuron
- **Difficulty**: `easy`
- **Category**: Deep Learning

### Description & Spec
Write a Python function that simulates a single neuron with a sigmoid activation function for binary classification, handling multidimensional input features. The function should take a list of feature vectors (each vector representing multiple features for an example), associated true binary labels, and the neuron's weights (one for each feature) and bias as input. It should return the predicted probabilities after sigmoid activation and the mean squared error between the predicted probabilities and the true labels, both rounded to four decimal places.

### Mathematical Derivation & Intuition
## Single Neuron Model with Multidimensional Input and Sigmoid Activation

This task involves a neuron model designed for binary classification with multidimensional input features, using the sigmoid activation function to output probabilities. It also involves calculating the mean squared error (MSE) to evaluate prediction accuracy.

### Mathematical Background

**Neuron Output Calculation:**
$$
z = \sum (weight_i \times feature_i) + bias
$$
$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

**MSE Calculation:**
$$
MSE = \frac{1}{n} \sum (predicted - true)^2
$$

### Explanation of Terms
- **\( z \)**: The sum of weighted inputs plus bias.
- **\( \sigma(z) \)**: The sigmoid activation output.
- **\( predicted \)**: The probabilities after sigmoid activation.
- **\( true \)**: The true binary labels.

### Practical Implementation
- Each feature vector is processed to calculate a combined weighted sum, which is then passed through the sigmoid function to determine the probability of the input belonging to the positive class.
- **MSE** provides a measure of error, offering insights into the model's performance and aiding in its optimization.

### Verified Pure NumPy Solution (From Scratch)
```python
import math
def single_neuron_model(features, labels, weights, bias):
    probabilities = []
    for feature_vector in features:
        z = sum(weight * feature for weight, feature in zip(weights, feature_vector)) + bias
        prob = 1 / (1 + math.exp(-z))
        probabilities.append(round(prob, 4))
    
    mse = sum((prob - label) ** 2 for prob, label in zip(probabilities, labels)) / len(labels)
    mse = round(mse, 4)
    
    return probabilities, mse
```

### Verified Sample Trace Data
```json
{
  "test": "print(single_neuron_model([[0.5, 1.0], [-1.5, -2.0], [2.0, 1.5]], [0, 1, 0], [0.7, -0.4], -0.1))",
  "expected_output": "([0.4626, 0.4134, 0.6682], 0.3349)"
}
```

---

## Problem 25: Single Neuron with Backpropagation
- **Difficulty**: `medium`
- **Category**: Deep Learning

### Description & Spec
Write a Python function that simulates a single neuron with sigmoid activation, and implements backpropagation to update the neuron's weights and bias. The function should take a list of feature vectors, associated true binary labels, initial weights, initial bias, a learning rate, and the number of epochs. The function should update the weights and bias using gradient descent based on the MSE loss, and return the updated weights, bias, and a list of MSE values for each epoch, each rounded to four decimal places.

### Mathematical Derivation & Intuition
## Neural Network Learning with Backpropagation

This task involves implementing backpropagation for a single neuron in a neural network. The neuron processes inputs and updates parameters to minimize the Mean Squared Error (MSE) between predicted outputs and true labels.

### Mathematical Background

**Forward Pass**  
Compute the neuron output by calculating the dot product of the weights and input features, and adding the bias:
$$
z = w_1x_1 + w_2x_2 + \dots + w_nx_n + b
$$
$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

**Loss Calculation (MSE)**  
The Mean Squared Error quantifies the error between the neuron's predictions and the actual labels:
$$
MSE = \frac{1}{n} \sum_{i=1}^{n} (\sigma(z_i) - y_i)^2
$$

### Backward Pass (Gradient Calculation)
Compute the gradient of the MSE with respect to each weight and the bias. This involves the partial derivatives of the loss function with respect to the output of the neuron, multiplied by the derivative of the sigmoid function:
$$
\frac{\partial MSE}{\partial w_j} = \frac{2}{n} \sum_{i=1}^{n} (\sigma(z_i) - y_i) \sigma'(z_i) x_{ij}
$$
$$
\frac{\partial MSE}{\partial b} = \frac{2}{n} \sum_{i=1}^{n} (\sigma(z_i) - y_i) \sigma'(z_i)
$$

### Parameter Update
Update each weight and the bias by subtracting a portion of the gradient, determined by the learning rate:
$$
w_j = w_j - \alpha \frac{\partial MSE}{\partial w_j}
$$
$$
b = b - \alpha \frac{\partial MSE}{\partial b}
$$

### Practical Implementation
This process refines the neuron's ability to predict accurately by iteratively adjusting the weights and bias based on the error gradients, optimizing the neural network's performance over multiple iterations.

### Verified Pure NumPy Solution (From Scratch)
```python
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def train_neuron(features, labels, initial_weights, initial_bias, learning_rate, epochs):
    weights = np.array(initial_weights)
    bias = initial_bias
    features = np.array(features)
    labels = np.array(labels)
    mse_values = []

    for _ in range(epochs):
        z = np.dot(features, weights) + bias
        predictions = sigmoid(z)
        
        mse = np.mean((predictions - labels) ** 2)
        mse_values.append(round(mse, 4))

        # Gradient calculation for weights and bias
        errors = predictions - labels
        weight_gradients = (2/len(labels)) * np.dot(features.T, errors * predictions * (1 - predictions))
        bias_gradient = (2/len(labels)) * np.sum(errors * predictions * (1 - predictions))
        
        # Update weights and bias
        weights -= learning_rate * weight_gradients
        bias -= learning_rate * bias_gradient

        # Round weights and bias for output
        updated_weights = np.round(weights, 4)
        updated_bias = round(bias, 4)

    return updated_weights.tolist(), updated_bias, mse_values
```

### Verified Sample Trace Data
```json
{
  "test": "print(train_neuron(np.array([[1.0, 2.0], [2.0, 1.0], [-1.0, -2.0]]), np.array([1, 0, 0]), np.array([0.1, -0.2]), 0.0, 0.1, 2))",
  "expected_output": "([0.1036, -0.1425], -0.0167, [0.3033, 0.2942])"
}
```

---

## Problem 26: Implementing Basic Autograd Operations
- **Difficulty**: `medium`
- **Category**: Deep Learning

### Description & Spec
Special thanks to Andrej Karpathy for making a video about this, if you haven't already check out his videos on YouTube https://youtu.be/VMj-3S1tku0?si=gjlnFP4o3JRN9dTg. Write a Python class similar to the provided 'Value' class that implements the basic autograd operations: addition, multiplication, and ReLU activation. The class should handle scalar values and should correctly compute gradients for these operations through automatic differentiation.

### Mathematical Derivation & Intuition
## Understanding Mathematical Concepts in Autograd Operations

*First, watch the video in the Solution section.*

This task focuses on implementing basic automatic differentiation mechanisms for neural networks. The operations of addition, multiplication, and ReLU are fundamental to neural network computations and their training through backpropagation.

### Mathematical Foundations

**Addition (`__add__`)**  
- **Forward Pass**: For two scalar values \( a \) and \( b \), their sum \( s \) is:
  $$
  s = a + b
  $$
- **Backward Pass**: The derivative of \( s \) with respect to both \( a \) and \( b \) is 1. During backpropagation, the gradient of the output is passed directly to both inputs.

**Multiplication (`__mul__`)**  
- **Forward Pass**: For two scalar values \( a \) and \( b \), their product \( p \) is:
  $$
  p = a \times b
  $$
- **Backward Pass**: The gradient of \( p \) with respect to \( a \) is \( b \), and with respect to \( b \) is \( a \). During backpropagation, each input's gradient is the product of the other input and the output's gradient.

**ReLU Activation (`relu`)**  
- **Forward Pass**: The ReLU function is defined as:
  $$
  R(x) = \max(0, x)
  $$
  This function outputs \( x \) if \( x \) is positive, and 0 otherwise.
- **Backward Pass**: The derivative of the ReLU function is 1 for \( x > 0 \) and 0 for \( x \leq 0 \). The gradient is propagated through the function only if the input is positive; otherwise, it stops.

### Conceptual Application in Neural Networks
- **Addition and Multiplication**: These operations are ubiquitous in neural networks, forming the basis for computing weighted sums of inputs in the neurons.
- **ReLU Activation**: Commonly used as an activation function in neural networks due to its simplicity and effectiveness in introducing non-linearity, making learning complex patterns possible.

Understanding these operations and their implications on gradient flow is crucial for designing and training effective neural network models. By implementing these from scratch, you gain deeper insights into the workings of more sophisticated deep learning libraries.

### Verified Pure NumPy Solution (From Scratch)
```python
class Value:
    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self.grad = 0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')
        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def relu(self):
        out = Value(0 if self.data < 0 else self.data, (self,), 'ReLU')
        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward
        return out

    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        self.grad = 1
        for v in reversed(topo):
            v._backward()
    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"
```

### Verified Sample Trace Data
```json
{
  "test": "a = Value(2);b = Value(3);c = Value(10);d = a + b * c  ;e = Value(7) * Value(2);f = e + d;g = f.relu()  \ng.backward()\nprint(a,b,c,d,e,f,g)\n",
  "expected_output": " Value(data=2, grad=1) Value(data=3, grad=10) Value(data=10, grad=3) Value(data=32, grad=1) Value(data=14, grad=1) Value(data=46, grad=1) Value(data=46, grad=1)"
}
```

---

## Problem 39: Implementation of Log Softmax Function
- **Difficulty**: `easy`
- **Category**: Deep Learning

### Description & Spec
In machine learning and statistics, the softmax function is a generalization of the logistic function that converts a vector of scores into probabilities. The log-softmax function is the logarithm of the softmax function, and it is often used for numerical stability when computing the softmax of large numbers.

Given a 1D numpy array of scores, implement a Python function to compute the log-softmax of the array.

### Mathematical Derivation & Intuition
## Understanding Log Softmax Function

The log softmax function is a numerically stable way of calculating the logarithm of the softmax function. The softmax function converts a vector of arbitrary values (logits) into a vector of probabilities, where each value lies between 0 and 1, and the values sum to 1.

### Softmax Function
The softmax function is given by:
$$
\text{softmax}(x_i) = \frac{e^{x_i}}{\sum_{j=1}^n e^{x_j}}
$$

### Log Softmax Function
Directly applying the logarithm to the softmax function can lead to numerical instability, especially when dealing with large numbers. To prevent this, we use the log-softmax function, which incorporates a shift by subtracting the maximum value from the input vector:
$$
\text{log softmax}(x_i) = x_i - \max(x) - \log\left(\sum_{j=1}^n e^{x_j - \max(x)}\right)
$$

This formulation helps to avoid overflow issues that can occur when exponentiating large numbers. The log-softmax function is particularly useful in machine learning for calculating probabilities in a stable manner, especially when used with cross-entropy loss functions.

### Verified Pure NumPy Solution (From Scratch)
```python
import numpy as np

def log_softmax(scores: list) -> np.ndarray:
    # Subtract the maximum value for numerical stability
    scores = scores - np.max(scores)
    return scores - np.log(np.sum(np.exp(scores)))
```

### Verified Sample Trace Data
```json
{
  "test": "print(log_softmax([1, 2, 3]))",
  "expected_output": "[-2.4076, -1.4076, -0.4076]"
}
```

---

## Problem 40: Implementing a Custom Dense Layer in Python
- **Difficulty**: `hard`
- **Category**: Deep Learning

### Description & Spec
## Implementing a Custom Dense Layer in Python

You are provided with a base `Layer` class that defines the structure of a neural network layer. Your task is to implement a subclass called `Dense`, which represents a fully connected neural network layer. The `Dense` class should extend the `Layer` class and implement the following methods:

1. **Initialization (`__init__`)**:
   - Define the layer with a specified number of neurons (`n_units`) and an optional input shape (`input_shape`).
   - Set up placeholders for the layer's weights (`W`), biases (`w0`), and optimizers.

2. **Weight Initialization (`initialize`)**:
   - Initialize the weights `W` using a uniform distribution with a limit of `1 / sqrt(input_shape[0])`, and bias `w0` should be set to zero.
   - Initialize optimizers for `W` and `w0`.

3. **Parameter Count (`parameters`)**:
   - Return the total number of trainable parameters in the layer, which includes the parameters in `W` and `w0`.

4. **Forward Pass (`forward_pass`)**:
   - Compute the output of the layer by performing a dot product between the input `X` and the weight matrix `W`, and then adding the bias `w0`.

5. **Backward Pass (`backward_pass`)**:
   - Calculate and return the gradient with respect to the input.
   - If the layer is trainable, update the weights and biases using the optimizer's update rule.

6. **Output Shape (`output_shape`)**:
   - Return the shape of the output produced by the forward pass, which should be `(self.n_units,)`.

**Objective**:  
Extend the `Layer` class by implementing the `Dense` class to ensure it functions correctly within a neural network framework.

### Mathematical Derivation & Intuition
## Understanding the Dense Layer

The Dense layer, also known as a fully connected layer, is a fundamental building block in neural networks. It connects each input neuron to each output neuron, hence the term "fully connected."

### 1. Weight Initialization
In the `initialize` method, weights are typically initialized using a uniform distribution within a certain range. For a Dense layer, a common practice is to set this range as:
$$
\text{limit} = \frac{1}{\sqrt{\text{input\_shape}}}
$$
This initialization helps in maintaining a balance in the distribution of weights, preventing issues like vanishing or exploding gradients during training.

### 2. Forward Pass
During the forward pass, the input data \( X \) is multiplied by the weight matrix \( W \) and added to the bias \( w_0 \) to produce the output:
$$
\text{output} = X \cdot W + w_0
$$

### 3. Backward Pass
The backward pass computes the gradients of the loss function with respect to the input data, weights, and biases. If the layer is trainable, it updates the weights and biases using the optimizer's update rule:
$$
W = W - \eta \cdot \text{grad}_W
$$
$$
w_0 = w_0 - \eta \cdot \text{grad}_{w_0}
$$
where \( \eta \) is the learning rate and \( \text{grad}_W \) and \( \text{grad}_{w_0} \) are the gradients of the weights and biases, respectively.

### 4. Output Shape
The shape of the output from a Dense layer is determined by the number of neurons in the layer. If a layer has `n_units` neurons, the output shape will be \( (n\_units,) \).

### Resources
- [CS231n: Fully Connected Layer](https://cs231n.github.io/neural-networks-2/#fc)

### Verified Pure NumPy Solution (From Scratch)
```python
class Dense(Layer):
    def __init__(self, n_units, input_shape=None):
        self.layer_input = None
        self.input_shape = input_shape
        self.n_units = n_units
        self.trainable = True
        self.W = None
        self.w0 = None

    def initialize(self, optimizer):
        limit = 1 / math.sqrt(self.input_shape[0])
        self.W  = np.random.uniform(-limit, limit, (self.input_shape[0], self.n_units))
        self.w0 = np.zeros((1, self.n_units))
        self.W_opt  = copy.copy(optimizer)
        self.w0_opt = copy.copy(optimizer)

    def parameters(self):
        return np.prod(self.W.shape) + np.prod(self.w0.shape)

    def forward_pass(self, X, training=True):
        self.layer_input = X
        return X.dot(self.W) + self.w0

    def backward_pass(self, accum_grad):
        W = self.W
        if self.trainable:
            grad_w = self.layer_input.T.dot(accum_grad)
            grad_w0 = np.sum(accum_grad, axis=0, keepdims=True)
            self.W = self.W_opt.update(self.W, grad_w)
            self.w0 = self.w0_opt.update(self.w0, grad_w0)
        accum_grad = accum_grad.dot(W.T)
        return accum_grad

    def output_shape(self):
        return (self.n_units, )
```

### Verified Sample Trace Data
```json
{
  "test": "\ndense_layer = Dense(n_units=3, input_shape=(2,))\n\nclass MockOptimizer:\n    def update(self, weights, grad):\n        return weights - 0.01 * grad\n\noptimizer = MockOptimizer()\ndense_layer.initialize(optimizer)\n\nX = np.array([[1, 2]])\noutput = dense_layer.forward_pass(X)\n\naccum_grad = np.array([[0.1, 0.2, 0.3]])\nback_output = dense_layer.backward_pass(accum_grad)\nprint(back_output)",
  "expected_output": "[[ 0.20816524, -0.22928937]]"
}
```

---

## Problem 41: Simple Convolutional 2D Layer
- **Difficulty**: `medium`
- **Category**: Deep Learning

### Description & Spec
In this problem, you need to implement a 2D convolutional layer in Python. This function will process an input matrix using a specified convolutional kernel, padding, and stride.

### Mathematical Derivation & Intuition
## Simple Convolutional 2D Layer

The Convolutional layer is a fundamental component used extensively in Computer Vision tasks. Here are the crucial parameters:

### Parameters
1. **input_matrix**:  
   A 2D NumPy array representing the input data, such as an image. Each element in this array corresponds to a pixel or a feature value in the input space. The dimensions of the input matrix are typically represented as $ \text{height} \times \text{width} $.

2. **kernel**:  
   Another 2D NumPy array representing the convolutional filter. The kernel is smaller than the input matrix and slides over it to perform the convolution operation. Each element in the kernel serves as a weight that modifies the input during convolution. The kernel size is denoted as $ \text{kernel\_height} \times \text{kernel\_width} $.

3. **padding**:  
   An integer specifying the number of rows and columns of zeros added around the input matrix. Padding controls the spatial dimensions of the output, allowing the kernel to process edge elements effectively or to maintain the original input size.

4. **stride**:  
   An integer that represents the number of steps the kernel moves across the input matrix for each convolution. A stride greater than one reduces the output size, as the kernel skips over elements.

### Implementation
1. **Padding the Input**:  
   The input matrix is padded with zeros based on the specified `padding` value. This increases the input size and enables the kernel to cover elements at the borders and corners.

2. **Calculating Output Dimensions**:  
   The height and width of the output matrix are calculated using the following formulas:
   $$
   \text{output\_height} = \left( \frac{\text{input\_height, padded} - \text{kernel\_height}}{\text{stride}} \right) + 1
   $$
   $$
   \text{output\_width} = \left( \frac{\text{input\_width, padded} - \text{kernel\_width}}{\text{stride}} \right) + 1
   $$

3. **Performing Convolution**:
   - A nested loop iterates over each position where the kernel can be applied to the padded input matrix.
   - At each position, a region of the input matrix, matching the size of the kernel, is selected.
   - Element-wise multiplication between the kernel and the input region is performed, followed by summing the results to produce a single value. This value is then stored in the corresponding position of the output matrix.

4. **Output**:  
   The function returns the output matrix, which contains the results of the convolution operation performed across the entire input.

### Verified Pure NumPy Solution (From Scratch)
```python
import numpy as np

def simple_conv2d(input_matrix: np.ndarray, kernel: np.ndarray, padding: int, stride: int):
    input_height, input_width = input_matrix.shape
    kernel_height, kernel_width = kernel.shape

    padded_input = np.pad(input_matrix, ((padding, padding), (padding, padding)), mode='constant')
    input_height_padded, input_width_padded = padded_input.shape

    output_height = (input_height_padded - kernel_height) // stride + 1
    output_width = (input_width_padded - kernel_width) // stride + 1

    output_matrix = np.zeros((output_height, output_width))

    for i in range(output_height):
        for j in range(output_width):
            region = padded_input[i*stride:i*stride + kernel_height, j*stride:j*stride + kernel_width]
            output_matrix[i, j] = np.sum(region * kernel)

    return output_matrix
```

### Verified Sample Trace Data
```json
{
  "test": "input_matrix = np.array([\n    [1., 2., 3., 4., 5.],\n    [6., 7., 8., 9., 10.],\n    [11., 12., 13., 14., 15.],\n    [16., 17., 18., 19., 20.],\n    [21., 22., 23., 24., 25.],\n])\nkernel = np.array([\n    [1., 2.],\n    [3., -1.],\n])\npadding, stride = 0, 1\nexpected = np.array([\n    [ 16., 21., 26., 31.],\n    [ 41., 46., 51., 56.],\n    [ 66., 71., 76., 81.],\n    [ 91., 96., 101., 106.],\n])\noutput = simple_conv2d(input_matrix, kernel, padding, stride)\nprint(output)",
  "expected_output": "[[ 16.,  21.,  26.,  31.],\n [ 41.,  46.,  51.,  56.],\n [ 66.,  71.,  76.,  81.],\n [ 91.,  96., 101., 106.]]"
}
```

---

## Problem 42: Implement ReLU Activation Function
- **Difficulty**: `easy`
- **Category**: Deep Learning

### Description & Spec
Write a Python function `relu` that implements the Rectified Linear Unit (ReLU) activation function. The function should take a single float as input and return the value after applying the ReLU function. The ReLU function returns the input if it's greater than 0, otherwise, it returns 0.

### Mathematical Derivation & Intuition
## Understanding the ReLU Activation Function

The ReLU (Rectified Linear Unit) activation function is widely used in neural networks, particularly in hidden layers of deep learning models. It maps any real-valued number to the non-negative range $[0, \infty)$, which helps introduce non-linearity into the model while maintaining computational efficiency.

### Mathematical Definition
The ReLU function is mathematically defined as:
$$
f(z) = \max(0, z)
$$
where $z$ is the input to the function.

### Characteristics
- **Output Range**: The output is always in the range $[0, \infty)$. Values below 0 are mapped to 0, while positive values are retained.
- **Shape**: The function has an "L" shaped curve with a horizontal axis at $y = 0$ and a linear increase for positive $z$.
- **Gradient**: The gradient is 1 for positive values of $z$ and 0 for non-positive values. This means the function is linear for positive inputs and flat (zero gradient) for negative inputs.

This function is particularly useful in deep learning models as it introduces non-linearity while being computationally efficient, helping to capture complex patterns in the data.

### Verified Pure NumPy Solution (From Scratch)
```python
def relu(z: float) -> float:
    return max(0, z)
```

### Verified Sample Trace Data
```json
{
  "test": "print(relu(0))",
  "expected_output": "0"
}
```

---

## Problem 44: Leaky ReLU Activation Function
- **Difficulty**: `easy`
- **Category**: Deep Learning

### Description & Spec
Write a Python function `leaky_relu` that implements the Leaky Rectified Linear Unit (Leaky ReLU) activation function. The function should take a float `z` as input and an optional float `alpha`, with a default value of 0.01, as the slope for negative inputs. The function should return the value after applying the Leaky ReLU function.

### Mathematical Derivation & Intuition
## Understanding the Leaky ReLU Activation Function

The Leaky ReLU (Leaky Rectified Linear Unit) activation function is a variant of the ReLU function used in neural networks. It addresses the "dying ReLU" problem by allowing a small, non-zero gradient when the input is negative. This small slope for negative inputs helps keep the function active and prevents neurons from becoming inactive.

### Mathematical Definition
The Leaky ReLU function is mathematically defined as:
$$
f(z) = \begin{cases} 
z & \text{if } z > 0 \\ 
\alpha z & \text{if } z \leq 0 
\end{cases}
$$
where $z$ is the input to the function and $\alpha$ is a small positive constant, typically $\alpha = 0.01$.

In this definition, the function returns $z$ for positive values, and for negative values, it returns $\alpha z$, allowing a small gradient to pass through.

### Characteristics
- **Output Range**: The output is in the range $(-\infty, \infty)$. Positive values are retained, while negative values are scaled by the factor $\alpha$, allowing them to be slightly negative.
- **Shape**: The function has a similar "L" shaped curve as ReLU, but with a small negative slope on the left side for negative $z$, creating a small gradient for negative inputs.
- **Gradient**: The gradient is 1 for positive values of $z$ and $\alpha$ for non-positive values. This allows the function to remain active even for negative inputs, unlike ReLU, where the gradient is zero for negative inputs.

This function is particularly useful in deep learning models as it mitigates the issue of "dead neurons" in ReLU by ensuring that neurons can still propagate a gradient even when the input is negative, helping to improve learning dynamics in the network.

### Verified Pure NumPy Solution (From Scratch)
```python
def leaky_relu(z: float, alpha: float = 0.01) -> float|int:
    return z if z > 0 else alpha * z
```

### Verified Sample Trace Data
```json
{
  "test": "print(leaky_relu(5))",
  "expected_output": "5"
}
```

---

## Problem 49: Implement Adam Optimization Algorithm
- **Difficulty**: `medium`
- **Category**: Deep Learning

### Description & Spec
Implement the Adam (Adaptive Moment Estimation) optimization algorithm in Python. Adam is an optimization algorithm that adapts the learning rate for each parameter. Your task is to write a function `adam_optimizer` that updates the parameters of a given function using the Adam algorithm.

The function should take the following parameters:

- `f`: The objective function to be optimized
- `grad`: A function that computes the gradient of `f`
- `x0`: Initial parameter values
- `learning_rate`: The step size (default: 0.001)
- `beta1`: Exponential decay rate for the first moment estimates (default: 0.9)
- `beta2`: Exponential decay rate for the second moment estimates (default: 0.999)
- `epsilon`: A small constant for numerical stability (default: 1e-8)
- `num_iterations`: Number of iterations to run the optimizer (default: 1000)

The function should return the optimized parameters.

### Mathematical Derivation & Intuition
## Understanding the Adam Optimization Algorithm

Adam (Adaptive Moment Estimation) is an optimization algorithm commonly used in training deep neural networks. It combines ideas from two other optimization algorithms: RMSprop and Momentum.

### Key Concepts
1. **Adaptive Learning Rates**:  
   Adam computes individual adaptive learning rates for different parameters from estimates of first and second moments of the gradients.
2. **Momentum**:  
   It keeps track of an exponentially decaying average of past gradients, similar to momentum.
3. **RMSprop**:  
   It also keeps track of an exponentially decaying average of past squared gradients.
4. **Bias Correction**:  
   Adam includes bias correction terms to account for the initialization of the first and second moment estimates.

### The Adam Algorithm
Given parameters $ \theta $, objective function $ f(\theta) $, and its gradient $ \nabla_\theta f(\theta) $:
1. **Initialize**:
   - Time step $ t = 0 $
   - Parameters $ \theta_0 $
   - First moment vector $ m_0 = 0 $
   - Second moment vector $ v_0 = 0 $
   - Hyperparameters $ \alpha $ (learning rate), $ \beta_1 $, $ \beta_2 $, and $ \epsilon $
2. **While not converged, do**:
   1. Increment time step: $ t = t + 1 $
   2. Compute gradient: $ g_t = \nabla_\theta f_t(\theta_{t-1}) $
   3. Update biased first moment estimate: $ m_t = \beta_1 \cdot m_{t-1} + (1 - \beta_1) \cdot g_t $
   4. Update biased second raw moment estimate: $ v_t = \beta_2 \cdot v_{t-1} + (1 - \beta_2) \cdot g_t^2 $
   5. Compute bias-corrected first moment estimate: $ \hat{m}_t = m_t / (1 - \beta_1^t) $
   6. Compute bias-corrected second raw moment estimate: $ \hat{v}_t = v_t / (1 - \beta_2^t) $
   7. Update parameters: $ \theta_t = \theta_{t-1} - \alpha \cdot \hat{m}_t / (\sqrt{\hat{v}_t} + \epsilon) $

Adam combines the advantages of AdaGrad, which works well with sparse gradients, and RMSProp, which works well in online and non-stationary settings. Adam is generally regarded as being fairly robust to the choice of hyperparameters, though the learning rate may sometimes need to be changed from the suggested default.

### Verified Pure NumPy Solution (From Scratch)
```python
import numpy as np

def adam_optimizer(f, grad, x0, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, num_iterations=10):
    x = x0
    m = np.zeros_like(x)
    v = np.zeros_like(x)

    for t in range(1, num_iterations + 1):
        g = grad(x)
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * g**2
        m_hat = m / (1 - beta1**t)
        v_hat = v / (1 - beta2**t)
        x = x - learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)

    return x
```

### Verified Sample Trace Data
```json
{
  "test": "\nimport numpy as np\n\ndef objective_function(x):\n    return x[0]**2 + x[1]**2\n\ndef gradient(x):\n    return np.array([2*x[0], 2*x[1]])\n\nx0 = np.array([1.0, 1.0])\nx_opt = adam_optimizer(objective_function, gradient, x0)\n\nprint(x_opt)\n",
  "expected_output": "[0.99000325 0.99000325]"
}
```

---

## Problem 53: Implement Self-Attention Mechanism
- **Difficulty**: `medium`
- **Category**: Deep Learning

### Description & Spec
## Task: Implement the Self-Attention Mechanism

Your task is to implement the **self-attention** mechanism, which is a fundamental component of transformer models, widely used in natural language processing and computer vision tasks. The self-attention mechanism allows a model to dynamically focus on different parts of the input sequence when generating a contextualized representation.

Your function should return the self-attention output as a numpy array.

### Mathematical Derivation & Intuition
## Self-Attention Mechanism

The **self-attention mechanism** is a fundamental concept in **transformer models** and is widely used in **natural language processing (NLP)** and **computer vision (CV)**. It allows models to dynamically weigh different parts of the input sequence, enabling them to capture **long-range dependencies** effectively.

---

### **Understanding Self-Attention**

Self-attention helps a model determine **which parts of an input sequence are relevant to each other**. Instead of treating every word or token equally, self-attention assigns different weights to different parts of the sequence, allowing the model to capture contextual relationships.

For example, in machine translation, self-attention allows the model to **focus on relevant words** from the input sentence when generating each word in the output.

---

### **Mathematical Formulation of Self-Attention**

Given an input sequence $X$, self-attention computes three key components:

1. **Query ($Q$)**: Represents the current token we are processing.
2. **Key ($K$)**: Represents each token in the sequence.
3. **Value ($V$)**: Contains the actual token embeddings.

The Query, Key, and Value matrices are computed as:

$$
Q = X W_Q, \quad K = X W_K, \quad V = X W_V
$$

where $W_Q$, $W_K$, and $W_V$ are learned weight matrices.

The attention scores are computed using the **scaled dot-product attention**:

$$
\text{Attention}(Q, K, V) = \text{softmax} \left( \frac{Q K^T}{\sqrt{d_k}} \right) V
$$

where $d_k$ is the dimensionality of the key vectors.

---

### **Why Self-Attention is Powerful?**

- **Captures long-range dependencies**: Unlike RNNs, which process input sequentially, self-attention can relate any word in the sequence to any other word, regardless of distance.
- **Parallelization**: Since self-attention is computed **simultaneously** across the entire sequence, it is much faster than sequential models like LSTMs.
- **Contextual Understanding**: Each token is **contextually enriched** by attending to relevant tokens in the sequence.

---

### **Example Calculation**

Consider an input sequence of three tokens:

$$
X = \begin{bmatrix} x_1 \\ x_2 \\ x_3 \end{bmatrix}
$$

We compute $Q$, $K$, and $V$ as:

$$
Q = X W_Q, \quad K = X W_K, \quad V = X W_V
$$

Next, we compute the attention scores:

$$
S = \frac{Q K^T}{\sqrt{d_k}}
$$

Applying the softmax function:

$$
A = \text{softmax}(S)
$$

Finally, the weighted sum of values:

$$
\text{Output} = A V
$$

---

### **Applications of Self-Attention**

Self-attention is widely used in:
- **Transformer models (e.g., BERT, GPT-3)** for language modeling.
- **Speech processing models** for transcribing audio.
- **Vision Transformers (ViTs)** for computer vision tasks.
- **Recommender systems** for learning item-user relationships.

Mastering self-attention is essential for understanding modern deep learning architectures, especially in NLP and computer vision.

### Verified Pure NumPy Solution (From Scratch)
```python
import numpy as np

def compute_qkv(X, W_q, W_k, W_v):
    Q = np.dot(X, W_q)
    K = np.dot(X, W_k)
    V = np.dot(X, W_v)
    return Q, K, V

def self_attention(Q, K, V):
    d_k = Q.shape[1]
    scores = np.matmul(Q, K.T) / np.sqrt(d_k)
    attention_weights = np.exp(scores) / np.sum(np.exp(scores), axis=1, keepdims=True)
    attention_output = np.matmul(attention_weights, V)
    return attention_output
```

### Verified Sample Trace Data
```json
{
  "test": "import numpy as np\n\nX = np.array([[1, 0], [0, 1]])\nW_q = np.array([[1, 0], [0, 1]])\nW_k = np.array([[1, 0], [0, 1]])\nW_v = np.array([[1, 2], [3, 4]])\n\nQ, K, V = compute_qkv(X, W_q, W_k, W_v)\noutput = self_attention(Q, K, V)\nprint(output)",
  "expected_output": "[[1.660477, 2.660477], [2.339523, 3.339523]]"
}
```

---

## Problem 54: Implementing a Simple RNN
- **Difficulty**: `medium`
- **Category**: Deep Learning

### Description & Spec
Write a Python function that implements a simple Recurrent Neural Network (RNN) cell. The function should process a sequence of input vectors and produce the final hidden state. Use the tanh activation function for the hidden state updates. The function should take as inputs the sequence of input vectors, the initial hidden state, the weight matrices for input-to-hidden and hidden-to-hidden connections, and the bias vector. The function should return the final hidden state after processing the entire sequence, rounded to four decimal places.

### Mathematical Derivation & Intuition
## Understanding Recurrent Neural Networks (RNNs)

Recurrent Neural Networks (RNNs) are a class of neural networks designed to handle sequential data by maintaining a hidden state that captures information from previous inputs.

### Mathematical Formulation

For each time step $t$, the RNN updates its hidden state $h_t$ using the current input $x_t$ and the previous hidden state $h_{t-1}$:

$$
h_t = \tanh(W_x x_t + W_h h_{t-1} + b)
$$

Where:
1. $W_x$ is the weight matrix for the input-to-hidden connections.
2. $W_h$ is the weight matrix for the hidden-to-hidden connections.
3. $b$ is the bias vector.
4. $\tanh$ is the hyperbolic tangent activation function applied element-wise.

### Implementation Steps

1. **Initialization**: Start with the initial hidden state $h_0$.

2. **Sequence Processing**: For each input $x_t$ in the sequence:

   $$
   h_t = \tanh(W_x x_t + W_h h_{t-1} + b)
   $$

3. **Final Output**: After processing all inputs, the final hidden state $h_T$ (where $T$ is the length of the sequence) contains information from the entire sequence.

### Example Calculation

Given:
1. Inputs: $x_1 = 1.0$, $x_2 = 2.0$, $x_3 = 3.0$
2. Initial hidden state: $h_0 = 0.0$
3. Weights:
   - $W_x = 0.5$
   - $W_h = 0.8$
4. Bias: $b = 0.0$

**Compute**:

1. First time step ($t = 1$):

   $$
   h_1 = \tanh(0.5 \times 1.0 + 0.8 \times 0.0 + 0.0) = \tanh(0.5) \approx 0.4621
   $$

2. Second time step ($t = 2$):

   $$
   h_2 = \tanh(0.5 \times 2.0 + 0.8 \times 0.4621 + 0.0) = \tanh(1.0 + 0.3697) = \tanh(1.3697) \approx 0.8781
   $$

3. Third time step ($t = 3$):

   $$
   h_3 = \tanh(0.5 \times 3.0 + 0.8 \times 0.8781 + 0.0) = \tanh(1.5 + 0.7025) = \tanh(2.2025) \approx 0.9750
   $$

The final hidden state $h_3$ is approximately 0.9750.

### Applications

RNNs are widely used in natural language processing, time-series prediction, and any task involving sequential data.

### Verified Pure NumPy Solution (From Scratch)
```python
import numpy as np

def rnn_forward(input_sequence, initial_hidden_state, Wx, Wh, b):
    h = np.array(initial_hidden_state)
    Wx = np.array(Wx)
    Wh = np.array(Wh)
    b = np.array(b)
    for x in input_sequence:
        x = np.array(x)
        h = np.tanh(np.dot(Wx, x) + np.dot(Wh, h) + b)
    final_hidden_state = np.round(h, 4)
    return final_hidden_state.tolist()
```

### Verified Sample Trace Data
```json
{
  "test": "print(rnn_forward([[1.0], [2.0], [3.0]], [0.0], [[0.5]], [[0.8]], [0.0]))",
  "expected_output": "[0.9759]"
}
```

---

## Problem 56: KL Divergence Between Two Normal Distributions
- **Difficulty**: `easy`
- **Category**: Deep Learning

### Description & Spec
## Task: Implement KL Divergence Between Two Normal Distributions

Your task is to compute the Kullback Leibler (KL) divergence between two normal distributions. KL divergence measures how one probability distribution differs from a second, reference probability distribution.

Write a function kl_divergence_normal(mu_p, sigma_p, mu_q, sigma_q) that calculates the KL divergence between two normal distributions.

The function should return the KL divergence as a floating point number.

### Mathematical Derivation & Intuition
## Understanding Kullback-Leibler Divergence (KL Divergence)

The **Kullback-Leibler (KL) divergence**, also known as relative entropy, measures the difference between two probability distributions. It quantifies how much information is lost when approximating one distribution with another.

---

### Definition of KL Divergence

For continuous variables, the KL divergence is defined as:

$$
KL(P \parallel Q) = \int p(x) \log \frac{p(x)}{q(x)} \, dx
$$

where:
- $p(x)$ is the probability density function of the **reference** distribution $P$.
- $q(x)$ is the probability density function of the **comparison** distribution $Q$.

---

### KL Divergence Between Two Normal Distributions

Consider two normal distributions $P$ and $Q$:

- $P \sim N(\mu_P, \sigma_P^2)$  
- $Q \sim N(\mu_Q, \sigma_Q^2)$

For these, the KL divergence simplifies to:

$$
KL(P \parallel Q) = \int p(x) \left[
\log \frac{\sigma_Q}{\sigma_P}
+ \frac{\sigma_P^2 + (\mu_P - \mu_Q)^2}{2\sigma_Q^2}
- \frac{1}{2}
\right] dx
$$

Since $p(x)$ is the PDF of $x$ under $P$, the integral over $p(x)$ just multiplies by 1 for each constant term. Thus, the final closed form is:

$$
KL(P \parallel Q) =
\log \frac{\sigma_Q}{\sigma_P}
+ \frac{\sigma_P^2 + (\mu_P - \mu_Q)^2}{2\sigma_Q^2}
- \frac{1}{2}
$$

---

### Interpretation

This expression quantifies how one normal distribution $P$ **diverges** from another normal distribution $Q$. A KL divergence of zero indicates the two distributions are identical. As the divergence grows, it signals that $Q$ is a poorer approximation of $P$.

The KL divergence is **asymmetric**:

$$
KL(P \parallel Q) \neq KL(Q \parallel P)
$$

making it sensitive to the **direction** of comparison.

### Verified Pure NumPy Solution (From Scratch)
```python
import numpy as np

def kl_divergence_normal(mu_p, sigma_p, mu_q, sigma_q):
    term1 = np.log(sigma_q / sigma_p)
    term2 = (sigma_p ** 2 + (mu_p - mu_q) ** 2) / (2 * sigma_q ** 2)
    kl_div = term1 + term2 - 0.5
    return kl_div
```

### Verified Sample Trace Data
```json
{
  "test": "import numpy as np\n\nmu_p = 0.0\nsigma_p = 1.0\nmu_q = 0.0\nsigma_q = 1.0\nprint(kl_divergence_normal(mu_p, sigma_p, mu_q, sigma_q))",
  "expected_output": "0.0"
}
```

---

## Problem 59: Implement Long Short-Term Memory (LSTM) Network
- **Difficulty**: `medium`
- **Category**: Deep Learning

### Description & Spec
## Task: Implement Long Short-Term Memory (LSTM) Network

Your task is to implement an LSTM network that processes a sequence of inputs and produces the final hidden state and cell state after processing all inputs.

Write a class `LSTM` with the following methods:

- `__init__(self, input_size, hidden_size)`: Initializes the LSTM with random weights and zero biases.
- `forward(self, x, initial_hidden_state, initial_cell_state)`: Processes a sequence of inputs and returns the hidden states at each time step, as well as the final hidden state and cell state.

The LSTM should compute the forget gate, input gate, candidate cell state, and output gate at each time step to update the hidden state and cell state.

### Mathematical Derivation & Intuition
## Understanding Long Short-Term Memory Networks (LSTMs)

Long Short-Term Memory Networks are a special type of RNN designed to capture long-term dependencies in sequential data by using a more complex hidden state structure.

### LSTM Gates and Their Functions

For each time step $t$, the LSTM updates its cell state $c_t$ and hidden state $h_t$ using the current input $x_t$, the previous cell state $c_{t-1}$, and the previous hidden state $h_{t-1}$. The LSTM architecture consists of several gates that control the flow of information:

#### Forget Gate $f_t$:

This gate decides what information to discard from the cell state. It looks at the previous hidden state $h_{t-1}$ and the current input $x_t$, and outputs a number between 0 and 1 for each number in the cell state. A 1 represents "keep this" while a 0 represents "forget this".

$$
f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)
$$

#### Input Gate $i_t$:

This gate decides which new information will be stored in the cell state. It consists of two parts:
- A sigmoid layer that decides which values we'll update.
- A tanh layer that creates a vector of new candidate values $\tilde{c}_t$ that could be added to the state.

$$
i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)
$$

$$
\tilde{c}_t = \tanh(W_c \cdot [h_{t-1}, x_t] + b_c)
$$

#### Cell State Update $c_t$:

This step updates the old cell state $c_{t-1}$ into the new cell state $c_t$. It multiplies the old state by the forget gate output, then adds the product of the input gate and the new candidate values.

$$
c_t = f_t \circ c_{t-1} + i_t \circ \tilde{c}_t
$$

#### Output Gate $o_t$:

This gate decides what parts of the cell state we're going to output. It uses a sigmoid function to determine which parts of the cell state to output, and then multiplies it by a tanh of the cell state to get the final output.

$$
o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)
$$

$$
h_t = o_t \circ \tanh(c_t)
$$

Where:
- $(W_f, W_i, W_c, W_o)$ are weight matrices for the forget gate, input gate, cell state, and output gate respectively.
- $(b_f, b_i, b_c, b_o)$ are bias vectors.
- $\sigma$ is the sigmoid activation function.
- $\circ$ denotes element-wise multiplication.

### Implementation Steps

1. **Initialization**: Start with the initial cell state $c_0$ and hidden state $h_0$.
2. **Sequence Processing**: For each input $x_t$ in the sequence:
   - Compute forget gate $f_t$, input gate $i_t$, candidate cell state $\tilde{c}_t$, and output gate $o_t$.
   - Update cell state $c_t$ and hidden state $h_t$.
3. **Final Output**: After processing all inputs, the final hidden state $h_T$ (where $T$ is the length of the sequence) contains information from the entire sequence.

### Example Calculation

Given:
- Inputs: $x_1 = 1.0$, $x_2 = 2.0$, $x_3 = 3.0$
- Initial states: $c_0 = 0.0$, $h_0 = 0.0$
- Simplified weights (for demonstration): $W_f = W_i = W_c = W_o = 0.5$
- All biases: $b_f = b_i = b_c = b_o = 0.1$

#### Compute:

**First time step $t = 1$:**

$$
f_1 = \sigma(0.5 \times 1.0 + 0.1) = 0.6487
$$

$$
i_1 = \sigma(0.5 \times 1.0 + 0.1) = 0.6487
$$

$$
\tilde{c}_1 = \tanh(0.5 \times 1.0 + 0.1) = 0.5370
$$

$$
c_1 = f_1 \times 0.0 + i_1 \times \tilde{c}_1 = 0.6487 \times 0.0 + 0.6487 \times 0.5370 = 0.3484
$$

$$
o_1 = \sigma(0.5 \times 1.0 + 0.1) = 0.6487
$$

$$
h_1 = o_1 \times \tanh(c_1) = 0.6487 \times \tanh(0.3484) = 0.2169
$$

**Second time step $t = 2$:**
(Calculations omitted for brevity, but follow the same pattern using $x_2 = 2.0$ and the previous states)

**Third time step $t = 3$:**
(Calculations omitted for brevity, but follow the same pattern using $x_3 = 3.0$ and the previous states)

The final hidden state $h_3$ would be the result after these calculations.

### Applications

LSTMs are extensively used in various sequence modeling tasks, including machine translation, speech recognition, and time series forecasting, where capturing long-term dependencies is crucial.

### Verified Pure NumPy Solution (From Scratch)
```python
import numpy as np

class LSTM:
    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Initialize weights and biases
        self.Wf = np.random.randn(hidden_size, input_size + hidden_size)
        self.Wi = np.random.randn(hidden_size, input_size + hidden_size)
        self.Wc = np.random.randn(hidden_size, input_size + hidden_size)
        self.Wo = np.random.randn(hidden_size, input_size + hidden_size)

        self.bf = np.zeros((hidden_size, 1))
        self.bi = np.zeros((hidden_size, 1))
        self.bc = np.zeros((hidden_size, 1))
        self.bo = np.zeros((hidden_size, 1))

    def forward(self, x, initial_hidden_state, initial_cell_state):
        h = initial_hidden_state
        c = initial_cell_state
        outputs = []

        for t in range(len(x)):
            xt = x[t].reshape(-1, 1)
            concat = np.vstack((h, xt))

            # Forget gate
            ft = self.sigmoid(np.dot(self.Wf, concat) + self.bf)

            # Input gate
            it = self.sigmoid(np.dot(self.Wi, concat) + self.bi)
            c_tilde = np.tanh(np.dot(self.Wc, concat) + self.bc)

            # Cell state update
            c = ft * c + it * c_tilde

            # Output gate
            ot = self.sigmoid(np.dot(self.Wo, concat) + self.bo)

            # Hidden state update
            h = ot * np.tanh(c)

            outputs.append(h)

        return np.array(outputs), h, c

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
```

### Verified Sample Trace Data
```json
{
  "test": "import numpy as np\n\ninput_sequence = np.array([[1.0], [2.0], [3.0]])\ninitial_hidden_state = np.zeros((1, 1))\ninitial_cell_state = np.zeros((1, 1))\n\nlstm = LSTM(input_size=1, hidden_size=1)\n# Set weights and biases for reproducibility\nlstm.Wf = np.array([[0.5, 0.5]])\nlstm.Wi = np.array([[0.5, 0.5]])\nlstm.Wc = np.array([[0.3, 0.3]])\nlstm.Wo = np.array([[0.5, 0.5]])\nlstm.bf = np.array([[0.1]])\nlstm.bi = np.array([[0.1]])\nlstm.bc = np.array([[0.1]])\nlstm.bo = np.array([[0.1]])\n\noutputs, final_h, final_c = lstm.forward(input_sequence, initial_hidden_state, initial_cell_state)\n\nprint(final_h)",
  "expected_output": "[[0.73698596]]"
}
```

---

## Problem 62: Implement a Simple RNN with Backpropagation Through Time (BPTT)
- **Difficulty**: `hard`
- **Category**: Deep Learning

### Description & Spec
## Task: Implement a Simple RNN with Backpropagation Through Time (BPTT)

Your task is to implement a simple Recurrent Neural Network (RNN) and backpropagation through time (BPTT) to learn from sequential data. The RNN will process input sequences, update hidden states, and perform backpropagation to adjust weights based on the error gradient.

Write a class `SimpleRNN` with the following methods:

- `__init__(self, input_size, hidden_size, output_size)`: Initializes the RNN with random weights and zero biases.
- `forward(self, x)`: Processes a sequence of inputs and returns the hidden states and output.
- `backward(self, x, y, learning_rate)`: Performs backpropagation through time (BPTT) to adjust the weights based on the loss.

In this task, the RNN will be trained on sequence prediction, where the network will learn to predict the next item in a sequence. You should use 1/2 * Mean Squared Error (MSE) as the loss function and make sure to aggregate the losses at each time step by summing.

### Mathematical Derivation & Intuition
# Understanding Recurrent Neural Networks (RNNs) and Backpropagation Through Time (BPTT)

Recurrent Neural Networks (RNNs) are designed to handle sequential data by maintaining a hidden state that captures information from previous inputs. They are particularly useful for tasks where context or sequential order is important, such as language modeling, time series forecasting, and sequence prediction.

## RNN Architecture

An RNN processes inputs one at a time while maintaining a hidden state that gets updated at each time step. The core equations governing the forward pass of an RNN are:

### 1) Hidden State Update

$$
h_t = \tanh(W_{xh} x_t + W_{hh} h_{t-1} + b_h)
$$

### 2) Output Computation

$$
y_t = W_{hy} h_t + b_y
$$

Where:

1. $x_t$ is the input at time step $t$.
2. $h_t$ is the hidden state at time step $t$.
3. $W_{xh}$ is the weight matrix for input to hidden state.
4. $W_{hh}$ is the weight matrix for hidden state to hidden state.
5. $W_{hy}$ is the weight matrix for hidden state to output.
6. $b_h$ and $b_y$ are the bias terms for the hidden state and output, respectively.
7. $\tanh$ is the hyperbolic tangent activation function applied element-wise.

## Forward Pass Implementation

In the forward pass, we iterate over each element in the input sequence, updating the hidden state and computing the output:

1. **Initialize the hidden state** $h_0$ to zeros.
2. For each time step $t$:
   - Compute $h_t = \tanh(W_{xh} x_t + W_{hh} h_{t-1} + b_h)$.
   - Compute $y_t = W_{hy} h_t + b_y$.
   - Store $h_t$ and $y_t$ for use in backpropagation.

## Loss Function

The loss function measures the discrepancy between the predicted outputs and the actual target values. For sequence prediction tasks, we often use the **Mean Squared Error (MSE)** loss:

$$
\text{Loss} = \frac{1}{T} \sum_{t=1}^{T} (\hat{y}_t - y_t)^2
$$

Where $T$ is the length of the sequence, $\hat{y}_t$ is the predicted output, and $y_t$ is the actual target at time step $t$.

## Backpropagation Through Time (BPTT)

BPTT is the process of training RNNs by unrolling them through time and applying backpropagation to compute gradients for each time step. The key steps in BPTT are:

1. Compute the gradient of the loss with respect to the outputs:

$$
\frac{dL}{dy_t} = \hat{y}_t - y_t
$$

2. Compute the gradients for the output layer weights and biases:

$$
dW_{hy} += \frac{dL}{dy_t} \cdot h_t^T
$$

$$
db_y += \frac{dL}{dy_t}
$$

3. Backpropagate the gradients through the hidden layers:

$$
dh_t = W_{hy}^T \cdot \frac{dL}{dy_t} + dh_{t+1}
$$

$$
dh_{\text{raw}} = dh_t \circ (1 - h_t^2)
$$

Here, $\circ$ denotes element-wise multiplication, and $(1 - h_t^2)$ is the derivative of the $\tanh$ activation function.

4. Compute the gradients for the hidden layer weights and biases:

$$
dW_{xh} += dh_{\text{raw}} \cdot x_t^T
$$

$$
dW_{hh} += dh_{\text{raw}} \cdot h_{t-1}^T
$$

$$
db_h += dh_{\text{raw}}
$$

We repeat steps 1-4 for each time step $t$ in reverse order (from $T$ to 1), accumulating the gradients. The term $dh_{t+1}$ represents the gradient flowing from the next time step, initialized to zeros at the last time step.

## Updating Weights

After computing the gradients, we update the weights and biases using gradient descent:

$$
W_{xh} -= \text{learning\_rate} \times dW_{xh}
$$

$$
W_{hh} -= \text{learning\_rate} \times dW_{hh}
$$

$$
W_{hy} -= \text{learning\_rate} \times dW_{hy}
$$

$$
b_h -= \text{learning\_rate} \times db_h
$$

$$
b_y -= \text{learning\_rate} \times db_y
$$

## Implementing the RNN

To implement the RNN with BPTT, follow these steps:

1. **Initialization**: Initialize the weight matrices $W_{xh}, W_{hh}, W_{hy}$ with small random values and biases $b_h, b_y$ with zeros.
2. **Forward Pass**: Implement the forward method to process the input sequence, updating the hidden states and computing the outputs at each time step. Store the inputs, hidden states, and outputs for use in backpropagation.
3. **Backward Pass**: Implement the backward method to perform BPTT. Compute the gradients at each time step in reverse order, accumulate them, and update the weights and biases.
4. **Training Loop**: Train the RNN over multiple epochs by repeatedly performing forward and backward passes and updating the weights.

## Tips for Implementation

1. **Gradient Clipping**: To prevent exploding gradients, consider applying gradient clipping, which scales down gradients if they exceed a certain threshold.
2. **Learning Rate**: Choose an appropriate learning rate. If the learning rate is too high, the training may become unstable.
3. **Debugging**: Check the dimensions of all matrices and vectors to ensure they align correctly during matrix multiplication.
4. **Testing**: Start with small sequences and hidden sizes to test your implementation before scaling up.

## Example Calculation

Suppose we have an input sequence $x = [x_1, x_2]$ and target sequence $y = [y_1, y_2]$. Here's how you would compute the forward and backward passes:

### 1) Forward Pass

- At $t = 1$:
  - Compute $h_1 = \tanh(W_{xh} x_1 + W_{hh} h_0 + b_h)$.
  - Compute $\hat{y}_1 = W_{hy} h_1 + b_y$.

- At $t = 2$:
  - Compute $h_2 = \tanh(W_{xh} x_2 + W_{hh} h_1 + b_h)$.
  - Compute $\hat{y}_2 = W_{hy} h_2 + b_y$.

### 2) Compute Loss

$$
L = \frac{1}{2} \left[ (\hat{y}_1 - y_1)^2 + (\hat{y}_2 - y_2)^2 \right]
$$

### 3) Backward Pass

Starting from $t = 2$ to $t = 1$:

- At $t = 2$:
  - Compute $\frac{dL}{d\hat{y}_2} = \hat{y}_2 - y_2$.
  - Backpropagate to find $dh_2$, $dW_{hy}$, $db_y$.

- At $t = 1$:
  - Use the chain rule to compute gradients with respect to inputs and weights.

## Conclusion

RNNs with BPTT are powerful for modeling sequences, but they come with challenges such as vanishing and exploding gradients. Techniques like gradient clipping, long short-term memory (LSTM) cells, or gated recurrent units (GRUs) can help mitigate these issues. Implementing an RNN from scratch provides a deeper understanding of the underlying mechanisms and prepares you for working with more complex architectures.

### Verified Pure NumPy Solution (From Scratch)
```python
import numpy as np

class SimpleRNN:
    def __init__(self, input_size, hidden_size, output_size):
        self.hidden_size = hidden_size
        self.W_xh = np.random.randn(hidden_size, input_size) * 0.01
        self.W_hh = np.random.randn(hidden_size, hidden_size) * 0.01
        self.W_hy = np.random.randn(output_size, hidden_size) * 0.01
        self.b_h = np.zeros((hidden_size, 1))
        self.b_y = np.zeros((output_size, 1))

    def forward(self, x):
        h = np.zeros((self.hidden_size, 1))  # Initialize hidden state
        outputs = []
        self.last_inputs = []
        self.last_hiddens = [h]
        
        for t in range(len(x)):
            self.last_inputs.append(x[t].reshape(-1, 1))
            h = np.tanh(np.dot(self.W_xh, self.last_inputs[t]) + np.dot(self.W_hh, h) + self.b_h)
            y = np.dot(self.W_hy, h) + self.b_y
            outputs.append(y)
            self.last_hiddens.append(h)
        
        self.last_outputs = outputs
        return np.array(outputs)

    def backward(self, x, y, learning_rate):
        dW_xh = np.zeros_like(self.W_xh)
        dW_hh = np.zeros_like(self.W_hh)
        dW_hy = np.zeros_like(self.W_hy)
        db_h = np.zeros_like(self.b_h)
        db_y = np.zeros_like(self.b_y)

        dh_next = np.zeros((self.hidden_size, 1))

        for t in reversed(range(len(x))):
            dy = self.last_outputs[t] - y[t].reshape(-1, 1)  # (Predicted - Actual)
            dW_hy += np.dot(dy, self.last_hiddens[t+1].T)
            db_y += dy

            dh = np.dot(self.W_hy.T, dy) + dh_next
            dh_raw = (1 - self.last_hiddens[t+1] ** 2) * dh  # Derivative of tanh

            dW_xh += np.dot(dh_raw, self.last_inputs[t].T)
            dW_hh += np.dot(dh_raw, self.last_hiddens[t].T)
            db_h += dh_raw

            dh_next = np.dot(self.W_hh.T, dh_raw)

        # Update weights and biases
        self.W_xh -= learning_rate * dW_xh
        self.W_hh -= learning_rate * dW_hh
        self.W_hy -= learning_rate * dW_hy
        self.b_h -= learning_rate * db_h
        self.b_y -= learning_rate * db_y
```

### Verified Sample Trace Data
```json
{
  "test": "import numpy as np\n\nnp.random.seed(42) \n\ninput_sequence = np.array([[1.0], [2.0], [3.0], [4.0]])\nexpected_output = np.array([[2.0], [3.0], [4.0], [5.0]])\n\nrnn = SimpleRNN(input_size=1, hidden_size=5, output_size=1)\n\n# Train the RNN over multiple epochs\nfor epoch in range(100):\n    output = rnn.forward(input_sequence)\n    rnn.backward(input_sequence, expected_output, learning_rate=0.01)\n\nprint(output)",
  "expected_output": "[[[2.24143915]],[[3.18450265]],[[4.04305928]],[[4.57419398]]]"
}
```

---

## Problem 85: Positional Encoding Calculator
- **Difficulty**: `hard`
- **Category**: Deep Learning

### Description & Spec
Write a Python function to implement the Positional Encoding layer for Transformers.
      The function should calculate positional encodings for a sequence length (`position`) and model dimensionality (`d_model`) using sine and cosine functions as specified in the Transformer architecture.
      The function should return -1 if `position` is 0, or if `d_model` is less than or equal to 0. The output should be a numpy array of type `float16`.

### Mathematical Derivation & Intuition
## **The Positional Encoding Layer in Transformers**

The Positional Encoding layer in Transformers plays a critical role by providing necessary positional information to the model. 
This is particularly important because the Transformer architecture, unlike RNNs or LSTMs, processes input sequences in parallel
and lacks inherent mechanisms to account for the sequential order of tokens.

The mathematical intuition behind the Positional Encoding layer in Transformers is centered on enabling the model to incorporate
information about the order of tokens in a sequence.

---

### **Function Parameters**

- **`position`**: Total positions or length of the sequence.
- **`d_model`**: Dimensionality of the model's output.

---

### **Generating the Base Matrix**

- **`angle_rads`**: Creates a matrix where rows represent sequence positions and columns represent feature dimensions.
 Values are scaled by dividing each position index by:  
  $10000^{\frac{2 \cdot i}{d_{model}}}$

---

### **Applying Sine and Cosine Functions**

- For even indices: Apply the sine function to encode positions.  
  $PE(\text{pos}, 2i) = \sin\left(\frac{\text{pos}}{10000^{\frac{2i}{d_{model}}}}\right)$

- For odd indices: Apply the cosine function for a phase-shifted encoding.  
  $PE(\text{pos}, 2i+1) = \cos\left(\frac{\text{pos}}{10000^{\frac{2i}{d_{model}}}}\right)$

---

### **Creating the Positional Encoding Tensor**

- The matrix is expanded to match input shape expectations of models like Transformers and cast to `float32`.

---

### **Output**

Returns a TensorFlow tensor of shape $(1, \text{position}, \text{d\_model})$, ready to be added to input embeddings
to incorporate positional information.

### Verified Pure NumPy Solution (From Scratch)
```python
import numpy as np

def pos_encoding(position: int, d_model: int):
    
    if position == 0 or d_model <= 0:
        return -1

    # Create position and dimension indices
    pos = np.arange(position, dtype=np.float32).reshape(position, 1)
    ind = np.arange(d_model, dtype=np.float32).reshape(1, d_model)

    # Compute the angles
    angle_rads = pos / np.power(10000, (2 * (ind // 2)) / d_model)

    # Apply sine to even indices, cosine to odd indices
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])  # Even indices (0, 2, 4...)
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])  # Odd indices (1, 3, 5...)

    # Convert to float16 as required
    return angle_rads.astype(np.float16)
```

### Verified Sample Trace Data
```json
{
  "test": "print(pos_encoding(2, 8))",
  "expected_output": "[[0.     , 1.     , 0.     , 1.     , 0.     , 1.     , 0.     ,\n        1.     ],\n       [0.8413 , 0.5405 , 0.09985, 0.995  , 0.01   , 1.     , 0.001  ,\n        1.     ]]"
}
```

---

## Problem 87: Adam Optimizer
- **Difficulty**: `medium`
- **Category**: Deep Learning

### Description & Spec
Implement the Adam optimizer update step function. Your function should take the current parameter value, gradient, and moving averages as inputs, and return the updated parameter value and new moving averages. The function should also handle scalar and array inputs and include bias correction for the moving averages.

### Mathematical Derivation & Intuition
# Implementing Adam Optimizer

## Introduction
Adam (Adaptive Moment Estimation) is a popular optimization algorithm used in training deep learning models. It combines the benefits of two other optimization algorithms: RMSprop and momentum optimization.

## Learning Objectives
- Understand how Adam optimizer works
- Learn to implement adaptive learning rates
- Understand bias correction in optimization algorithms
- Gain practical experience with gradient-based optimization

## Theory
Adam maintains moving averages of both gradients (first moment) and squared gradients (second moment) to adapt the learning rate for each parameter. The key equations are:

$m_t = \beta_1 m_{t-1} + (1-\beta_1)g_t$ (First moment)
$v_t = \beta_2 v_{t-1} + (1-\beta_2)g_t^2$ (Second moment)

Bias correction:
$\hat{m}_t = \frac{m_t}{1-\beta_1^t}$
$\hat{v}_t = \frac{v_t}{1-\beta_2^t}$

Parameter update:
$\theta_t = \theta_{t-1} - \alpha \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$

## Problem Statement
Implement the Adam optimizer update step function. Your function should take the current parameter value, gradient, and moving averages as inputs, and return the updated parameter value and new moving averages.

### Input Format
The function should accept:
- parameter: Current parameter value
- grad: Current gradient
- m: First moment estimate
- v: Second moment estimate
- t: Current timestep
- learning_rate: Learning rate (default=0.001)
- beta1: First moment decay rate (default=0.9)
- beta2: Second moment decay rate (default=0.999)
- epsilon: Small constant for numerical stability (default=1e-8)

### Output Format
Return tuple: (updated_parameter, updated_m, updated_v)

## Example
# Example usage:
parameter = 1.0
grad = 0.1
m = 0.0
v = 0.0
t = 1

new_param, new_m, new_v = adam_optimizer(parameter, grad, m, v, t)
## Tips
- Initialize m and v as zeros
- Keep track of timestep t for bias correction
- Use numpy for numerical operations
- Test with both scalar and array inputs

### Verified Pure NumPy Solution (From Scratch)
```python
import numpy as np

def adam_optimizer(parameter, grad, m, v, t, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
    """
    Update parameters using the Adam optimizer.
    Adjusts the learning rate based on the moving averages of the gradient and squared gradient.
    :param parameter: Current parameter value
    :param grad: Current gradient
    :param m: First moment estimate
    :param v: Second moment estimate
    :param t: Current timestep
    :param learning_rate: Learning rate (default=0.001)
    :param beta1: First moment decay rate (default=0.9)
    :param beta2: Second moment decay rate (default=0.999)
    :param epsilon: Small constant for numerical stability (default=1e-8)
    :return: tuple: (updated_parameter, updated_m, updated_v)
    """
    # Update biased first moment estimate
    m = beta1 * m + (1 - beta1) * grad

    # Update biased second raw moment estimate
    v = beta2 * v + (1 - beta2) * (grad**2)

    # Compute bias-corrected first moment estimate
    m_hat = m / (1 - beta1**t)

    # Compute bias-corrected second raw moment estimate
    v_hat = v / (1 - beta2**t)

    # Update parameters
    update = learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
    parameter = parameter - update

    return np.round(parameter,5), np.round(m,5), np.round(v,5)
```

### Verified Sample Trace Data
```json
{
  "test": "print(adam_optimizer(1.0, 0.1, 0.0, 0.0, 1))",
  "expected_output": "(0.999, 0.01, 0.0001)"
}
```

---

## Problem 89: The Pattern Weaver's Code
- **Difficulty**: `medium`
- **Category**: Deep Learning

### Description & Spec
Deep in the Crystal Cave, the enigmatic Pattern Weaver creates stunning sequences by uncovering the intricate relationships between crystals. Each crystal is marked by a unique numeric value, and the Weaver emphasizes that the true power of any crystal depends on how it interacts with all others. You have discovered N crystals, each with a specific value, and your task is to reveal their enhanced patterns by analyzing these relationships using self-attention. Given a sequence of crystals and their values, your task is to implement a simplified self-attention mechanism. For each crystal, calculate its relationship with every other crystal, compute the attention scores using the softmax function, and derive the final weighted pattern for each crystal. This Problem was made with the help of GroundZero AI

### Mathematical Derivation & Intuition
## Understanding Self-Attention

Self-attention is a core concept in modern deep learning architectures, particularly transformers. It helps a model understand relationships between elements in a sequence by comparing each element with every other element.

### Key Formula

The attention score between two elements $i$ and $j$ is calculated as:

$$
\text{Attention Score}_{i,j} = \frac{\text{Value}_i \times \text{Value}_j}{\sqrt{\text{Dimension}}}
$$

### Softmax Function

The softmax function converts raw attention scores into probabilities:

$$
\text{Softmax}(x_i) = \frac{e^{x_i}}{\sum_{j} e^{x_j}}
$$

### Weighted Sum

Using the softmax scores, the final value for each element is calculated as a weighted sum:

$$
\text{Final Value}_i = \sum_{j} \text{Softmax Score}_{i,j} \times \text{Value}_j
$$

### Example Calculation

Consider the following values:

- Crystal values: $[4, 2, 7, 1, 9]$
- Dimension: $1$

#### Step 1: Calculate Attention Scores

For crystal $i = 1$ ($4$):

$$
\text{Score}_{1,1} = \frac{4 \times 4}{\sqrt{1}} = 16,
\quad \text{Score}_{1,2} = \frac{4 \times 2}{\sqrt{1}} = 8,
\ldots
$$

#### Step 2: Apply Softmax

Convert scores to probabilities using softmax.

#### Step 3: Compute Weighted Sum

Multiply probabilities by crystal values and sum them to get the final value.

### Applications

Self-attention is widely used in:

- Natural Language Processing (e.g., transformers)
- Computer Vision (e.g., Vision Transformers)
- Sequence Analysis

Mastering self-attention provides a foundation for understanding advanced AI architectures.

### Verified Pure NumPy Solution (From Scratch)
```python
import numpy as np

def softmax(values):
    exps = np.exp(values - np.max(values))
    return exps / np.sum(exps)

def pattern_weaver(n, crystal_values, dimension):
    dimension_sqrt = np.sqrt(dimension)
    final_patterns = []

    for i in range(n):
        attention_scores = []
        for j in range(n):
            score = crystal_values[i] * crystal_values[j] / dimension_sqrt
            attention_scores.append(score)

        softmax_scores = softmax(attention_scores)
        weighted_sum = sum(softmax_scores[k] * crystal_values[k] for k in range(n))
        final_patterns.append(round(weighted_sum, 4))

    return final_patterns
```

### Verified Sample Trace Data
```json
{
  "test": "print(pattern_weaver(5, [4, 2, 7, 1, 9], 1))",
  "expected_output": "[8.9993, 8.9638, 9.0, 8.7259, 9.0]"
}
```

---

## Problem 94: Implement Multi-Head Attention
- **Difficulty**: `hard`
- **Category**: Deep Learning

### Description & Spec
Implement the multi-head attention mechanism, a critical component of transformer models. Given Query (Q), Key (K), and Value (V) matrices, compute the attention outputs for multiple heads and concatenate the results.

### Mathematical Derivation & Intuition
## Understanding Multi-Head Attention

Multi-head attention is a fundamental mechanism in transformer models, allowing the model to focus on different parts of the input sequence simultaneously. This enables the model to capture a wider variety of relationships and dependencies, which is crucial for handling complex data, such as natural language. By using multiple attention heads, the model learns to attend to various aspects of the input at different levels of abstraction, enhancing its ability to capture complex relationships.

### Concepts

The attention mechanism allows the model to weigh the importance of different input elements based on their relevance to a specific task. In tasks like machine translation, for example, attention helps the model focus on relevant words in a sentence to understand the overall meaning. Multi-head attention extends this concept by using multiple attention heads, each learning different representations of the input data, which improves the model's ability to capture richer relationships and dependencies.

The process of multi-head attention involves several key steps:

1. **Computing Attention Scores:** This involves calculating how much focus each element in the input should receive based on its relationship with other elements.
2. **Applying Softmax:** The attention scores are transformed into probabilities using the softmax function, which normalizes the scores so that they sum to one.
3. **Aggregating Results:** The final output is computed by taking a weighted sum of the input values, where the weights are determined by the attention scores.

### Structure of Multi-Head Attention

The attention mechanism can be described with Query (Q), Key (K), and Value (V) matrices. The process of multi-head attention works by repeating the standard attention mechanism multiple times in parallel, with different sets of learned weight matrices for each attention head.

#### 1. Splitting Q, K, and V

Assume that the input Query (Q), Key (K), and Value (V) matrices have dimensions $(\text{seqLen}, d_{model})$, where $d_{\text{model}}$ is the model dimension. In multi-head attention, these matrices are divided into n smaller matrices, each corresponding to a different attention head. Each smaller matrix has dimensions $(\text{seqLen}, d_k)$, where $d_k = \frac{d_{\text{model}}}{n}$ is the dimensionality of each head.

For each attention $\text{head}_i$, we get its subset of Query $\text{Q}_i$, Key $\text{K}_i$, and Value $\text{V}_i$. These subsets are computed independently for each head.

#### 2. Computing Attention for Each Head

Each head independently computes its attention output. The calculation is similar to the single-head attention mechanism:

$$
\text{score}_i = \frac{Q_i K_i^T}{\sqrt{d_k}}
$$

Where $$d_k$$ is the dimensionality of the key space for each head. The scaling factor $$\frac{1}{\sqrt{d_k}}$$ ensures the dot product doesn't grow too large, preventing instability in the softmax function.

The softmax function is applied to the scores to normalize them, transforming them into attention weights for each head:

$$
\text{SoftmaxScore}_i = \text{softmax}(\text{score}_i)
$$

#### 3. Softmax Calculation and Numerical Stability

When computing the softmax function, especially in the context of attention mechanisms, there's a risk of numerical overflow or underflow, which can occur when the attention scores become very large or very small. This issue arises because the exponential function $$\exp$$ grows very quickly, and when dealing with large numbers, it can result in values that are too large for the computer to handle, leading to overflow errors.

To prevent this, we apply a common technique: subtracting the maximum score from each attention score before applying the exponential function. This helps to ensure that the largest value in the attention scores becomes zero, reducing the likelihood of overflow. Here's how it's done:

$$
\text{SoftmaxScore} = \frac{\exp(\text{score} - \text{score}_{\text{max}})}{\sum\exp(\text{score} - \text{score}_{\text{max}})}
$$

Where $$\text{score}_{i,\text{max}}$$ is the maximum value of the attention scores for the \(i\)-th head. Subtracting the maximum score from each individual score ensures that the largest value becomes 0, which prevents the exponentials from becoming too large.

This subtraction does not affect the final result of the softmax calculation because the softmax is a relative function it's the ratios of the exponentials that matter. Therefore, this adjustment ensures numerical stability while maintaining the correctness of the computation.

To summarize, when computing softmax in multi-head attention:

- Subtract the maximum score from each attention score before applying the exponential function.
- This technique prevents overflow by ensuring that the largest value becomes 0, which keeps the exponential values within a manageable range.
- The relative relationships between the scores remain unchanged, so the softmax output remains correct.

By applying this numerical stability trick, the softmax function becomes more robust and prevents computational issues that could arise during training or inference, especially when dealing with large models or sequences.

Finally, the attention output for each $$\text{head}_i$$ is computed as:

$$
\text{head}_i = \text{SoftmaxScore}_i \cdot V_i
$$

#### 4. Concatenation and Linear Transformation

After computing the attention output for each head, the outputs are concatenated along the feature dimension. This results in a matrix of dimensions $$(\text{seqLen}, d_{\text{model}})$$, where the concatenated attention outputs are passed through a final linear transformation to obtain the final multi-head attention output.

$$
\text{MultiHeadOutput} = \text{concat}(\text{head}_1, \text{head}_2, \dots, \text{head}_n)
$$

The concatenated result is then linearly transformed using a weight matrix $W_{\text{o}}$ to obtain the final output. However, in our case, obtaining the multi-head attention output without this final transformation is sufficient:

$$
\text{MultiHeadOutput} = W_o \cdot \text{MultiHeadOutput}
$$

### Key Points

- Each attention head processes the input independently using its own set of learned weights. This allows each head to focus on different relationships in the data.
- Each head calculates its attention scores based on its corresponding Query, Key, and Value matrices, producing different attention outputs.
- The outputs of all attention heads are concatenated to form a unified representation. This concatenated result is then linearly transformed to generate the final output.

Multi-head attention allows the model to attend to different aspects of the input sequence in parallel, making it more capable of learning complex and diverse relationships. This parallelization of attention heads enhances the model's ability to understand the data from multiple angles simultaneously, contributing to improved performance in tasks like machine translation, text generation, and more.

### Verified Pure NumPy Solution (From Scratch)
```python
import numpy as np
from typing import Tuple, List

def compute_qkv(X: np.ndarray, W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the Query (Q), Key (K), and Value (V) matrices.
    
    Args:
    X: numpy array of shape (seq_len, d_model), input sequence
    W_q, W_k, W_v: numpy arrays of shape (d_model, d_model), weight matrices for Q, K, and V
    
    Returns:
    Q, K, V: numpy arrays of shape (seq_len, d_model)
    """
    Q = np.dot(X, W_q)  # Compute the Query matrix Q
    K = np.dot(X, W_k)  # Compute the Key matrix K
    V = np.dot(X, W_v)  # Compute the Value matrix V
    return Q, K, V

def self_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray) -> np.ndarray:
    """
    Compute self-attention for a single head.
    
    Args:
    Q: numpy array of shape (seq_len, d_k), Query matrix
    K: numpy array of shape (seq_len, d_k), Key matrix
    V: numpy array of shape (seq_len, d_k), Value matrix
    
    Returns:
    attention_output: numpy array of shape (seq_len, d_k), output of the self-attention mechanism
    """
    d_k = Q.shape[1]  # Get the dimension of the keys
    scores = np.matmul(Q, K.T) / np.sqrt(d_k)  # Compute scaled dot-product attention scores
    score_max = np.max(scores, axis=1, keepdims=True)  # Find the maximum score for numerical stability
    attention_weights = np.exp(scores - score_max) / np.sum(np.exp(scores - score_max), axis=1, keepdims=True)  # Compute softmax to get attention weights
    attention_output = np.matmul(attention_weights, V)  # Compute the final attention output
    return attention_output

def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray, n_heads: int) -> np.ndarray:
    """
    Compute multi-head attention.
    
    Args:
    Q, K, V: numpy arrays of shape (seq_len, d_model), Query, Key, and Value matrices
    n_heads: int, number of attention heads
    
    Returns:
    attention_output: numpy array of shape (seq_len, d_model), final attention output
    """
    d_model = Q.shape[1]  # Get the model dimension
    assert d_model % n_heads == 0  # Ensure d_model is divisible by n_heads
    d_k = d_model // n_heads  # Dimension for each head

    # Reshape Q, K, V to separate heads
    Q_reshaped = Q.reshape(Q.shape[0], n_heads, d_k).transpose(1, 0, 2)  # Reshape and transpose to (n_heads, seq_len, d_k)
    K_reshaped = K.reshape(K.shape[0], n_heads, d_k).transpose(1, 0, 2)  # Reshape and transpose to (n_heads, seq_len, d_k)
    V_reshaped = V.reshape(V.shape[0], n_heads, d_k).transpose(1, 0, 2)  # Reshape and transpose to (n_heads, seq_len, d_k)

    # Compute attention scores for each head
    attentions = []  # Store attention outputs for each head

    for i in range(n_heads):
        attn = self_attention(Q_reshaped[i], K_reshaped[i], V_reshaped[i])  # Compute attention for the i-th head
        attentions.append(attn)  # Collect attention output

    # Concatenate all head outputs
    attention_output = np.concatenate(attentions, axis=-1)  # Concatenate along the last axis (columns)
    return attention_output  # Return the final attention output
```

### Verified Sample Trace Data
```json
{
  "test": "m, n = 4, 4\nn_heads = 2\nnp.random.seed(42)\nX = np.arange(m*n).reshape(m,n)\nX = np.random.permutation(X.flatten()).reshape(m, n)\nW_q = np.random.randint(0,4,size=(n,n))\nW_k = np.random.randint(0,5,size=(n,n))\nW_v = np.random.randint(0,6,size=(n,n))\nQ, K, V = compute_qkv(X, W_q, W_k, W_v)\nprint(multi_head_attention(Q, K, V, n_heads))",
  "expected_output": "np.array([[103, 109, 46, 99],\n                                [103, 109, 46, 99],\n                                [103, 109, 46, 99],\n                                [103, 109, 46, 99]])"
}
```

---

## Problem 96: Implement the Hard Sigmoid Activation Function
- **Difficulty**: `easy`
- **Category**: Deep Learning

### Description & Spec
Implement the Hard Sigmoid activation function, a computationally efficient approximation of the standard sigmoid function. Your function should take a single input value and return the corresponding output based on the Hard Sigmoid definition.

### Mathematical Derivation & Intuition
## Understanding the Hard Sigmoid Activation Function

The Hard Sigmoid is a piecewise linear approximation of the sigmoid activation function. It's computationally more efficient than the standard sigmoid function while maintaining similar characteristics. This function is particularly useful in neural networks where computational efficiency is crucial.

### Mathematical Definition

The Hard Sigmoid function is mathematically defined as:

$$
HardSigmoid(x) = \begin{cases} 
0 & \text{if } x \leq -2.5 \\ 
0.2x + 0.5 & \text{if } -2.5 < x < 2.5 \\ 
1 & \text{if } x \geq 2.5 
\end{cases}
$$

Where $x$ is the input to the function.

### Characteristics

- **Output Range:** The output is always bounded in the range $[0, 1]$
- **Shape:** The function consists of three parts:
  - A constant value of 0 for inputs <= -2.5
  - A linear segment with slope 0.2 for inputs between -2.5 and 2.5
  - A constant value of 1 for inputs >= 2.5
- **Gradient:** The gradient is 0.2 in the linear region and 0 in the saturated regions

### Advantages in Neural Networks

This function is particularly useful in neural networks as it provides:
- Computational efficiency compared to standard sigmoid
- Bounded output range similar to sigmoid
- Simple gradient computation

### Verified Pure NumPy Solution (From Scratch)
```python
def hard_sigmoid(x: float) -> float:
    """
    Implements the Hard Sigmoid activation function.

    Args:
        x (float): Input value

    Returns:
        float: The Hard Sigmoid of the input
    """
    if x <= -2.5:
        return 0.0
    elif x >= 2.5:
        return 1.0
    else:
        return 0.2 * x + 0.5
```

### Verified Sample Trace Data
```json
{
  "test": "print(hard_sigmoid(.56))",
  "expected_output": "0.612"
}
```

---

## Problem 97: Implement the ELU Activation Function
- **Difficulty**: `easy`
- **Category**: Deep Learning

### Description & Spec
Implement the ELU (Exponential Linear Unit) activation function, which helps mitigate the limitations of ReLU by providing negative outputs for negative inputs. The function should compute the ELU activation value for a given input.

### Mathematical Derivation & Intuition
## Understanding the ELU Activation Function

The ELU (Exponential Linear Unit) activation function is an advanced activation function that addresses some limitations of ReLU by providing negative values for negative inputs, which can help prevent the "dying ReLU" problem and speed up learning.

### Mathematical Definition

The ELU function is mathematically defined as:

$$
ELU(x) = \begin{cases} 
x & \text{if } x > 0 \\
\alpha(e^x - 1) & \text{otherwise}
\end{cases}
$$

Where:
- $x$ is the input to the function
- $\alpha$ is a hyperparameter (typically set to 1.0) that controls the value to which an ELU saturates for negative inputs
- $e$ is the base of natural logarithms (Euler's number)

### Characteristics

- **Output Range:** The output is in the range $[-\alpha, \infty)$. For positive inputs, it behaves like the identity function, while for negative inputs, it has a smooth exponential curve that approaches -alpha.
- **Smoothness:** Unlike ReLU, ELU is smooth everywhere, including at x = 0, which can lead to faster learning.
- **Gradient:** The gradient is 1 for positive values and $\alpha e^x$ for negative values, providing non-zero gradients for negative inputs.

### Advantages

1. Reduces the vanishing gradient problem
2. Can produce negative outputs, allowing the function to push mean unit activations closer to zero
3. Smoother gradient flow compared to ReLU
4. Better handling of noise in the data due to the bounded negative part

### Visual Comparison with ReLU

While ReLU simply outputs zero for all negative inputs, ELU provides a smooth negative response:

- For x > 0: Both ReLU and ELU output x
- For x <= 0: 
  - ReLU outputs 0
  - ELU outputs $\alpha(e^x - 1)$, which smoothly approaches -alpha

ELU is particularly useful in deep neural networks where you want to maintain some of the benefits of ReLU while addressing its limitations regarding negative inputs.

### Verified Pure NumPy Solution (From Scratch)
```python
import math

def elu(x: float, alpha: float = 1.0) -> float:
    """
    Compute the ELU activation function.

    Args:
        x (float): Input value
        alpha (float): ELU parameter for negative values (default: 1.0)

    Returns:
        float: ELU activation value
    """
    return round(x if x > 0 else alpha * (math.exp(x) - 1),4)
```

### Verified Sample Trace Data
```json
{
  "test": "print(elu(0))",
  "expected_output": "0.0"
}
```

---

## Problem 98: Implement the PReLU Activation Function
- **Difficulty**: `easy`
- **Category**: Deep Learning

### Description & Spec
Implement the PReLU (Parametric ReLU) activation function, a variant of the ReLU activation function that introduces a learnable parameter for negative inputs. Your task is to compute the PReLU activation value for a given input.

### Mathematical Derivation & Intuition
### Understanding the PReLU (Parametric ReLU) Activation Function

The PReLU (Parametric Rectified Linear Unit) is an advanced variant of the ReLU activation function that introduces a learnable parameter for negative inputs. This makes it more flexible than standard ReLU and helps prevent the "dying ReLU" problem.

#### Mathematical Definition

The PReLU function is defined as:

$$
PReLU(x) = \begin{cases}
x & \text{if } x > 0 \\
\alpha x & \text{otherwise}
\end{cases}
$$

Where:
- $x$ is the input value
- $\alpha$ is a learnable parameter (typically initialized to a small value like 0.25)

#### Key Characteristics

1. **Adaptive Slope**: Unlike ReLU which has a zero slope for negative inputs, PReLU learns the optimal negative slope parameter ($\alpha$) during training.

2. **Output Range**: 
   - For $x > 0$: Output equals input ($y = x$)
   - For $x \leq 0$: Output is scaled by $\alpha$ ($y = \alpha x$)

3. **Advantages**:
   - Helps prevent the "dying ReLU" problem
   - More flexible than standard ReLU
   - Can improve model performance through learned parameter
   - Maintains the computational efficiency of ReLU

4. **Special Cases**:
   - When $\alpha = 0$, PReLU becomes ReLU
   - When $\alpha = 1$, PReLU becomes a linear function
   - When $\alpha$ is small (e.g., 0.01), PReLU behaves similarly to Leaky ReLU

PReLU is particularly useful in deep neural networks where the optimal negative slope might vary across different layers or channels.

### Verified Pure NumPy Solution (From Scratch)
```python
def prelu(x: float, alpha: float = 0.25) -> float:
    """
    Implements the PReLU (Parametric ReLU) activation function.

    Args:
        x: Input value
        alpha: Slope parameter for negative values (default: 0.25)

    Returns:
        float: PReLU activation value
    """
    return x if x > 0 else alpha * x
```

### Verified Sample Trace Data
```json
{
  "test": "print(prelu(2.0))",
  "expected_output": "2.0"
}
```

---

## Problem 99: Implement the Softplus Activation Function
- **Difficulty**: `easy`
- **Category**: Deep Learning

### Description & Spec
Implement the Softplus activation function, a smooth approximation of the ReLU function. Your task is to compute the Softplus value for a given input, handling edge cases to prevent numerical overflow or underflow.

### Mathematical Derivation & Intuition
### Understanding the Softplus Activation Function

The Softplus activation function is a smooth approximation of the ReLU function. It's used in neural networks where a smoother transition around zero is desired. Unlike ReLU which has a sharp transition at x=0, Softplus provides a more gradual change.

### Mathematical Definition

The Softplus function is mathematically defined as:

$$
Softplus(x) = \log(1 + e^x)
$$

Where:
- $x$ is the input to the function
- $e$ is Euler's number (approximately 2.71828)
- $\log$ is the natural logarithm

### Characteristics

1. **Output Range**: 
   - The output is always positive: $(0, \infty)$
   - Unlike ReLU, Softplus never outputs exactly zero

2. **Smoothness**:
   - Softplus is continuously differentiable
   - The transition around x=0 is smooth, unlike ReLU's sharp "elbow"

3. **Relationship to ReLU**:
   - Softplus can be seen as a smooth approximation of ReLU
   - As x becomes very negative, Softplus approaches 0
   - As x becomes very positive, Softplus approaches x

4. **Derivative**:
   - The derivative of Softplus is the logistic sigmoid function:
   $$
   \frac{d}{dx}Softplus(x) = \frac{1}{1 + e^{-x}}
   $$

### Use Cases
- When smooth gradients are important for optimization
- In neural networks where a continuous approximation of ReLU is needed
- Situations where strictly positive outputs are required with smooth transitions

### Verified Pure NumPy Solution (From Scratch)
```python
import math

def softplus(x: float) -> float:
    """
    Compute the softplus activation function.

    Args:
        x: Input value

    Returns:
        The softplus value: log(1 + e^x)
    """
    # To prevent overflow for large positive values
    if x > 100:
        return x
    # To prevent underflow for large negative values
    if x < -100:
        return 0.0

    return round (math.log(1.0 + math.exp(x)),4)
```

### Verified Sample Trace Data
```json
{
  "test": "print(softplus(0))",
  "expected_output": "0.6931"
}
```

---

## Problem 100: Implement the Softsign Activation Function
- **Difficulty**: `easy`
- **Category**: Deep Learning

### Description & Spec
Implement the Softsign activation function, a smooth activation function used in neural networks. Your task is to compute the Softsign value for a given input, ensuring the output is bounded between -1 and 1.

### Mathematical Derivation & Intuition
## Understanding the Softsign Activation Function

The Softsign activation function is a smooth, non-linear activation function used in neural networks. It’s similar to the hyperbolic tangent (tanh) function but with different properties, particularly in its tails which approach their limits more slowly.

### Mathematical Definition

The Softsign function is mathematically defined as:

$$
Softsign(x) = \frac{x}{1 + |x|}
$$

Where:
- $x$ is the input to the function
- $|x|$ represents the absolute value of $x$

### Characteristics

- **Output Range:** The output is bounded between -1 and 1, approaching these values asymptotically as $x$ approaches $\pm \infty$.
- **Shape:** The function has an S-shaped curve, similar to tanh but with a smoother approach to its asymptotes.
- **Gradient:** The gradient is smoother and more gradual compared to tanh, which can help prevent vanishing gradient problems in deep networks.
- **Symmetry:** The function is symmetric around the origin $(0,0)$.

### Key Properties

- **Bounded Output:** Unlike ReLU, Softsign naturally bounds its output between -1 and 1.
- **Smoothness:** The function is continuous and differentiable everywhere.
- **No Saturation:** The gradients approach zero more slowly than in tanh or sigmoid functions.
- **Zero-Centered:** The function crosses through the origin, making it naturally zero-centered.

This activation function can be particularly useful in scenarios where you need bounded outputs with more gradual saturation compared to tanh or sigmoid functions.

### Verified Pure NumPy Solution (From Scratch)
```python
def softsign(x: float) -> float:
    """
    Implements the Softsign activation function.

    Args:
        x (float): Input value

    Returns:
        float: The Softsign of the input, calculated as x/(1 + |x|)
    """
    return round(x / (1 + abs(x)), 4)
```

### Verified Sample Trace Data
```json
{
  "test": "print(softsign(0))",
  "expected_output": "0.0"
}
```

---

## Problem 102: Implement the Swish Activation Function
- **Difficulty**: `easy`
- **Category**: Deep Learning

### Description & Spec
Implement the Swish activation function, a self-gated activation function that has shown superior performance in deep neural networks compared to ReLU. Your task is to compute the Swish value for a given input.

### Mathematical Derivation & Intuition
## Understanding the Swish Activation Function

The Swish activation function is a modern self-gated activation function introduced by researchers at Google Brain. It has been shown to perform better than ReLU in many deep networks, particularly in deeper architectures.

### Mathematical Definition

The Swish function is defined as:

$$Swish(x) = x \times \sigma(x)$$

where $\sigma(x)$ is the sigmoid function defined as:

$$\sigma(x) = \frac{1}{1 + e^{-x}}$$

### Characteristics

- **Output Range**: Unlike ReLU which has a range of $[0, \infty)$, Swish has a range of $(-\infty, \infty)$
- **Smoothness**: Swish is smooth and non-monotonic, making it differentiable everywhere
- **Shape**: The function has a slight dip below 0 for negative values, then curves up smoothly for positive values
- **Properties**:
  - For large positive x: Swish(x) ~ x (similar to linear function)
  - For large negative x: Swish(x) ~ 0 (similar to ReLU)
  - Has a minimal value around x ~ -1.28

### Advantages

- Smooth function with no hard zero threshold like ReLU
- Self-gated nature allows for more complex relationships
- Often provides better performance in deep neural networks
- Reduces the vanishing gradient problem compared to sigmoid

### Verified Pure NumPy Solution (From Scratch)
```python
import math

def swish(x: float) -> float:
    """
    Implements the Swish activation function.

    Args:
        x: Input value

    Returns:
        The Swish activation value
    """
    return x * (1 / (1 + math.exp(-x)))
```

### Verified Sample Trace Data
```json
{
  "test": "print(round(swish(0),4))",
  "expected_output": "0.0"
}
```

---

## Problem 103: Implement the SELU Activation Function
- **Difficulty**: `easy`
- **Category**: Deep Learning

### Description & Spec
Implement the SELU (Scaled Exponential Linear Unit) activation function, a self-normalizing variant of ELU. Your task is to compute the SELU value for a given input while ensuring numerical stability.

### Mathematical Derivation & Intuition
## Understanding the SELU Activation Function

The SELU (Scaled Exponential Linear Unit) activation function is a self-normalizing variant of the ELU activation function, introduced in 2017. It's particularly useful in deep neural networks as it automatically ensures normalized outputs with zero mean and unit variance.

### Mathematical Definition

The SELU function is defined as:

$$
SELU(x) = \lambda \begin{cases} 
x & \text{if } x > 0 \\
\alpha(e^x - 1) & \text{if } x \leq 0
\end{cases}
$$

Where:
- $\lambda \approx 1.0507$ is the scale parameter
- $\alpha \approx 1.6733$ is the alpha parameter

### Characteristics

- **Output Range:** The function maps inputs to $(-\lambda\alpha, \infty)$
- **Self-Normalizing:** Automatically maintains mean close to 0 and variance close to 1
- **Continuous:** The function is continuous and differentiable everywhere
- **Non-Linear:** Provides non-linearity while preserving gradients for negative values
- **Parameters:** Uses carefully chosen values for $\lambda$ and $\alpha$ to ensure self-normalization

### Advantages

1. **Self-Normalization:** Eliminates the need for batch normalization in many cases
2. **Robust Learning:** Helps prevent vanishing and exploding gradients
3. **Better Performance:** Often leads to faster training in deep neural networks
4. **Internal Normalization:** Maintains normalized activations throughout the network

### Use Cases

SELU is particularly effective in:
- Deep neural networks where maintaining normalized activations is crucial
- Networks that require self-normalizing properties
- Scenarios where batch normalization might be problematic or expensive

### Verified Pure NumPy Solution (From Scratch)
```python
import math

def selu(x: float) -> float:
    """
    Implements the SELU (Scaled Exponential Linear Unit) activation function.

    Args:
        x: Input value

    Returns:
        SELU activation value
    """
    alpha = 1.6732632423543772
    scale = 1.0507009873554804
    return round(scale * x if x > 0 else scale * alpha * (math.exp(x) - 1), 4)
```

### Verified Sample Trace Data
```json
{
  "test": "print(round(selu(1.0), 4))",
  "expected_output": "1.0507"
}
```

---

## Problem 107: Implement Masked Self-Attention
- **Difficulty**: `medium`
- **Category**: Deep Learning

### Description & Spec
Implement masked self-attention, a variation of the attention mechanism used in sequence modeling tasks such as text generation. Your task is to compute masked self-attention using query (Q), key (K), value (V) matrices and an attention mask.

### Mathematical Derivation & Intuition
## Understanding Masked Attention

Masked attention is a variation of the attention mechanism used primarily in sequence modeling tasks, such as language modeling and text generation. The key idea behind masked attention is to control the flow of information by selectively masking certain elements in the input sequence. This ensures that the model attends only to valid positions when computing attention scores.

Masked attention is particularly useful in autoregressive tasks where future information should not influence the current prediction. By masking out future tokens, the model is constrained to attend only to preceding tokens or the current token, preserving causality during training and inference.

### Concepts

The attention mechanism enables the model to weigh the importance of different elements in the input sequence based on their relevance to a specific task. Masked attention modifies this process by incorporating a mask, which defines which elements the model is allowed to attend to. This ensures that the attention mechanism respects temporal or structural constraints, such as the directionality of time in sequence data.

The process of masked attention involves the following steps:

1. **Computing Attention Scores:** The model calculates how much focus each element in the sequence should receive based on its relationship with other elements.
2. **Applying the Mask:** A mask is applied to restrict attention to specific positions in the sequence. Elements outside the allowed range are effectively ignored.
3. **Normalizing Scores:** The masked scores are transformed into probabilities using the softmax function.
4. **Computing the Output:** The final output is computed as a weighted sum of the input values, with weights determined by the normalized attention scores.

### Structure of Masked Attention

The attention mechanism can be described using Query (Q), Key (K), and Value (V) matrices. In masked attention, these matrices interact with an additional mask to determine the attention distribution.

#### 1. Query, Key, and Value Matrices

- **Query (Q):** Represents the current element for which the model is computing attention.
- **Key (K):** Encodes information about all elements in the sequence.
- **Value (V):** Contains the representations that will be aggregated into the output.

Assume that the input sequence has a length of $\text{seqLen}$ and the model dimension is $d_{\text{model}}$. The dimensions of the Q, K, and V matrices are:

- Query (Q): $(\text{seqLen}, d_{\text{model}})$
- Key (K): $(\text{seqLen}, d_{\text{model}})$
- Value (V): $(\text{seqLen}, d_{\text{model}})$

#### 2. Computing Attention Scores

The raw attention scores are computed as the scaled dot product between the Query (Q) and Key (K) matrices:

$$
\text{score} = \frac{QK^T}{\sqrt{d_k}}
$$

Where $d_k$ is the dimensionality of the key space. The scaling factor $\frac{1}{\sqrt{d_k}}$ ensures that the dot product values do not grow excessively large, preventing instability in the softmax function.

#### 3. Applying the Mask

The mask is used to control which elements the model is allowed to attend to. Typically, the mask is a binary matrix of dimensions $(\text{seqLen}, \text{seqLen})$, where:

- A value of 0 indicates that attention is allowed.
- A value of $-\infty$ (or a very large negative value like $-1e9$) indicates that attention is prohibited.

The raw attention scores are modified by adding the mask:

$$
\text{maskedScore} = \text{score} + \text{mask}
$$

This ensures that prohibited positions receive attention scores that are effectively $-\infty$, making their softmax probabilities zero.

#### 4. Softmax Calculation

The softmax function is applied to the masked scores to compute attention weights. To ensure numerical stability, the maximum score in each row is subtracted before applying the softmax function:

$$
\text{SoftmaxScore} = \frac{\exp(\text{maskedScore} - \text{maskedScore}_{\text{max}})}{\sum\exp(\text{maskedScore} - \text{maskedScore}_{\text{max}})}
$$

#### 5. Computing the Output

The final attention output is computed as a weighted sum of the Value (V) matrix, with weights determined by the attention scores:

$$
\text{output} = \text{SoftmaxScore} \cdot V
$$

### Key Points

- **Masking Future Tokens:** In autoregressive tasks, a triangular mask is used to prevent the model from attending to future positions. For a sequence of length $n$, the mask is an upper triangular matrix with 0s in the lower triangle and $-\infty$ in the upper triangle.

  Example:
  $$
  \text{mask} = \begin{bmatrix}
  0 & -\infty & -\infty \\
  0 & 0 & -\infty \\
  0 & 0 & 0
  \end{bmatrix}
  $$

- **Numerical Stability:** Subtracting the maximum score before applying softmax ensures numerical stability and prevents overflow or underflow errors.
- **Flexibility:** The mask can be customized to handle other constraints, such as ignoring padding tokens in variable-length sequences.

By selectively controlling the flow of information through masking, masked attention ensures that the model respects temporal or structural constraints, enabling it to generate coherent and contextually accurate outputs in sequence modeling tasks.

### Verified Pure NumPy Solution (From Scratch)
```python
import numpy as np

def compute_qkv(X: np.ndarray, W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray):
    Q = np.dot(X, W_q)
    K = np.dot(X, W_k)
    V = np.dot(X, W_v)
    return Q, K, V

def masked_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray, mask: np.ndarray) -> np.ndarray:
    d_k = Q.shape[1]
    scores = np.matmul(Q, K.T) / np.sqrt(d_k)
    scores = scores + mask  # Apply mask
    attention_weights = np.exp(scores - np.max(scores, axis=1, keepdims=True))
    attention_weights = attention_weights / np.sum(attention_weights, axis=1, keepdims=True)
    return np.matmul(attention_weights, V)
```

### Verified Sample Trace Data
```json
{
  "test": "np.random.seed(42)\nX = np.arange(48).reshape(6,8)\nX = np.random.permutation(X.flatten()).reshape(6, 8)\nmask = np.triu(np.ones((6, 6))*(-np.inf), k=1)\nW_q = np.random.randint(0,4,size=(8,8))\nW_k = np.random.randint(0,5,size=(8,8))\nW_v = np.random.randint(0,6,size=(8,8))\nQ, K, V = compute_qkv(X, W_q, W_k, W_v)\nprint(masked_attention(Q, K, V, mask))",
  "expected_output": "[[547. 490. 399. 495. 485. 439. 645. 393.]\n [547. 490. 399. 495. 485. 439. 645. 393.]\n [471. 472. 429. 538. 377. 450. 531. 362.]\n [471. 472. 429. 538. 377. 450. 531. 362.]\n [471. 472. 429. 538. 377. 450. 531. 362.]\n [471. 472. 429. 538. 377. 450. 531. 362.]]"
}
```

---

## Problem 113: Implement a Simple Residual Block with Shortcut Connection
- **Difficulty**: `easy`
- **Category**: Deep Learning

### Description & Spec
Implement a function that creates a simple residual block using NumPy. The block should take a 1D input array, process it through two weight layers (using matrix multiplication), apply ReLU activations, and add the original input via a shortcut connection before a final ReLU activation.

### Mathematical Derivation & Intuition
## Understanding Residual Blocks in ResNet

Residual blocks are the cornerstone of ResNet (Residual Network), a deep learning architecture designed to train very deep neural networks by addressing issues like vanishing gradients. The key idea is to allow the network to learn residuals differences between the input and the desired output rather than the full transformation.

### Core Concept: Residual Learning
In a traditional neural network layer, the output is a direct transformation of the input, such as $H(x)$, where $x$ is the input. In a residual block, instead of learning $H(x)$ directly, the network learns the residual $F(x) = H(x) - x$. The output of the block is then:

$$
y = F(x) + x
$$

Here, $F(x)$ represents the transformation applied by the layers within the block (e.g., weight layers and activations), and $x$ is the input, added back via a shortcut connection. This structure allows the network to learn an identity function ($F(x) = 0$, so $y = x$) if needed, which helps in training deeper networks.

### Mathematical Structure
A typical residual block involves two weight layers with an activation function between them. The activation function used in ResNet is ReLU, defined as:

$$
\text{ReLU}(z) = \max(0, z)
$$

The block takes an input $x$, applies a transformation $F(x)$ through the weight layers and activations, and then adds the input $x$ back. Mathematically, if the weight layers are represented by matrices $W_1$ and $W_2$, the transformation $F(x)$ might look like a composition of operations involving $W_1 \cdot x$, a ReLU activation, and $W_2$ applied to the result. The final output $y$ is the sum of $F(x)$ and $x$, often followed by another ReLU activation to ensure non-negativity.

### Why Shortcut Connections?
- **Ease of Learning**: If the optimal transformation is close to an identity function, the block can learn $F(x) \approx 0$, making $y \approx x$.
- **Gradient Flow**: The shortcut connection allows gradients to flow directly through the addition operation during backpropagation, helping to train deeper networks without vanishing gradients.

### Conceptual Example
Suppose the input $x$ is a vector of length 2, and the weight layers are matrices $W_1$ and $W_2$. The block computes $F(x)$ by applying $W_1$, a ReLU activation, and $W_2$, then adds $x$ to the result. The shortcut connection ensures that even if $F(x)$ is small, the output $y$ retains information from $x$, making it easier for the network to learn.

This structure is what enables ResNet to scale to hundreds of layers while maintaining performance, as shown in the diagram of the residual block.

### Verified Pure NumPy Solution (From Scratch)
```python
import numpy as np

def residual_block(x: np.ndarray, w1: np.ndarray, w2: np.ndarray) -> np.ndarray:
    # First weight layer
    y = np.dot(w1, x)
    # First ReLU
    y = np.maximum(0, y)
    # Second weight layer
    y = np.dot(w2, y)
    # Add shortcut connection (x + F(x))
    y = y + x
    # Final ReLU
    y = np.maximum(0, y)
    return y
```

### Verified Sample Trace Data
```json
{
  "test": "x = np.array([1.0, 2.0])\nw1 = np.array([[1.0, 0.0], [0.0, 1.0]])\nw2 = np.array([[0.5, 0.0], [0.0, 0.5]])\nprint(residual_block(x, w1, w2))",
  "expected_output": "[1.5,3.]"
}
```

---

## Problem 114: Implement Global Average Pooling
- **Difficulty**: `easy`
- **Category**: Deep Learning

### Description & Spec
Implement a function that performs Global Average Pooling on a 3D NumPy array representing feature maps from a convolutional layer. The function should take an input of shape (height, width, channels) and return a 1D array of shape (channels,), where each element is the average of all values in the corresponding feature map.

### Mathematical Derivation & Intuition
## Understanding Global Average Pooling

Global Average Pooling (GAP) is a pooling operation commonly used in convolutional neural networks (CNNs) to reduce the spatial dimensions of feature maps. Unlike traditional pooling methods like max pooling or average pooling with a fixed window size, GAP computes the average of each entire feature map, resulting in a single value per channel.

### How It Works

Given a 3D input tensor of shape $(H, W, C)$, where:
- $H$ is the height,
- $W$ is the width,
- $C$ is the number of channels (feature maps),

Global Average Pooling produces a 1D output vector of shape $(C,)$, where each element is the average of all values in the corresponding feature map.

Mathematically, for each channel $c$:

$$
\text{GAP}(x)_c = \frac{1}{H \times W} \sum_{i=1}^{H} \sum_{j=1}^{W} x_{i,j,c}
$$

### Benefits of Global Average Pooling

- **Parameter Reduction**: By replacing fully connected layers with GAP, the number of parameters is significantly reduced, which helps in preventing overfitting.
- **Spatial Invariance**: GAP captures the global information from each feature map, making the model more robust to spatial translations.
- **Simplicity**: It is a straightforward operation that doesn't require tuning hyperparameters like pooling window size or stride.

### Use in Modern Architectures

Global Average Pooling is a key component in architectures like ResNet, where it is used before the final classification layer. It allows the network to handle inputs of varying sizes, as the output depends only on the number of channels, not the spatial dimensions.

### Example

Consider a 2x2x3 input tensor:

$$
x = \begin{bmatrix}
\begin{bmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \end{bmatrix},
\begin{bmatrix} 7 & 8 & 9 \\ 10 & 11 & 12 \end{bmatrix}
\end{bmatrix}
$$

Applying GAP:

- For channel 0: $\frac{1+4+7+10}{4} = \frac{22}{4} = 5.5$
- For channel 1: $\frac{2+5+8+11}{4} = \frac{26}{4} = 6.5$
- For channel 2: $\frac{3+6+9+12}{4} = \frac{30}{4} = 7.5$

Thus, the output is $[5.5, 6.5, 7.5]$.

This operation effectively summarizes each feature map into a single value, capturing the essence of the features learned by the network.

### Verified Pure NumPy Solution (From Scratch)
```python
import numpy as np

def global_avg_pool(x: np.ndarray) -> np.ndarray:
    return np.mean(x, axis=(0, 1))
```

### Verified Sample Trace Data
```json
{
  "test": "x = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])\nprint(global_avg_pool(x))",
  "expected_output": "[5.5,6.5,7.5]"
}
```

---

## Problem 115: Implement Batch Normalization for BCHW Input
- **Difficulty**: `easy`
- **Category**: Deep Learning

### Description & Spec
Implement a function that performs Batch Normalization on a 4D NumPy array representing a batch of feature maps in the BCHW format (batch, channels, height, width). The function should normalize the input across the batch and spatial dimensions for each channel, then apply scale (gamma) and shift (beta) parameters. Use the provided epsilon value to ensure numerical stability.

### Mathematical Derivation & Intuition
## Understanding Batch Normalization

Batch Normalization (BN) is a widely used technique that helps to accelerate the training of deep neural networks and improve model performance. By normalizing the inputs to each layer so that they have a mean of zero and a variance of one, BN stabilizes the learning process, speeds up convergence, and introduces regularization, which can reduce the need for other forms of regularization like dropout.

### Concepts

Batch Normalization operates on the principle of reducing **internal covariate shift**, which occurs when the distribution of inputs to a layer changes during training as the model weights get updated. This can slow down training and make hyperparameter tuning more challenging. By normalizing the inputs, BN reduces this problem, allowing the model to train faster and more reliably.

The process of Batch Normalization consists of the following steps:

1. **Compute the Mean and Variance:** For each mini-batch, compute the mean and variance of the activations for each feature (dimension).
2. **Normalize the Inputs:** Normalize the activations using the computed mean and variance.
3. **Apply Scale and Shift:** After normalization, apply a learned scale (gamma) and shift (beta) to restore the model's ability to represent the data's original distribution.
4. **Training and Inference:** During training, the mean and variance are computed from the current mini-batch. During inference, a running average of the statistics from the training phase is used.

### Structure of Batch Normalization for BCHW Input

For an input tensor with the shape **BCHW**, where:
- **B**: batch size,
- **C**: number of channels,
- **H**: height,
- **W**: width,
the Batch Normalization process operates on specific dimensions based on the task's requirement.

#### 1. Mean and Variance Calculation

- In **Batch Normalization**, we typically normalize the activations **across the batch** and **over the spatial dimensions (height and width)** for each **channel**. This means we calculate the mean and variance **per channel** (C) for the **batch and spatial dimensions** (H, W).

For each channel $c$, we compute the **mean** $\mu_c$ and **variance** $\sigma_c^2$ over the mini-batch and spatial dimensions:

$$
\mu_c = \frac{1}{B \cdot H \cdot W} \sum_{i=1}^{B} \sum_{h=1}^{H} \sum_{w=1}^{W} x_{i,c,h,w}
$$

$$
\sigma_c^2 = \frac{1}{B \cdot H \cdot W} \sum_{i=1}^{B} \sum_{h=1}^{H} \sum_{w=1}^{W} (x_{i,c,h,w} - \mu_c)^2
$$

Where:
- $x_{i,c,h,w}$ is the input activation at batch index $i$, channel $c$, height $h$, and width $w$.
- $B$ is the batch size.
- $H$ and $W$ are the spatial dimensions (height and width).
- $C$ is the number of channels.

The mean and variance are computed **over all spatial positions (H, W)** and **across all samples in the batch (B)** for each **channel (C)**.

#### 2. Normalization

Once the mean $\mu_c$ and variance $\sigma_c^2$ have been computed for each channel, the next step is to **normalize** the input. The normalization is done by subtracting the mean and dividing by the standard deviation (plus a small constant $\epsilon$ for numerical stability):

$$
\hat{x}_{i,c,h,w} = \frac{x_{i,c,h,w} - \mu_c}{\sqrt{\sigma_c^2 + \epsilon}}
$$

Where:
- $\hat{x}_{i,c,h,w}$ is the normalized activation for the input at batch index $i$, channel $c$, height $h$, and width $w$.
- $\epsilon$ is a small constant to avoid division by zero (for numerical stability).

#### 3. Scale and Shift

After normalization, the next step is to apply a **scale** ($\gamma_c$) and **shift** ($\beta_c$) to the normalized activations for each channel. These learned parameters allow the model to adjust the output distribution of each feature, preserving the flexibility of the original activations.

$$
y_{i,c,h,w} = \gamma_c \hat{x}_{i,c,h,w} + \beta_c
$$

Where:
- $\gamma_c$ is the scaling factor for channel $c$.
- $\beta_c$ is the shifting factor for channel $c$.

#### 4. Training and Inference

- **During Training**: The mean and variance are computed for each mini-batch and used for normalization across the batch and spatial dimensions for each channel.
- **During Inference**: The model uses a running average of the statistics (mean and variance) that were computed during training to ensure consistent behavior when deployed.

### Key Points

- **Normalization Across Batch and Spatial Dimensions**: In Batch Normalization for **BCHW** input, the normalization is done **across the batch (B) and spatial dimensions (H, W)** for each **channel (C)**. This ensures that each feature channel has zero mean and unit variance, making the training process more stable.

- **Channel-wise Normalization**: Batch Normalization normalizes the activations independently for each **channel (C)** because different channels in convolutional layers often have different distributions and should be treated separately.

- **Numerical Stability**: The small constant $\epsilon$ is added to the variance to avoid numerical instability when dividing by the square root of variance, especially when the variance is very small.

- **Improved Gradient Flow**: By reducing internal covariate shift, Batch Normalization allows the gradients to flow more easily during backpropagation, helping the model train faster and converge more reliably.

- **Regularization Effect**: Batch Normalization introduces noise into the training process because it relies on the statistics of a mini-batch. This noise acts as a form of regularization, which can prevent overfitting and improve generalization.

### Why Normalize Over Batch and Spatial Dimensions?

- **Across Batch**: Normalizing across the batch helps to stabilize the input distribution across all samples in a mini-batch. This allows the model to avoid the problem of large fluctuations in the input distribution as weights are updated.

- **Across Spatial Dimensions**: In convolutional networks, the spatial dimensions (height and width) are highly correlated, and normalizing over these dimensions ensures that the activations are distributed consistently throughout the spatial field, helping to maintain a stable learning process.

- **Channel-wise Normalization**: Each channel can have its own distribution of values, and normalization per channel ensures that each feature map is scaled and shifted independently, allowing the model to learn representations that are not overly sensitive to specific channels' scaling.

By normalizing across the batch and spatial dimensions and applying a per-channel transformation, Batch Normalization helps reduce internal covariate shift and speeds up training, leading to faster convergence and better overall model performance.

### Verified Pure NumPy Solution (From Scratch)
```python
import numpy as np

def batch_normalization(X: np.ndarray, gamma: np.ndarray, beta: np.ndarray, epsilon: float = 1e-5) -> np.ndarray:
    # Compute mean and variance across the batch and spatial dimensions
    mean = np.mean(X, axis=(0, 2, 3), keepdims=True)  # Mean over (B, H, W)
    variance = np.var(X, axis=(0, 2, 3), keepdims=True)  # Variance over (B, H, W)
    # Normalize
    X_norm = (X - mean) / np.sqrt(variance + epsilon)
    # Scale and shift
    norm_X = gamma * X_norm + beta
    return norm_X
```

### Verified Sample Trace Data
```json
{
  "test": "B, C, H, W = 2, 2, 2, 2\nnp.random.seed(42)\nX = np.random.randn(B, C, H, W)\ngamma = np.ones(C).reshape(1, C, 1, 1)\nbeta = np.zeros(C).reshape(1, C, 1, 1)\nactual_output = batch_normalization(X, gamma, beta)\nexpected_output = np.array([[[[ 0.42859934, -0.51776438], [ 0.65360963,  1.95820707]], [[ 0.02353721,  0.02355215], [ 1.67355207,  0.93490043]]], [[[-1.01139563,  0.49692747], [-1.00236882, -1.00581468]], [[ 0.45676349, -1.50433085], [-1.33293647, -0.27503802]]]])\nprint(actual_output)",
  "expected_output": "[[[[ 0.42859934, -0.51776438], [ 0.65360963,1.95820707]],[[ 0.02353721,  0.02355215], [ 1.67355207,  0.93490043]]], [[[-1.01139563,  0.49692747], [-1.00236882, -1.00581468]], [[ 0.45676349, -1.50433085], [-1.33293647, -0.27503802]]]]"
}
```

---

## Problem 123: Calculate Computational Efficiency of MoE
- **Difficulty**: `easy`
- **Category**: Deep Learning

### Description & Spec
Calculate the computational cost savings of an MoE layer compared to a dense layer, as discussed in the paper 'Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer.' Given the number of experts, sparsity (number of active experts), and input/output dimensions, compute the floating-point operations (FLOPs) for both and determine the savings percentage.

### Mathematical Derivation & Intuition
## Understanding MoE Efficiency

The paper *"Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer"* introduces the idea of activating only a few expert networks per input to drastically reduce computation. This is known as **conditional computation**, and it allows models to scale to billions of parameters without significantly increasing cost.

---

### Key Idea

In a **dense layer**, every input goes through the full set of parameters.  
In a **Mixture-of-Experts (MoE)** layer, only $k$ out of $n$ experts are active for each input.

---

### FLOPs Formulas

Let:
- $d_{in}$ = input dimension  
- $d_{out}$ = output dimension  
- $n$ = total experts  
- $k$ = active experts per input  

Then:
- **Dense layer FLOPs**:  
  $$
  \text{FLOPs}_{\text{dense}} = n \cdot d_{in} \cdot d_{out}
  $$
- **MoE layer FLOPs**:  
  $$
  \text{FLOPs}_{\text{moe}} = k \cdot d_{in} \cdot d_{out}
  $$
- **Efficiency gain**:
  $$
  \text{Savings}(\%) = \left( \frac{\text{FLOPs}_{\text{dense}} - \text{FLOPs}_{\text{moe}}}{\text{FLOPs}_{\text{dense}}} \right) \cdot 100
  $$

---

### Example

Suppose:
- $n = 1000$, $k = 2$  
- $d_{in} = d_{out} = 512$  

Then:
- MoE FLOPs = $2 \cdot 512 \cdot 512 = 524,\!288$  
- Full dense (all 1000 experts): $1000 \cdot 512 \cdot 512 = 262,\!144,\!000$  
- Savings:
  $$
  \left( \frac{262,\!144,\!000 - 524,\!288}{262,\!144,\!000} \right) \cdot 100 \approx 99.8\%
  $$

This means the MoE layer uses just **0.2%** of the computation compared to a full dense version — an enormous gain in efficiency.

---

### Summary

By activating only a small number of experts per input, MoE layers reduce computation while maintaining high model capacity. This makes it feasible to train outrageously large models efficiently.

### Verified Pure NumPy Solution (From Scratch)
```python
def compute_efficiency(n_experts, k_active, d_in, d_out):
    dense_flops = n_experts * d_in * d_out
    moe_flops = k_active * d_in * d_out
    savings = (dense_flops - moe_flops) / dense_flops * 100
    return round(savings, 1)
```

### Verified Sample Trace Data
```json
{
  "test": "print(round(compute_efficiency(1000, 2, 512, 512),1))",
  "expected_output": "99.8"
}
```

---

## Problem 124: Implement the Noisy Top-K Gating Function
- **Difficulty**: `medium`
- **Category**: Deep Learning

### Description & Spec
Implement the Noisy Top-K gating mechanism used in Mixture-of-Experts (MoE) models. Given an input matrix, weight matrices, pre-sampled noise, and a sparsity constraint k, compute the final gating probabilities matrix.

### Mathematical Derivation & Intuition
## Noisy Top-K Gating

Noisy Top-K Gating is a sparse selection mechanism used in Mixture-of-Experts (MoE) models. It routes input tokens to a subset of available experts, enhancing efficiency and model capacity.

### Overview

The core idea is to add learned noise to the gating logits and then select only the top-k experts for each input. This encourages exploration and helps balance load across experts.

### Step-by-Step Breakdown

1. **Compute Raw Gate Scores**  
   First, compute two linear projections of the input:
   $$
   H_{\text{base}} = X W_g
   $$
   $$
   H_{\text{noise}} = X W_{\text{noise}}
   $$

2. **Apply Noise with Softplus Scaling**  
   Add pre-sampled Gaussian noise, scaled by a softplus transformation:
   $$
   H = H_{\text{base}} + N \odot \text{Softplus}(H_{\text{noise}})
   $$

3. **Top-K Masking**  
   Keep only the top-k elements in each row (i.e., per input), setting the rest to $-\infty$:
   $$
   H' = \text{TopK}(H, k)
   $$

4. **Softmax Over Top-K**  
   Normalize the top-k scores into a valid probability distribution:
   $$
   G = \text{Softmax}(H')
   $$

### Worked Example

Let:
- $X = [[1.0, 2.0]]$
- $W_g = [[1.0, 0.0], [0.0, 1.0]]$
- $W_{\text{noise}} = [[0.5, 0.5], [0.5, 0.5]]$
- $N = [[1.0, -1.0]]$
- $k = 2$

Step-by-step:
- $H_{\text{base}} = [1.0, 2.0]$
- $H_{\text{noise}} = [1.5, 1.5]$
- $\text{Softplus}(H_{\text{noise}}) \approx [1.804, 1.804]$
- $H = [1.0 + 1.804, 2.0 - 1.804] = [2.804, 0.196]$
- Softmax over these gives: $[0.917, 0.0825]$

### Benefits

- **Computational Efficiency**: Activates only k experts per input.
- **Load Balancing**: Injected noise encourages diversity in expert selection.
- **Improved Generalization**: Acts as a regularizer via noise-based gating.

This technique is used in large sparse models like GShard and Switch Transformers.

### Verified Pure NumPy Solution (From Scratch)
```python
import numpy as np

def noisy_topk_gating(
    X: np.ndarray,
    W_g: np.ndarray,
    W_noise: np.ndarray,
    N: np.ndarray,
    k: int
) -> np.ndarray:
    H_base = X @ W_g
    H_noise = X @ W_noise
    softplus = np.log1p(np.exp(H_noise))
    H = H_base + N * softplus

    def top_k_masked(row, k):
        mask = np.full_like(row, -np.inf)
        top_idx = np.argsort(row)[-k:]
        mask[top_idx] = row[top_idx]
        return mask

    masked_H = np.vstack([top_k_masked(row, k) for row in H])
    exps = np.exp(masked_H - np.max(masked_H, axis=1, keepdims=True))
    return exps / np.sum(exps, axis=1, keepdims=True)
```

### Verified Sample Trace Data
```json
{
  "test": "import numpy as np\nnp.random.seed(0)\nX = np.array([[1.0, 2.0]])\nW_g = np.array([[1.0, 0.0], [0.0, 1.0]])\nW_noise = np.zeros((2,2))\nN = np.zeros((1,2))\nprint(noisy_topk_gating(X, W_g, W_noise, N, k=1))",
  "expected_output": "[[0. 1.]]"
}
```

---

## Problem 125: Implement a Sparse Mixture of Experts Layer
- **Difficulty**: `hard`
- **Category**: Deep Learning

### Description & Spec
Implement a Mixture-of-Experts (MoE) layer using softmax gating and top-k routing. Given an input tensor, a set of expert weight matrices, a gating weight matrix, and parameters specifying the number of experts and the value of k, compute the final MoE output by selecting the top-k experts per token, applying their transformations, and aggregating the results weighted by the normalized gating probabilities.

### Mathematical Derivation & Intuition
## Mixture of Experts Layer

Mixture-of-Experts layers route each token through a small subset of expert networks, reducing computation while retaining flexibility.

### 1. Gating with Softmax  
- **Logits**: For each token $t$, compute a vector of gating scores $g_t \in \mathbb{R}^E$, where $E$ is the number of experts.  
- **Softmax**: Convert scores into a probability distribution  
  $$
  \alpha_{t,j}
    = \frac{\exp\bigl(g_{t,j} - \max_j g_{t,j}\bigr)}
           {\sum_{j'=1}^{E}\exp\bigl(g_{t,j'} - \max_j g_{t,j'}\bigr)}.
  $$

### 2. Top-$k$ Selection  
- **Sparsity**: Keep only the $k$ largest weights per token, zeroing out the rest.  
- **Renormalize**: For token $t$, let $\mathcal{K}_t$ be the indices of the top $k$ experts. Then  
  $$
  \tilde\alpha_{t,j} =
    \begin{cases}
      \displaystyle\frac{\alpha_{t,j}}{\sum_{i \in \mathcal{K}_t}\alpha_{t,i}}
        & j \in \mathcal{K}_t,\\[8pt]
      0
        & \text{otherwise.}
    \end{cases}
  $$

### 3. Expert Computation  
Each expert $i$ applies its own linear transform to the token embedding $x_t$:  
$$
O_t^{(i)} = x_t\,W_e^{(i)},
$$  
where $W_e^{(i)}$ is the expert's $d \times d$ weight matrix.

### 4. Weighted Aggregation  
Combine the selected experts' outputs for each token:  
$$
y_t = \sum_{i=1}^{E} \tilde\alpha_{t,i}\,O_t^{(i)}.
$$  
The result $y_t$ lives in the original embedding space $\mathbb{R}^d$.

---

### Example Walk Through

Suppose one sentence of length 2, embedding size 3, $E=4$ experts, and $k=2$.  
- After flattening, you get 2 softmax distributions of length 4.  
- You pick the top 2 experts for each token and renormalize their weights.  
- Each selected expert produces a 3-dimensional output for its tokens.  
- You weight and sum those outputs to yield the final 3-dimensional vector per token.

This sparse routing mechanism dramatically cuts computation only $k$ experts run per token instead of all $E$ while retaining the expressivity of a full ensemble.

### Verified Pure NumPy Solution (From Scratch)
```python
import numpy as np

def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

def get_top_k(arr: np.ndarray, k: int):
    idx = np.argpartition(arr, -k)[..., -k:]
    vals = np.take_along_axis(arr, idx, axis=-1)
    return idx, vals

def expert(x: np.ndarray, We_i: np.ndarray):
    # x: [n_tokens, d_model]
    # We_i: [d_model, d_model]
    return x @ We_i

def gate(x: np.ndarray, Wg: np.ndarray):
    # x: [n_batch * l_seq, d_model]
    # Wg: [n_batch * l_seq, n_experts]
    return x @ Wg

def moe(x: np.ndarray, We: np.ndarray, Wg: np.ndarray, n_experts: int, top_k: int):
    # x: [n_batch, l_seq, d_model]
    # We: [n_experts, d_model, d_model]
    # Wg: [n_batch * l_seq, n_experts]

    n_batch, l_seq, d_model = x.shape

    # flatten batch and sequence dimensions for easier indexing
    # x_flat: [n_batch * l_seq, d_model]
    x_flat = x.reshape(-1, d_model)
    n_tokens, _ = x_flat.shape

    gating_logits = gate(x_flat, Wg)
    gating_weights = softmax(gating_logits, axis=-1)

    topk_idx, topk_weights = get_top_k(gating_weights, top_k)
    topk_idx_flat = topk_idx.flatten()  # [n_tokens * top_k]
    # mapping from top K expert indices to token indices: [n_tokens * top_k]
    token_idx_flat = np.arange(n_tokens).repeat(top_k)

    topk_weights_norm = topk_weights / topk_weights.sum(axis=1, keepdims=True)
    topk_weights_norm_flat = topk_weights_norm.flatten()

    # prepare result memory for aggregation: [n_tokens, d_model]
    output_flat = np.zeros_like(x_flat)
    for i in range(n_experts):
        mask = topk_idx_flat == i
        tokens_expert_i = token_idx_flat[mask]

        if tokens_expert_i.size > 0:
            x_expert_i = x_flat[tokens_expert_i]
            output_expert_i = expert(x_expert_i, We[i, ...])
            output_expert_i *= topk_weights_norm_flat[mask, None]

            # scatter add to result memory
            np.add.at(output_flat, tokens_expert_i, output_expert_i)

    return output_flat.reshape(n_batch, l_seq, d_model)
```

### Verified Sample Trace Data
```json
{
  "test": "import numpy as np\nnp.random.seed(42)\nd_model = 2\nn_experts = 4\nl_seq = 3\nn_batch = 2\ntop_k = 2\nx = np.random.rand(n_batch, l_seq, d_model)\nWe = np.random.rand(n_experts, d_model, d_model)\nWg = np.random.rand(d_model, n_experts)\noutput = moe(x, We, Wg, n_experts, top_k)\nprint(np.round(output, 4))",
  "expected_output": "[[[0.5148 0.4329]\n  [0.5554 0.5447]\n  [0.1285 0.102 ]]\n\n [[0.339  0.3046]\n  [0.5391 0.417 ]\n  [0.3597 0.3262]]]"
}
```

---

## Problem 126: Implement Group Normalization
- **Difficulty**: `medium`
- **Category**: Deep Learning

### Description & Spec
Write a Python function to perform Group Normalization on a 4D input tensor with shape (B, C, H, W). The function should normalize over smaller groups of channels, then apply a learned scale (gamma) and shift (beta).

### Mathematical Derivation & Intuition
## Understanding Group Normalization

Group Normalization (GN) is a normalization technique that divides the channels into groups and normalizes the activations within each group. Unlike Batch Normalization, which normalizes over the entire mini-batch, Group Normalization normalizes over groups of channels and is less dependent on the batch size. This makes it particularly useful for tasks with small batch sizes or when using architectures such as segmentation networks where spatial resolution is important.

### Concepts

Group Normalization operates on the principle of normalizing within smaller groups of channels. The process reduces **internal covariate shift** within these groups and helps stabilize training, especially in scenarios where the batch size is small or varies across tasks.

The process of Group Normalization consists of the following steps:

1. **Divide Channels into Groups:** Split the feature channels into several groups. The number of groups is determined by the **n_groups** parameter.
2. **Compute the Mean and Variance within Each Group:** For each group, compute the mean and variance of the activations within the group, across the spatial dimensions and batch.
3. **Normalize the Inputs:** Normalize the activations of each group using the computed mean and variance.
4. **Apply Scale and Shift:** After normalization, apply a learned scale (gamma) and shift (beta) to restore the model's ability to represent the data's original distribution.

### Structure of Group Normalization for BCHW Input

For an input tensor with the shape **BCHW** , where:
- **B**: batch size,
- **C**: number of channels,
- **H**: height,
- **W**: width,
the Group Normalization process operates on specific dimensions based on the task's requirement.

#### 1. Group Division

- The input feature dimension **C** (channels) is divided into several groups. The number of groups is determined by the **n_groups** parameter, and the size of each group is calculated as:

  $$
  \text{groupSize} = \frac{C}{n_{\text{groups}}}
  $$

  Where:
  - **C** is the number of channels.
  - **n_groups** is the number of groups into which the channels are divided.
  - **groupSize** is the number of channels in each group.

  The input tensor is then reshaped to group the channels into the specified groups.

#### 2. Mean and Variance Calculation within Groups

- For each group, the **mean** $\mu_g$ and **variance** $\sigma_g^2$ are computed over the spatial dimensions and across the batch. This normalization helps to stabilize the activations within each group.

  $$ 
  \mu_g = \frac{1}{B \cdot H \cdot W \cdot \text{groupSize}} \sum_{i=1}^{B} \sum_{h=1}^{H} \sum_{w=1}^{W} \sum_{g=1}^{\text{groupSize}} x_{i,g,h,w}
  $$

  $$
  \sigma_g^2 = \frac{1}{B \cdot H \cdot W \cdot \text{groupSize}} \sum_{i=1}^{B} \sum_{h=1}^{H} \sum_{w=1}^{W} \sum_{g=1}^{\text{groupSize}} (x_{i,g,h,w} - \mu_g)^2
  $$

  Where:
  - $x_{i,g,h,w}$ is the activation at batch index $i$, group index $g$, height $h$, and width $w$.
  - $B$ is the batch size.
  - $H$ and $W$ are the spatial dimensions (height and width).
  - $\text{groupSize}$ is the number of channels in each group.

#### 3. Normalization

Once the mean $\mu_g$ and variance $\sigma_g^2$ have been computed for each group, the next step is to **normalize** the input. The normalization is done by subtracting the mean and dividing by the standard deviation (square root of the variance, plus a small constant $\epsilon$ for numerical stability):

$$
\hat{x}_{i,g,h,w} = \frac{x_{i,g,h,w} - \mu_g}{\sqrt{\sigma_g^2 + \epsilon}}
$$

Where:
- $\hat{x}_{i,g,h,w}$ is the normalized activation for the input at batch index $i$, group index $g$, height $h$, and width $w$.
- $\epsilon$ is a small constant to avoid division by zero.

#### 4. Scale and Shift

After normalization, the next step is to apply a **scale** ($\gamma_g$) and **shift** ($\beta_g$) to the normalized activations for each group. These learned parameters allow the model to adjust the output distribution of each group:

$$
y_{i,g,h,w} = \gamma_g \hat{x}_{i,g,h,w} + \beta_g
$$

Where:
- $\gamma_g$ is the scaling factor for group $g$.
- $\beta_g$ is the shifting factor for group $g$.

#### 5. Training and Inference

- **During Training**: The mean and variance are computed for each mini-batch and used for normalization within each group.
- **During Inference**: The model uses running averages of the statistics (mean and variance) that were computed during training to ensure consistent behavior when deployed.

### Key Points

- **Group-wise Normalization**: Group Normalization normalizes within smaller groups of channels instead of normalizing over the entire batch and all channels. This allows for more stable training in cases with small batch sizes.

- **Number of Groups**: The number of groups is a hyperparameter (**n_groups**) that can significantly affect the model’s performance. It is typically set to divide the total number of channels into groups of equal size.

- **Smaller Batch Sizes**: Group Normalization is less dependent on the batch size, making it ideal for situations where batch sizes are small (e.g., segmentation tasks).

- **Numerical Stability**: As with other normalization techniques, a small constant $\epsilon$ is added to the variance to avoid numerical instability when dividing by the square root of variance.

- **Improved Convergence**: Group Normalization can help improve the gradient flow throughout the network, making it easier to train deep networks with small batch sizes. It also helps speed up convergence and stabilize training.

- **Regularization Effect**: Similar to Batch Normalization, Group Normalization introduces a form of regularization through the normalization process. It can reduce overfitting by acting as a noise source during training.

### Why Normalize Over Groups?

- **Group-wise Normalization**: By dividing the channels into smaller groups, Group Normalization ensures that each group has a stable distribution of activations, making it effective even when batch sizes are small.

- **Less Dependency on Batch Size**: Unlike Batch Normalization, Group Normalization does not require large batch sizes to compute accurate statistics. This makes it well-suited for tasks such as image segmentation, where large batch sizes may not be feasible.

- **Channel-wise Learning**: Group Normalization allows each group to learn independently, preserving flexibility while also controlling the complexity of normalization over channels.

By normalizing over smaller groups, Group Normalization can reduce internal covariate shift and allow for faster and more stable training, even in situations where Batch Normalization may be less effective due to small batch sizes.

### Verified Pure NumPy Solution (From Scratch)
```python
def group_normalization(X: np.ndarray, gamma: np.ndarray, beta: np.ndarray, num_groups: int, epsilon: float = 1e-5) -> np.ndarray:
    '''
    Perform Group Normalization.
    
    Args:
    X: numpy array of shape (B, C, H, W), input data
    gamma: numpy array of shape (C,), scale parameter
    beta: numpy array of shape (C,), shift parameter
    num_groups: number of groups for normalization
    epsilon: small constant to avoid division by zero
    
    Returns:
    norm_X: numpy array of shape (B, C, H, W), normalized output
    '''
    batch_size, num_channels, height, width = X.shape
    group_size = num_channels // num_groups

    # Reshape X into groups
    X_reshaped = X.reshape(batch_size, num_groups, group_size, height, width)

    # Compute mean and variance for each group
    mean = np.mean(X_reshaped, axis=(2, 3, 4), keepdims=True)
    variance = np.var(X_reshaped, axis=(2, 3, 4), keepdims=True)

    X_norm = (X_reshaped - mean) / np.sqrt(variance + epsilon)

    # Reshape back to the original shape
    X_norm = X_norm.reshape(batch_size, num_channels, height, width)

    # Apply scale and shift
    norm_X = gamma * X_norm + beta
    return norm_X
```

### Verified Sample Trace Data
```json
{
  "test": "np.random.seed(42)\nB, C, H, W = 2, 2, 2, 2\nX = np.random.randn(B, C, H, W)\ngamma = np.ones(C).reshape(1, C, 1, 1)\nbeta = np.zeros(C).reshape(1, C, 1, 1)\nnum_groups = 2\noutput = group_normalization(X, gamma, beta, num_groups)\nprint(np.round(output, 4))",
  "expected_output": "[[[[-0.2287, -1.2998], [ 0.026, 1.5025]], [[-0.926, -0.9259], [1.46, 0.3919]]], [[[-0.5848, 1.732 ], [-0.5709, -0.5762]], [[1.4005, -1.0503], [-0.8361, 0.486 ]]]]"
}
```

---

## Problem 128: Dynamic Tanh: Normalization-Free Transformer Activation
- **Difficulty**: `easy`
- **Category**: Deep Learning

### Description & Spec
Implement the Dynamic Tanh (DyT) function, a normalization-free transformation inspired by the Tanh function. DyT replaces layer normalization in Transformer architectures while preserving squashing behavior and enabling stable training.

### Mathematical Derivation & Intuition
A new study (https://arxiv.org/pdf/2503.10622) demonstrates that layer normalization, that is ubiquitous in transformers, produces Tanh-like S-shapes. By incorporating a new layer replacement for normalization called "Dynamic Tanh" (DyT for short), Transformers without normalization can match or exceed the performance of their normalized counterparts, mostly without hyperparameter tuning.

### Normalization layer
Consider an standard NLP task, where an input $x$ has a shape of $(B,T,C)$, where $B$ is the batch size, $T$ - number of tokens (sequence length) and $C$ - embedding dimensions. Then an output of a normalization layer is generally computed as $norm(x)=\gamma(\frac{x-\mu}{\sqrt{\sigma^2+\varepsilon}})+\beta$, where $\gamma$ and $\beta$ are learnable parameters of shape $(C,)$. Distribution's statistics are calculated as follows: $\mu_k=\frac{1}{BT}\sum_i^B\sum_j^Tx_{ij}$; $\sigma_k^2=\frac{1}{B T} \sum_{i, j}\left(x_{i j k}-\mu_k\right)^2$

### Hyperboloic tangent (Tanh)
Tanh function is defined as a ratio: $tanh(x)=\frac{sinh(x)}{cosh(x)}=\frac{exp(x)-exp(-x)}{exp(x)+exp(-x)}$. Essentially the function allows transformation of an arbitrary domain to $[-1,1]$. 

### Dynamic Tanh (DyT)
Turns out that LN (layer normalization) produces different parts of a $tanh(kx)$, where $k$ controls the curvature of the tanh curve in the center. The smaller the $k$, the smoother is the change from $-1$ to $1$. Hence the study proposes a drop-in replacement for LN given an input tensor $x$:

$$
DyT(x)=\gamma*tanh(\alpha x)+\beta,
$$

where:
* $\alpha$ - learnable parameter that allows scaling the input differently based on its range (tokens producing **smaller variance** produce **less smoother curves**). Authors suggest a **default value** of $0.5$.
* $\gamma, \beta$ - learnable parameters, that scale our output based on the input. Authors suggest initializing these vectors with following **default values**:
    * $\gamma$ as all-one vector 
    * $\beta$ as all-zero

Despite not calculating statistics, DyT preserves the "squashing" effect of LN on extreme values in a non-linear fashion, while almost linearly transforming central parts of the input.

### Verified Pure NumPy Solution (From Scratch)
```python
import numpy as np

def dynamic_tanh(x: np.ndarray, alpha: float, gamma: float, beta: float) -> list[float]:
    def tanh(x: np.ndarray) -> np.ndarray:
        return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

    x = tanh(alpha * x)
    return (x * gamma + beta).round(4).tolist()
```

### Verified Sample Trace Data
```json
{
  "test": "import numpy as np\nx = np.array([[[0.94378259]],[[0.97754654]],[[0.36168351]],[[0.51821078]],[[0.76961589]]])\ngamma = np.ones((1,))\nbeta = np.zeros((1,))\nprint(np.round(dynamic_tanh(x, 0.5, gamma, beta),4))",
  "expected_output": "[[[0.4397]], [[0.4532]], [[0.1789]], [[0.2535]], [[0.3669]]]"
}
```

---

## Problem 130: Implement a Simple CNN Training Function with Backpropagation
- **Difficulty**: `hard`
- **Category**: Deep Learning

### Description & Spec
Create a function that trains a basic Convolutional Neural Network (CNN) using backpropagation. The network should include one convolutional layer with ReLU activation, followed by flattening and a dense layer with softmax output, and a cross entropy loss. You need to handle the forward pass, compute the loss gradients, and update the weights and biases using stochastic gradient descent. Ensure the function processes input data as grayscale images and one-hot encoded labels, and returns the trained weights and biases for the convolutional and dense layers.

### Mathematical Derivation & Intuition
## Understanding a Simple Convolutional Neural Network with Backpropagation

A **Convolutional Neural Network** (CNN) learns two things at once:  
1. **What to look for** - small filters (kernels) that detect edges, textures, etc.  
2. **How to combine those detections** - a dense layer that converts them into class probabilities.

Below is the full training loop broken into intuitive steps that can be implemented directly in NumPy.

---

### 1. Forward Pass

**Convolution**  
The convolution layer slides a small filter over the input and produces feature maps:

$$
Z^c[p, q, k] = \sum_{i, j} X[p+i, q+j] \cdot W^c[i, j, k] + b^c[k]
$$

This results in a tensor of shape $(H - k + 1, W - k + 1, F)$, where $H$ and $W$ are the input height and width, $k$ is the kernel size, and $F$ is the number of filters.

**ReLU Activation**

$$
A^c = \max(0, Z^c)
$$

This introduces non-linearity by zeroing out negative values.

**Flattening**

The feature maps are reshaped into a vector:

$$
A^f = \text{flatten}(A^c)
$$

**Dense Layer**

$$
Z^d = A^f \cdot W^d + b^d
$$

Each entry in $A^f$ contributes to every output class via weight matrix $W^d$ and bias $b^d$.

**Softmax Activation**

$$
\hat{y}_c = \frac{e^{Z^d_c}}{\sum_j e^{Z^d_j}}
$$

This converts raw scores into probabilities for classification.

---

### 2. Loss Function – Cross Entropy

For one-hot encoded label $y$ and prediction $\hat{y}$:

$$
\mathcal{L}(\hat{y}, y) = -\sum_c y_c \log(\hat{y}_c)
$$

This penalizes incorrect predictions based on confidence.

---

### 3. Backward Pass

**Gradient of Softmax + Cross Entropy**

$$
\frac{\partial \mathcal{L}}{\partial Z^d} = \hat{y} - y
$$

**Dense Layer Gradients**

$\frac{\partial \mathcal{L}}{\partial W^d} = (A^f)^T \cdot \frac{\partial \mathcal{L}}{\partial Z^d}$,
and the gradient with respect to biases is
$\frac{\partial \mathcal{L}}{\partial b^d} = \frac{\partial \mathcal{L}}{\partial Z^d}$.

Reshape the upstream gradient to the shape of $A^c$ for backpropagation through ReLU.

**ReLU Gradient**

$$
\frac{\partial \mathcal{L}}{\partial Z^c} = \frac{\partial \mathcal{L}}{\partial A^c} \cdot \mathbf{1}(Z^c > 0)
$$

**Convolution Filter Gradients**

For each filter $k$:

$$
\frac{\partial \mathcal{L}}{\partial W^c_{i,j,k}} = \sum_{p,q} \frac{\partial \mathcal{L}}{\partial Z^c_{p,q,k}} \cdot X_{p+i, q+j}
$$

$$
\frac{\partial \mathcal{L}}{\partial b^c_k} = \sum_{p,q} \frac{\partial \mathcal{L}}{\partial Z^c_{p,q,k}}
$$

---

### 4. Updating Parameters (SGD)

With learning rate $\eta$:

$$
W \leftarrow W - \eta \cdot \frac{\partial \mathcal{L}}{\partial W}
$$

$$
b \leftarrow b - \eta \cdot \frac{\partial \mathcal{L}}{\partial b}
$$

Repeat this process for each sample (stochastic gradient descent) and for multiple epochs.

---

### 5. Example Walkthrough

Suppose $X$ is a grayscale image:

$$
X = \begin{bmatrix}
1 & 2 & 3 \\
4 & 5 & 6 \\
7 & 8 & 9
\end{bmatrix}
$$

And the kernel is:

$$
K = \begin{bmatrix}
1 & 0 \\
0 & -1
\end{bmatrix}
$$

Perform convolution at the top-left:

$$
(1 \cdot 1 + 2 \cdot 0 + 4 \cdot 0 + 5 \cdot (-1)) = 1 - 5 = -4
$$

After ReLU: max(0, -4) = 0  
Flatten the result -> Dense layer -> Softmax output -> Compute loss

Backpropagate the error to adjust weights, and repeat to learn better filters and classifications over time.

### Verified Pure NumPy Solution (From Scratch)
```python
import numpy as np

def train_simple_cnn_with_backprop(X, y, epochs, learning_rate, kernel_size=3, num_filters=1):
    '''
    Trains a simple CNN with one convolutional layer, ReLU activation, flattening, and a dense layer with softmax output using backpropagation.

    Assumes X has shape (n_samples, height, width) for grayscale images and y is one-hot encoded with shape (n_samples, num_classes).

    Parameters:
    X : np.ndarray, input data
    y : np.ndarray, one-hot encoded labels
    epochs : int, number of training epochs
    learning_rate : float, learning rate for weight updates
    kernel_size : int, size of the square convolutional kernel
    num_filters : int, number of filters in the convolutional layer

    Returns:
    W_conv, b_conv, W_dense, b_dense : Trained weights and biases for the convolutional and dense layers
    '''
    n_samples, height, width = X.shape
    num_classes = y.shape[1]

    # Initialize weights and biases
    W_conv = np.random.randn(kernel_size, kernel_size, num_filters) * 0.01
    b_conv = np.zeros(num_filters)
    output_height = height - kernel_size + 1
    output_width = width - kernel_size + 1
    flattened_size = output_height * output_width * num_filters
    W_dense = np.random.randn(flattened_size, num_classes) * 0.01
    b_dense = np.zeros(num_classes)

    for epoch in range(epochs):
        for i in range(n_samples):  # Stochastic Gradient Descent with batch size 1
            # Forward pass
            # Convolutional layer
            Z_conv = np.zeros((output_height, output_width, num_filters))
            for k in range(num_filters):
                for p in range(output_height):
                    for q in range(output_width):
                        Z_conv[p, q, k] = np.sum(X[i, p:p+kernel_size, q:q+kernel_size] * W_conv[:, :, k]) + b_conv[k]
            A_conv = np.maximum(Z_conv, 0)  # ReLU activation
            A_flat = A_conv.flatten()  # Flatten the output

            # Dense layer
            Z_dense = np.dot(A_flat, W_dense) + b_dense
            exp_Z_dense = np.exp(Z_dense - np.max(Z_dense))  # Numerical stability for softmax
            A_dense = exp_Z_dense / np.sum(exp_Z_dense)

            # Backpropagation
            # Loss gradient for cross-entropy with softmax
            dZ_dense = A_dense - y[i]

            # Dense layer gradients
            dW_dense = np.outer(A_flat, dZ_dense)
            db_dense = dZ_dense
            dA_flat = np.dot(dZ_dense, W_dense.T)

            # Reshape and backprop through ReLU
            dA_conv = dA_flat.reshape(A_conv.shape)
            dZ_conv = dA_conv * (A_conv > 0).astype(float)

            # Convolutional layer gradients
            dW_conv = np.zeros_like(W_conv)
            db_conv = np.zeros(num_filters)
            for k in range(num_filters):
                db_conv[k] = np.sum(dZ_conv[:, :, k])
                for ii in range(kernel_size):
                    for jj in range(kernel_size):
                   
                        dW_conv[ii, jj, k] = np.sum(dZ_conv[:, :, k] * X[i, ii:ii+output_height, jj:jj+output_width])

            
# Update weights and biases
            W_conv -= learning_rate * dW_conv
            b_conv -= learning_rate * db_conv
            W_dense -= learning_rate * dW_dense
            b_dense -= learning_rate * db_dense

    return W_conv, b_conv, W_dense, b_dense
```

### Verified Sample Trace Data
```json
{
  "test": "import numpy as np; np.random.seed(42); X = np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]); y = np.array([[1, 0]]); print(train_simple_cnn_with_backprop(X, y, 1, 0.01, 3, 1))",
  "expected_output": "(array([[[ 0.00501739],        [-0.00128214],        [ 0.00662764]],       [[ 0.01543131],        [-0.00209028],        [-0.00203986]],       [[ 0.01614389],        [ 0.00807636],        [-0.00424248]]]), array([5.02517066e-05]), array([[ 0.00635715, -0.00556573]]), array([ 0.00499531, -0.00499531]))"
}
```

---

## Problem 134: Compute Multi-class Cross-Entropy Loss
- **Difficulty**: `easy`
- **Category**: Deep Learning

### Description & Spec
Implement a function that computes the average cross-entropy loss for a batch of predictions in a multi-class classification task. Your function should take in a batch of predicted probabilities and one-hot encoded true labels, then return the average cross-entropy loss. Ensure that you handle numerical stability by clipping probabilities by epsilon.

### Mathematical Derivation & Intuition
## Multi-class Cross-Entropy Loss Implementation

Cross-entropy loss, also known as log loss, measures the performance of a classification model whose output is a probability value between 0 and 1. For multi-class classification tasks, we use the categorical cross-entropy loss.

### Mathematical Background

For a single sample with C classes, the categorical cross-entropy loss is defined as:

$L = -\sum_{c=1}^{C} y_c \log(p_c)$

where:

- $y_c$ is a binary indicator (0 or 1) if class label c is the correct classification for the sample
- $p_c$ is the predicted probability that the sample belongs to class c
- $C$ is the number of classes

### Implementation Requirements

Your task is to implement a function that computes the average cross-entropy loss across multiple samples:

$L_{batch} = -\frac{1}{N}\sum_{n=1}^{N}\sum_{c=1}^{C} y_{n,c} \log(p_{n,c})$

where N is the number of samples in the batch.

### Important Considerations

- Handle numerical stability by adding a small epsilon to avoid log(0)
- Ensure predicted probabilities sum to 1 for each sample
- Return average loss across all samples
- Handle invalid inputs appropriately

The function should take predicted probabilities and true labels as input and return the average cross-entropy loss.

### Verified Pure NumPy Solution (From Scratch)
```python
import numpy as np

def compute_cross_entropy_loss(predicted_probs: np.ndarray, true_labels: np.ndarray,epsilon = 1e-15) -> float:

    predicted_probs = np.clip(predicted_probs, epsilon, 1 - epsilon)

    #Write your code here
    log_probs = np.log(predicted_probs)
    loss = -np.sum(true_labels * log_probs, axis=1)
    return float(np.mean(loss))
```

### Verified Sample Trace Data
```json
{
  "test": "import numpy as np\npred = np.array([[1, 0, 0], [0, 1, 0]])\ntrue = np.array([[1, 0, 0], [0, 1, 0]])\nprint(round(compute_cross_entropy_loss(pred, true), 4))",
  "expected_output": "0.0"
}
```

---

## Problem 137: Implement a Dense Block with 2D Convolutions
- **Difficulty**: `hard`
- **Category**: Deep Learning

### Description & Spec
Create a function `dense_net_block` that performs the forward pass of a **DenseNet dense block** on a batch of images stored in an **NHWC** NumPy tensor `input_data` (shape `(N, H, W, C0)`). The block must run `num_layers` iterations; at each iteration it should (i) apply **ReLU** to the running feature tensor, (ii) convolve it with the corresponding kernel from `kernels` (using stride 1, no bias, and symmetric zero-padding so that `H` and `W` are preserved), and (iii) concatenate the convolution output (whose channel count equals `growth_rate`) to the running tensor along the channel axis. Every kernel `kernels[l]` therefore has shape `(kh, kw, C0 + l x growth_rate, growth_rate)`, where `(kh, kw)` equals `kernel_size` (default `(3, 3)`). After the final layer the function must return a tensor of shape `(N, H, W, C0 + num_layers x growth_rate)`. If any kernel's input-channel dimension does not match the current feature-map channels, the function should raise a `ValueError`.

### Mathematical Derivation & Intuition
## Understanding Dense Blocks and 2D Convolutions

Dense blocks are a key innovation in the DenseNet architecture. Each layer receives input from **all** previous layers, leading to rich feature reuse and efficient gradient flow.

### Dense Block Concept
For a dense block:
- **Each layer**: Applies ReLU, then 2D convolution, and then concatenates the output to previous features.
- Mathematically:
$$
x_l = H_l([x_0, x_1, \ldots, x_{l-1}])
$$
where $H_l(\cdot)$ is the convolution and activation operations.

### 2D Convolution Basics
A 2D convolution at a position $(i, j)$ for input $X$ and kernel $K$ is:
$$
Y[i, j] = \sum_{m=0}^{k_h - 1} \sum_{n=0}^{k_w - 1} X[i + m, j + n] \cdot K[m, n]
$$

### Padding to Preserve Spatial Dimensions
To preserve height and width:
$$
\text{padding} = \frac{k - 1}{2}
$$

### Dense Block Growth
- Each layer adds $\text{growth rate}$ channels.
- After $L$ layers, total channels = input channels + $L \times \text{growth rate}$.

### Putting It All Together
1️⃣ Start with an input tensor.  
2️⃣ Repeat for $\text{num layers}$:
- Apply ReLU activation.
- Apply 2D convolution (with padding).
- Concatenate the output along the channel dimension.

By understanding these core principles, you’re ready to build the dense block function!

### Verified Pure NumPy Solution (From Scratch)
```python
import numpy as np

def conv2d(x, kernel, padding=0):
    if padding > 0:
        x_padded = np.pad(x, ((0, 0), (padding, padding), (padding, padding), (0, 0)), mode='constant')
    else:
        x_padded = x
    batch_size, in_height, in_width, in_channels = x_padded.shape
    kh, kw, _, out_channels = kernel.shape
    out_height = in_height - kh + 1
    out_width = in_width - kw + 1
    output = np.zeros((batch_size, out_height, out_width, out_channels))
    for b in range(batch_size):
        for i in range(out_height):
            for j in range(out_width):
                for c_out in range(out_channels):
                    sum_val = 0.0
                    for c_in in range(in_channels):
                        sum_val += np.sum(x_padded[b, i:i+kh, j:j+kw, c_in] * kernel[:, :, c_in, c_out])
                    output[b, i, j, c_out] = sum_val
    return output

def dense_net_block(input_data, num_layers, growth_rate, kernels, kernel_size=(3, 3)):
    kh, kw = kernel_size
    padding = (kh - 1) // 2
    concatenated_features = input_data.copy()
    for l in range(num_layers):
        activated = np.maximum(concatenated_features, 0.0)
        conv_output = conv2d(activated, kernels[l], padding=padding)
        concatenated_features = np.concatenate([concatenated_features, conv_output], axis=3)
    return concatenated_features
```

### Verified Sample Trace Data
```json
{
  "test": "import numpy as np\nnp.random.seed(42)\nX = np.random.randn(1, 1, 1, 2)\nkernels = [np.random.randn(3, 3, 2 + i*1, 1) * 0.01 for i in range(2)]\nprint(dense_net_block(X, 2, 1, kernels))",
  "expected_output": "[[[[ 4.96714153e-01, -1.38264301e-01, -2.30186127e-03, -6.70426255e-05]]]]"
}
```

---

## Problem 143: Instance Normalization (IN) Implementation
- **Difficulty**: `medium`
- **Category**: Deep Learning

### Description & Spec
Implement the Instance Normalization operation for 4D tensors (B, C, H, W) using NumPy. For each instance in the batch and each channel, normalize the spatial dimensions (height and width) by subtracting the mean and dividing by the standard deviation, then apply a learned scale (gamma) and shift (beta).

### Mathematical Derivation & Intuition
## Understanding Instance Normalization

Instance Normalization (IN) is a normalization technique primarily used in image generation and style transfer tasks. Unlike Batch Normalization or Group Normalization, Instance Normalization normalizes each individual sample (or instance) separately, across its spatial dimensions. This is particularly effective in applications like style transfer, where normalization is needed per image to preserve the content while allowing different styles to be applied.

### Concepts

Instance Normalization operates on the principle of normalizing each individual sample independently. This helps to remove the style information from the images, leaving only the content. By normalizing each instance, the method allows the model to focus on the content of the image rather than the variations between images in a batch.

The process of Instance Normalization consists of the following steps:

1. **Compute the Mean and Variance for Each Instance:** For each instance (image), compute the mean and variance across its spatial dimensions.
2. **Normalize the Inputs:** Normalize each instance using the computed mean and variance.
3. **Apply Scale and Shift:** After normalization, apply a learned scale (gamma) and shift (beta) to restore the model's ability to represent the data's original distribution.

### Structure of Instance Normalization for BCHW Input

For an input tensor with the shape **BCHW** , where:
- **B**: batch size,
- **C**: number of channels,
- **H**: height,
- **W**: width,
Instance Normalization operates on the spatial dimensions (height and width) of each instance (image) separately.

#### 1. Mean and Variance Calculation for Each Instance

- For each individual instance in the batch (for each **b** in **B**), the **mean** $\mu_b$ and **variance** $\sigma_b^2$ are computed across the spatial dimensions (height and width), but **independently for each channel**.

  $$ 
  \mu_b = \frac{1}{H \cdot W} \sum_{h=1}^{H} \sum_{w=1}^{W} x_{b,c,h,w}
  $$

  $$
  \sigma_b^2 = \frac{1}{H \cdot W} \sum_{h=1}^{H} \sum_{w=1}^{W} (x_{b,c,h,w} - \mu_b)^2
  $$

  Where:
  - $x_{b,c,h,w}$ is the activation at batch index $b$, channel $c$, height $h$, and width $w$.
  - $H$ and $W$ are the spatial dimensions (height and width).

#### 2. Normalization

Once the mean $\mu_b$ and variance $\sigma_b^2$ have been computed for each instance, the next step is to **normalize** the input for each instance across the spatial dimensions (height and width), for each channel:

$$
\hat{x}_{b,c,h,w} = \frac{x_{b,c,h,w} - \mu_b}{\sqrt{\sigma_b^2 + \epsilon}}
$$

Where:
- $\hat{x}_{b,c,h,w}$ is the normalized activation for the input at batch index $b$, channel index $c$, height $h$, and width $w$.
- $\epsilon$ is a small constant added to the variance for numerical stability.

#### 3. Scale and Shift

After normalization, the next step is to apply a **scale** ($\gamma_c$) and **shift** ($\beta_c$) to the normalized activations for each channel. These learned parameters allow the model to adjust the output distribution for each channel:

$$
y_{b,c,h,w} = \gamma_c \hat{x}_{b,c,h,w} + \beta_c
$$

Where:
- $\gamma_c$ is the scaling factor for channel $c$.
- $\beta_c$ is the shifting factor for channel $c$.

#### 4. Training and Inference

- **During Training**: The mean and variance are computed for each instance in the mini-batch and used for normalization.
- **During Inference**: The model uses the running averages of the statistics (mean and variance) computed during training to ensure consistent behavior in production.

### Key Points

- **Instance-wise Normalization**: Instance Normalization normalizes each image independently, across its spatial dimensions (height and width) and across the channels.
  
- **Style Transfer**: This normalization technique is widely used in **style transfer** tasks, where each image must be normalized independently to allow for style information to be adjusted without affecting the content.

- **Batch Independence**: Instance Normalization does not depend on the batch size, as normalization is applied per instance, making it suitable for tasks where per-image normalization is critical.

- **Numerical Stability**: A small constant $\epsilon$ is added to the variance to avoid numerical instability when dividing by the square root of the variance.

- **Improved Training in Style-Related Tasks**: Instance Normalization helps to remove unwanted style-related variability across different images, allowing for better performance in tasks like style transfer, where the goal is to separate content and style information.

### Why Normalize Over Instances?

- **Content Preservation**: By normalizing each image individually, Instance Normalization allows the model to preserve the content of the images while adjusting the style. This makes it ideal for style transfer and other image manipulation tasks.
  
- **Batch Independence**: Unlike Batch Normalization, which requires large batch sizes to compute statistics, Instance Normalization normalizes each image independently, making it suitable for tasks where the batch size is small or varies.

- **Reducing Style Variability**: Instance Normalization removes the variability in style information across a batch, allowing for a consistent representation of content across different images.

In summary, Instance Normalization is effective for image-based tasks like style transfer, where the goal is to normalize each image independently to preserve its content while allowing style modifications.

### Verified Pure NumPy Solution (From Scratch)
```python
import numpy as np

def instance_normalization(X: np.ndarray, gamma: np.ndarray, beta: np.ndarray, epsilon: float = 1e-5) -> np.ndarray:
    # Reshape gamma, beta for broadcasting: (1, C, 1, 1)
    gamma = gamma.reshape(1, -1, 1, 1)
    beta = beta.reshape(1, -1, 1, 1)
    mean = np.mean(X, axis=(2, 3), keepdims=True)
    var = np.var(X, axis=(2, 3), keepdims=True)
    X_norm = (X - mean) / np.sqrt(var + epsilon)
    return gamma * X_norm + beta
```

### Verified Sample Trace Data
```json
{
  "test": "import numpy as np\nB, C, H, W = 2, 2, 2, 2\nnp.random.seed(42)\nX = np.random.randn(B, C, H, W)\ngamma = np.ones(C)\nbeta = np.zeros(C)\nout = instance_normalization(X, gamma, beta)\nprint(np.round(out[1][1], 4))",
  "expected_output": "[[ 1.4005, -1.0503] [-0.8361, 0.486 ]]"
}
```

---

## Problem 145: Adagrad Optimizer
- **Difficulty**: `easy`
- **Category**: Deep Learning

### Description & Spec
Implement the Adagrad optimizer update step function. Your function should take the current parameter value, gradient, and accumulated squared gradients as inputs, and return the updated parameter value and new accumulated squared gradients. The function should also handle scalar and array inputs, and include proper input validation.

### Mathematical Derivation & Intuition
# Implementing Adagrad Optimizer

## Introduction
Adagrad (Adaptive Gradient Algorithm) is an optimization algorithm that adapts the learning rate to each parameter, performing larger updates for infrequent parameters and smaller updates for frequent ones. This makes it particularly well-suited for dealing with sparse data.

## Learning Objectives
- Understand how Adagrad optimizer works
- Learn to implement adaptive learning rates
- Gain practical experience with gradient-based optimization

## Theory
Adagrad adapts the learning rate for each parameter based on the historical gradients. The key equations are:

$G_t = G_{t-1} + g_t^2$ (Accumulated squared gradients)

$\theta_t = \theta_{t-1} - \dfrac{\alpha}{\sqrt{G_t} + \epsilon} \cdot g_t$ (Parameter update)

Where:
- $G_t$ is the sum of squared gradients up to time step t
- $\alpha$ is the initial learning rate
- $\epsilon$ is a small constant for numerical stability
- $g_t$ is the gradient at time step t

Read more at:

1. Duchi, J., Hazan, E., & Singer, Y. (2011). Adaptive subgradient methods for online learning and stochastic optimization. Journal of Machine Learning Research, 12, 2121–2159. [PDF](https://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf)
2. Ruder, S. (2017). An overview of gradient descent optimization algorithms. [arXiv:1609.04747](https://arxiv.org/pdf/1609.04747)


## Problem Statement
Implement the Adagrad optimizer update step function. Your function should take the current parameter value, gradient, and accumulated squared gradients as inputs, and return the updated parameter value and new accumulated squared gradients.

### Input Format
The function should accept:
- parameter: Current parameter value
- grad: Current gradient
- G: Accumulated squared gradients
- learning_rate: Learning rate (default=0.01)
- epsilon: Small constant for numerical stability (default=1e-8)

### Output Format
Return tuple: (updated_parameter, updated_G)

## Example
```python
# Example usage:
parameter = 1.0
grad = 0.1
G = 1.0

new_param, new_G = adagrad_optimizer(parameter, grad, G)
```

## Tips
- Initialize G as zeros
- Use numpy for numerical operations
- Test with both scalar and array inputs

---

### Verified Pure NumPy Solution (From Scratch)
```python
import numpy as np

def adagrad_optimizer(parameter, grad, G, learning_rate=0.01, epsilon=1e-8):
    """
    Update parameters using the Adagrad optimizer.
    Adapts the learning rate for each parameter based on the historical gradients.

    Args:
        parameter: Current parameter value
        grad: Current gradient
        G: Accumulated squared gradients
        learning_rate: Learning rate (default=0.01)
        epsilon: Small constant for numerical stability (default=1e-8)

    Returns:
        tuple: (updated_parameter, updated_G)
    """
    assert learning_rate > 0, "Learning rate must be positive"
    assert epsilon > 0, "Epsilon must be positive"
    assert all(G >= 0) if isinstance(G, np.ndarray) else G >= 0, "G must be non-negative"

    # Update accumulated squared gradients
    G = G + grad**2

    # Update parameters using adaptive learning rate
    update = learning_rate * grad / (np.sqrt(G) + epsilon)
    parameter = parameter - update

    return np.round(parameter, 5), np.round(G, 5)
```

### Verified Sample Trace Data
```json
{
  "test": "print(adagrad_optimizer(1., 0.5, 1., 0.01, 1e-8))",
  "expected_output": "(0.99553, 1.25)"
}
```

---

## Problem 146: Momentum Optimizer
- **Difficulty**: `easy`
- **Category**: Deep Learning

### Description & Spec
Implement the momentum optimizer update step function. Your function should take the current parameter value, gradient, and velocity as inputs, and return the updated parameter value and new velocity. The function should also handle scalar and array inputs.

### Mathematical Derivation & Intuition
# Implementing Momentum Optimizer

## Introduction
Momentum is a popular optimization technique that helps accelerate gradient descent in the relevant direction and dampens oscillations. It works by adding a fraction of the previous update vector to the current gradient.

## Learning Objectives
- Understand how momentum optimization works
- Learn to implement momentum-based gradient updates
- Understand the effect of momentum on optimization

## Theory
Momentum optimization uses a moving average of gradients to determine the direction of the update. The key equations are:

$v_t = \gamma v_{t-1} + \eta \nabla_\theta J(\theta)$ (Velocity update)

$\theta_t = \theta_{t-1} - v_t$ (Parameter update)

Where:
- $v_t$ is the velocity at time t
- $\gamma$ is the momentum coefficient (typically 0.9)
- $\eta$ is the learning rate
- $\nabla_\theta J(\theta)$ is the gradient of the loss function

Read more at:

1. Ruder, S. (2017). An overview of gradient descent optimization algorithms. [arXiv:1609.04747](https://arxiv.org/pdf/1609.04747)


## Problem Statement
Implement the momentum optimizer update step function. Your function should take the current parameter value, gradient, and velocity as inputs, and return the updated parameter value and new velocity.

### Input Format
The function should accept:
- parameter: Current parameter value
- grad: Current gradient
- velocity: Current velocity
- learning_rate: Learning rate (default=0.01)
- momentum: Momentum coefficient (default=0.9)

### Output Format
Return tuple: (updated_parameter, updated_velocity)

## Example
```python
# Example usage:
parameter = 1.0
grad = 0.1
velocity = 0.1

new_param, new_velocity = momentum_optimizer(parameter, grad, velocity)
```

## Tips
- Initialize velocity as zero
- Use numpy for numerical operations
- Test with both scalar and array inputs

---

### Verified Pure NumPy Solution (From Scratch)
```python
import numpy as np

def momentum_optimizer(parameter, grad, velocity, learning_rate=0.01, momentum=0.9):
    """
    Update parameters using the momentum optimizer.
    Uses momentum to accelerate learning in relevant directions and dampen oscillations.

    Args:
        parameter: Current parameter value
        grad: Current gradient
        velocity: Current velocity/momentum term
        learning_rate: Learning rate (default=0.01)
        momentum: Momentum coefficient (default=0.9)

    Returns:
        tuple: (updated_parameter, updated_velocity)
    """
    assert learning_rate > 0, "Learning rate must be positive"
    assert 0 <= momentum < 1, "Momentum must be between 0 and 1"

    # Update velocity
    velocity = momentum * velocity + learning_rate * grad
    
    # Update parameters
    parameter = parameter - velocity

    return np.round(parameter, 5), np.round(velocity, 5)
```

### Verified Sample Trace Data
```json
{
  "test": "print(momentum_optimizer(1., 0.1, 0.5, 0.01, 0.9))",
  "expected_output": "(0.549, 0.451)"
}
```

---

## Problem 148: Adamax Optimizer
- **Difficulty**: `easy`
- **Category**: Deep Learning

### Description & Spec
Implement the Adamax optimizer update step function. Your function should take the current parameter value, gradient, and moving averages as inputs, and return the updated parameter value and new moving averages. The function should also handle scalar and array inputs and include bias correction for the moving averages.

### Mathematical Derivation & Intuition
# Implementing Adamax Optimizer

## Introduction
Adamax is a variant of Adam optimizer that uses the infinity norm (max) instead of the L2 norm for the second moment estimate. This makes it more robust in some cases and can lead to better convergence in certain scenarios, particularly when dealing with sparse gradients.

## Learning Objectives
- Understand how Adamax optimization works
- Learn to implement Adamax-based gradient updates
- Understand the effect of infinity norm on optimization

## Theory
Adamax maintains a moving average of gradients (first moment) and uses the infinity norm for the second moment estimate. The key equations are:

First moment estimate (same as Adam):

$m_t = \beta_1 m_{t-1} + (1-\beta_1)g_t$

The second moment estimate in Adam uses the $l_2$ norm:

$v_t = \beta_2 v_{t-1} + (1-\beta_2)|g_t|^2$

This can be generalized to the $l_p$ norm, but norms for large p values are numerically unstable. However, Adamax uses the $l_\infin$ norm (infinity norm), which converges to:

$u_t = \max(\beta_2 \cdot u_{t-1}, |g_t|)$

Unlike Adam, Adamax doesn't require bias correction for $u_t$ because the max operation makes it less susceptible to bias towards zero.

Bias correction:
$\hat{m}_t = \dfrac{m_t}{1-\beta_1^t}$

Parameter update:
$\theta_t = \theta_{t-1} - \dfrac{\eta}{u_t} \hat{m}_t$

Where:
- $m_t$ is the first moment estimate at time t
- $u_t$ is the infinity norm estimate at time t
- $\beta_1$ is the first moment coefficient (typically 0.9)
- $\beta_2$ is the second moment coefficient (typically 0.999)
- $\eta$ is the learning rate
- $g_t$ is the gradient at time t

Note: Unlike Adam, Adamax doesn't require bias correction for $u_t$ because the max operation makes it less susceptible to bias towards zero.

Read more at:

1. Kingma, D. and Ba, J. (2015). Adam: A Method for Stochastic Optimization. [arXiv:1412.6980](https://arxiv.org/abs/1412.6980)
2. Ruder, S. (2017). An overview of gradient descent optimization algorithms. [arXiv:1609.04747](https://arxiv.org/pdf/1609.04747)


## Problem Statement
Implement the Adamax optimizer update step function. Your function should take the current parameter value, gradient, and moment estimates as inputs, and return the updated parameter value and new moment estimates.

### Input Format
The function should accept:
- parameter: Current parameter value
- grad: Current gradient
- m: First moment estimate
- u: Infinity norm estimate
- t: Current timestep
- learning_rate: Learning rate (default=0.002)
- beta1: First moment decay rate (default=0.9)
- beta2: Second moment decay rate (default=0.999)
- epsilon: Small constant for numerical stability (default=1e-8)

### Output Format
Return tuple: (updated_parameter, updated_m, updated_u)

## Example
```python
# Example usage:
parameter = 1.0
grad = 0.1
m = 0.0
u = 0.0
t = 1

new_param, new_m, new_u = adamax_optimizer(parameter, grad, m, u, t)
```

## Tips
- Initialize m and u as zeros
- Keep track of timestep t for bias correction
- Use numpy for numerical operations
- Test with both scalar and array inputs
- Remember to apply bias correction to the first moment estimate

---

### Verified Pure NumPy Solution (From Scratch)
```python
import numpy as np

def adamax_optimizer(parameter, grad, m, u, t, learning_rate=0.002, beta1=0.9, beta2=0.999, epsilon=1e-8):
    """
    Update parameters using the Adamax optimizer.
    Adamax is a variant of Adam based on the infinity norm.
    It uses the maximum of past squared gradients instead of the exponential moving average.

    Args:
        parameter: Current parameter value
        grad: Current gradient
        m: First moment estimate
        u: Infinity norm estimate
        t: Current timestep
        learning_rate: Learning rate (default=0.002)
        beta1: First moment decay rate (default=0.9)
        beta2: Infinity norm decay rate (default=0.999)
        epsilon: Small constant for numerical stability (default=1e-8)

    Returns:
        tuple: (updated_parameter, updated_m, updated_u)
    """
    assert learning_rate > 0, "Learning rate must be positive"
    assert 0 <= beta1 < 1, "Beta1 must be between 0 and 1"
    assert 0 <= beta2 < 1, "Beta2 must be between 0 and 1"
    assert epsilon > 0, "Epsilon must be positive"
    assert all(u >= 0) if isinstance(u, np.ndarray) else u >= 0, "u must be non-negative"

    # Update biased first moment estimate
    m = beta1 * m + (1 - beta1) * grad

    # Update infinity norm estimate
    u = np.maximum(beta2 * u, np.abs(grad))

    # Compute bias-corrected first moment estimate
    m_hat = m / (1 - beta1**t)

    # Update parameters
    update = learning_rate * m_hat / (u + epsilon)
    parameter = parameter - update

    return np.round(parameter, 5), np.round(m, 5), np.round(u, 5)
```

### Verified Sample Trace Data
```json
{
  "test": "print(adamax_optimizer(1., 0.1, 1., 1., 1, 0.002, 0.9, 0.999, 1e-8))",
  "expected_output": "(0.98178, 0.91, 0.999)"
}
```

---

## Problem 149: Adadelta Optimizer
- **Difficulty**: `medium`
- **Category**: Deep Learning

### Description & Spec
Implement the Adadelta optimizer update step function. Your function should take the current parameter value, gradient, and moving averages as inputs, and return the updated parameter value and new moving averages. The function should handle both scalar and array inputs, and include proper input validation.

### Mathematical Derivation & Intuition
# Implementing Adadelta Optimizer

## Introduction
Adadelta is an extension of Adagrad that addresses two key issues: the aggressive, monotonically decreasing learning rate and the need for manual learning rate tuning. While Adagrad accumulates all past squared gradients, Adadelta restricts the influence of past gradients to a window of size w. Instead of explicitly storing w past gradients, it efficiently approximates this window using an exponential moving average with decay rate ρ, making it more robust to parameter updates. Additionally, it automatically handles the units of the updates, eliminating the need for a manually set learning rate.

## Learning Objectives
- Understand how Adadelta optimizer works
- Learn to implement adaptive learning rates with moving averages

## Theory
Adadelta uses two main ideas:
1. Exponential moving average of squared gradients to approximate a window of size w
2. Automatic unit correction through the ratio of parameter updates

The key equations are:

$v_t = \rho v_{t-1} + (1-\rho)g_t^2$ (Exponential moving average of squared gradients)

The above approximates a window size of $w \approx \dfrac{1}{1-\rho}$ 

$\Delta\theta_t = -\dfrac{\sqrt{u_{t-1} + \epsilon}}{\sqrt{v_t + \epsilon}} \cdot g_t$ (Parameter update with unit correction)

$u_t = \rho u_{t-1} + (1-\rho)\Delta\theta_t^2$ (Exponential moving average of squared parameter updates)

Where:
- $v_t$ is the exponential moving average of squared gradients (decay rate ρ)
- $u_t$ is the exponential moving average of squared parameter updates (decay rate ρ)
- $\rho$ is the decay rate (typically 0.9) that controls the effective window size w ≈ 1/(1-ρ)
- $\epsilon$ is a small constant for numerical stability
- $g_t$ is the gradient at time step t

The ratio $\dfrac{\sqrt{u_{t-1} + \epsilon}}{\sqrt{v_t + \epsilon}}$ serves as an adaptive learning rate that automatically handles the units of the updates, making the algorithm more robust to different parameter scales. Unlike Adagrad, Adadelta does not require a manually set learning rate, making it especially useful when tuning hyperparameters is difficult. This automatic learning rate adaptation is achieved through the ratio of the root mean squared (RMS) of parameter updates to the RMS of gradients.

Read more at:

1. Zeiler, M. D. (2012). ADADELTA: An Adaptive Learning Rate Method. [arXiv:1212.5701](https://arxiv.org/abs/1212.5701)
2. Ruder, S. (2017). An overview of gradient descent optimization algorithms. [arXiv:1609.04747](https://arxiv.org/pdf/1609.04747)

## Problem Statement
Implement the Adadelta optimizer update step function. Your function should take the current parameter value, gradient, and accumulated statistics as inputs, and return the updated parameter value and new accumulated statistics.

### Input Format
The function should accept:
- parameter: Current parameter value
- grad: Current gradient
- v: Exponentially decaying average of squared gradients
- u: Exponentially decaying average of squared parameter updates
- rho: Decay rate (default=0.9)
- epsilon: Small constant for numerical stability (default=1e-8)

### Output Format
Return tuple: (updated_parameter, updated_v, updated_u)

## Example
```python
# Example usage:
parameter = 1.0
grad = 0.1
v = 1.0
u = 1.0

new_param, new_v, new_u = adadelta_optimizer(parameter, grad, v, u)
```

## Tips
- Initialize v and u as zeros
- Use numpy for numerical operations
- Test with both scalar and array inputs
- The learning rate is automatically determined by the algorithm

---

### Verified Pure NumPy Solution (From Scratch)
```python
import numpy as np

def adadelta_optimizer(parameter, grad, u, v, rho=0.95, epsilon=1e-6):
    """
    Update parameters using the AdaDelta optimizer.
    AdaDelta is an extension of AdaGrad that seeks to reduce its aggressive,
    monotonically decreasing learning rate.

    Args:
        parameter: Current parameter value
        grad: Current gradient
        u: Running average of squared gradients
        v: Running average of squared parameter updates
        rho: Decay rate for the moving average (default=0.95)
        epsilon: Small constant for numerical stability (default=1e-6)

    Returns:
        tuple: (updated_parameter, updated_u, updated_v)
    """
    assert 0 <= rho < 1, "Rho must be between 0 and 1"
    assert epsilon > 0, "Epsilon must be positive"
    assert all(u >= 0) if isinstance(u, np.ndarray) else u >= 0, "u must be non-negative"
    assert all(v >= 0) if isinstance(v, np.ndarray) else v >= 0, "v must be non-negative"

    # Update running average of squared gradients
    u = rho * u + (1 - rho) * grad**2

    # Compute RMS of gradient
    RMS_g = np.sqrt(u + epsilon)

    # Compute RMS of parameter updates
    RMS_dx = np.sqrt(v + epsilon)

    # Compute parameter update
    dx = -RMS_dx / RMS_g * grad

    # Update running average of squared parameter updates
    v = rho * v + (1 - rho) * dx**2

    # Update parameters
    parameter = parameter + dx

    return np.round(parameter, 5), np.round(u, 5), np.round(v, 5)
```

### Verified Sample Trace Data
```json
{
  "test": "print(adadelta_optimizer(1., 0.5, 1., 1., 0.95, 1e-6))",
  "expected_output": "(0.49035, 0.9625, 0.96299)"
}
```

---

## Problem 150: Nesterov Accelerated Gradient Optimizer
- **Difficulty**: `easy`
- **Category**: Deep Learning

### Description & Spec
Implement the Nesterov Accelerated Gradient (NAG) optimizer update step function. Your function should take the current parameter value, gradient function, and velocity as inputs, and return the updated parameter value and new velocity. The function should use the "look-ahead" approach where momentum is applied before computing the gradient, and should handle both scalar and array inputs.

### Mathematical Derivation & Intuition
# Implementing Nesterov Accelerated Gradient (NAG) Optimizer

## Introduction
Nesterov Accelerated Gradient (NAG) is an improvement over classical momentum optimization. While momentum helps accelerate gradient descent in the relevant direction, NAG takes this a step further by looking ahead in the direction of the momentum before computing the gradient. This "look-ahead" property helps NAG make more informed updates and often leads to better convergence.

## Learning Objectives
- Understand how Nesterov Accelerated Gradient optimization works
- Learn to implement NAG-based gradient updates
- Understand the advantages of NAG over classical momentum
- Gain practical experience with advanced gradient-based optimization

## Theory
Nesterov Accelerated Gradient uses a "look-ahead" approach where it first makes a momentum-based step and then computes the gradient at that position. The key equations are:

$\theta_{lookahead, t-1} = \theta_{t-1} - \gamma v_{t-1}$ (Look-ahead position)

$v_t = \gamma v_{t-1} + \eta \nabla_\theta J(\theta_{lookahead, t-1})$ (Velocity update)

$\theta_t = \theta_{t-1} - v_t$ (Parameter update)

Where:
- $v_t$ is the velocity at time t
- $\gamma$ is the momentum coefficient (typically 0.9)
- $\eta$ is the learning rate
- $\nabla_\theta J(\theta)$ is the gradient of the loss function

The key difference from classical momentum is that the gradient is evaluated at $\theta_{lookahead, t-1}$ instead of $\theta_{t-1}$

Read more at:

1. Nesterov, Y. (1983). A method for solving the convex programming problem with convergence rate O(1/k²). Doklady Akademii Nauk SSSR, 269(3), 543-547.
2. Ruder, S. (2017). An overview of gradient descent optimization algorithms. [arXiv:1609.04747](https://arxiv.org/pdf/1609.04747)


## Problem Statement
Implement the Nesterov Accelerated Gradient optimizer update step function. Your function should take the current parameter value, gradient function, and velocity as inputs, and return the updated parameter value and new velocity.

### Input Format
The function should accept:
- parameter: Current parameter value
- gradient function: A function that accepts parameters and returns gradient computed at that point
- velocity: Current velocity
- learning_rate: Learning rate (default=0.01)
- momentum: Momentum coefficient (default=0.9)

### Output Format
Return tuple: (updated_parameter, updated_velocity)

## Example
```python
# Example usage:
def grad_func(parameter):
    # Returns gradient
    pass

parameter = 1.0
velocity = 0.1

new_param, new_velocity = nag_optimizer(parameter, grad_func, velocity)
```

## Tips
- Initialize velocity as zero
- Use numpy for numerical operations
- Test with both scalar and array inputs
- Remember that the gradient should be computed at the look-ahead position

---

### Verified Pure NumPy Solution (From Scratch)
```python
import numpy as np

def nag_optimizer(parameter, grad_fn, velocity, learning_rate=0.01, momentum=0.9):
    """
    Update parameters using the Nesterov Accelerated Gradient optimizer.
    Uses a "look-ahead" approach to improve convergence by applying momentum before computing the gradient.

    Args:
        parameter: Current parameter value
        grad_fn: Function that computes the gradient at a given position
        velocity: Current velocity (momentum term)
        learning_rate: Learning rate (default=0.01)
        momentum: Momentum coefficient (default=0.9)

    Returns:
        tuple: (updated_parameter, updated_velocity)
    """
    assert 0 <= momentum < 1, "Momentum must be between 0 and 1"
    assert learning_rate > 0, "Learning rate must be positive"

    # Compute look-ahead position
    look_ahead = parameter - momentum * velocity
    
    # Compute gradient at look-ahead position
    grad = grad_fn(look_ahead)
    
    # Update velocity using momentum and gradient
    velocity = momentum * velocity + learning_rate * grad
    
    # Update parameters using the new velocity
    parameter = parameter - velocity

    return np.round(parameter, 5), np.round(velocity, 5)
```

### Verified Sample Trace Data
```json
{
  "test": "import numpy as np\ndef gradient_function(x):\n    if isinstance(x, np.ndarray):\n        n = len(x)\n        return x - np.arange(n)\n    else:\n        return x - 0\nprint(nag_optimizer(1., gradient_function, 0.5, 0.01, 0.9))",
  "expected_output": "(0.5445, 0.4555)"
}
```

---

## Problem 151: Dropout Layer
- **Difficulty**: `medium`
- **Category**: Deep Learning

### Description & Spec
Implement a dropout layer that applies random neuron deactivation during training to prevent overfitting in neural networks. The layer should randomly zero out a proportion of input elements based on a dropout rate p, scale the remaining values by 1/(1-p) to maintain expected values, and pass inputs unchanged during inference. During backpropagation, gradients must be masked with the same dropout pattern and scaled by the same factor to ensure proper gradient flow.

### Mathematical Derivation & Intuition
# Implementing Dropout Layer

## Introduction
Dropout is a regularization technique that randomly deactivates neurons during training to prevent overfitting. It forces the network to learn with different neurons and prevents it from becoming too dependent on specific neurons.

## Learning Objectives
- Understand the concept and purpose of dropout
- Learn how dropout works during training and inference
- Implement dropout layer with proper scaling

## Theory
During training, dropout randomly sets a proportion of inputs to zero and scales up the remaining values to maintain the expected value. The mathematical formulation is:

During training:

$y = \dfrac{x \odot m}{1-p}$

During inference:

$y = x$

During backpropagation:

$grad = \dfrac{grad \odot m}{1-p}$

Where:
- $x$ is the input vector
- $m$ is a binary mask vector sampled from Bernoulli(p)
- $\odot$ represents element-wise multiplication
- $p$ is the dropout rate (probability of keeping a neuron)

The mask $m$ is randomly generated for each forward pass during training and is stored in memory to be used in the corresponding backward pass. This ensures that the same neurons are dropped during both forward and backward propagation for a given input.

The scaling factor $\frac{1}{1-p}$ during training ensures that the expected value of the output matches the input, making the network's behavior consistent between training and inference.

During backpropagation, the gradients must also be scaled by the same factor $\frac{1}{1-p}$ to maintain the correct gradient flow.

Dropout acts as a form of regularization by:
1. Preventing co-adaptation of neurons, forcing them to learn more robust features that are useful in combination with many different random subsets of other neurons
2. Creating an implicit ensemble of networks, as each forward pass uses a different subset of neurons, effectively training multiple networks that share parameters
3. Reducing the effective capacity of the network during training, which helps prevent overfitting by making the model less likely to memorize the training data

Read more at:

1. Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I., & Salakhutdinov, R. (2014). Dropout: A Simple Way to Prevent Neural Networks from Overfitting. Journal of Machine Learning Research, 15(1), 1929-1958. [PDF](https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf)

## Problem Statement
Implement a dropout layer class that can be used during both training and inference phases of a neural network. The implementation should:

1. Apply dropout during training by randomly zeroing out elements
2. Scale the remaining values appropriately to maintain expected values
3. Pass through inputs unchanged during inference
4. Support backpropagation by storing and using the dropout mask

### Requirements
The `DropoutLayer` class should implement:

1. `__init__(p: float)`: Initialize with dropout probability p
2. `forward(x: np.ndarray, training: bool = True) -> np.ndarray`: Apply dropout during forward pass
3. `backward(grad: np.ndarray) -> np.ndarray`: Handle gradient flow during backpropagation

### Input Parameters
- `p`: Dropout rate (probability of keeping a neuron), must be between 0 and 1
- `x`: Input tensor of any shape
- `training`: Boolean flag indicating if in training mode
- `grad`: Gradient tensor during backpropagation

### Output
- Forward pass: Tensor of same shape as input with dropout applied
- Backward pass: Gradient tensor with dropout mask applied

## Example
```python
# Example usage:
x = np.array([1.0, 2.0, 3.0, 4.0])
grad = np.array([0.1, 0.2, 0.3, 0.4])
p = 0.5  # 50% dropout rate

# During training
output_train = dropout_layer(x, p, training=True)

# During inference
output_inference = dropout_layer(x, p, training=False)

# Backward
grad_back = dropout.backward(grad)
```

## Tips
- Use numpy's random binomial generator for creating the mask
- Remember to scale up the output and gradients during training by 1/(1-p)
- Test with different dropout rates (typically between 0.2 and 0.5)
- Verify that the expected value of the output matches the input

## Common Pitfalls
- Using the same mask for all examples in a batch
- Setting dropout rate too high (can lead to underfitting)

---

### Verified Pure NumPy Solution (From Scratch)
```python
import numpy as np

class DropoutLayer:
    def __init__(self, p: float):
        """Initialize the dropout layer."""
        if p < 0 or p >= 1:
            raise ValueError("Dropout rate must be between 0 and 1 (1-exclusive)")
        
        self.p = p
        self.mask = None
        
    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """Forward pass of the dropout layer."""
        if not training:
            return x
                
        self.mask = np.random.binomial(1, 1 - self.p, x.shape)
        
        return x * self.mask / (1 - self.p)
        
    def backward(self, grad: np.ndarray) -> np.ndarray:
        """Backward pass of the dropout layer."""
        if self.mask is None:
            raise ValueError("Forward pass must be called before backward pass")
            
        return grad * self.mask / (1 - self.p)
```

### Verified Sample Trace Data
```json
{
  "test": "import numpy as np\nnp.random.seed(42)\nx = np.array([[1.0, 2.0], [3.0, 4.0]])\ngrad = np.array([[0.5, 0.2], [1.0, 2.0]])\n\ndropout = DropoutLayer(0.2)\n\nprint(dropout.forward(x, training=True), dropout.forward(x, training=False), dropout.backward(grad))",
  "expected_output": "(array([[1.25, 0.], [3.75, 5.]]), array([[1.0, 2.0], [3.0, 4.0]]), array([[0.625, 0.], [1.25, 2.5]]))"
}
```

---

## Problem 151: Implement SwiGLU activation function
- **Difficulty**: `easy`
- **Category**: Deep Learning

### Description & Spec
## Problem

Implement a Python function that applies the **SwiGLU activation function** to a NumPy array.

Assume the input array has already been passed through a linear projection and has shape `(batch_size, 2d)`. Round each output to four decimal places and return the result as a NumPy array of the shape `(batch_size, d)`.

### Mathematical Derivation & Intuition
## Understanding the SwiGLU Activation Function

As the name suggests the SwiGLU activation function is a combination of two activations - Swish (implemented as SiLU in PyTorch) and GLU (Gated Linear Unit). It is important that we understand Swish and GLU because SwiGLU inherits properties from both — the smooth self-gating behavior of Swish, the decoupled gating structure of GLU.

### Swish Activation (Self-Gating)

**Swish**, introduced by Google Brain, is a smooth, self-gated activation function defined as:

$$
\text{Swish}(x) = x \cdot \sigma(x)
$$

where the sigmoid function is:

$$
\sigma(x) = \frac{1}{1 + e^{-x}}
$$

In Swish, the same input $x$ is used to:
  - **Compute the gate**: $\sigma(x)$
  - **Modulate itself**: $x \cdot \sigma(x)$

This is called **self-gating** — the input both **creates** and **passes through** the gate. \
**Note:** When written in a PyTorch forward loop, it looks something like -
```bash
import torch.nn.functional as F

def forward(self, x):
   x1 = self.fc1(x)   # x1 = Wx + b where W, b are learnable params
   output = F.silu(x) # output = x1 * sigmoid(x1) 
   return output      # output = (Wx + b) * sigmoid(Wx + b)
```
This essentially means that the gate is learnable, and the model learns the best shape of the activation function.

### Gated Linear Unit (GLU)

**GLU**, introduced in *Language Modeling with Gated Convolutional Networks* (Dauphin et al., 2017), is a gated activation mechanism defined as:

$$
\text{GLU}(x_1, x_2) = x_1 \cdot \sigma(x_2)
$$

Here:
- $x_1$ is the **input signal**.
- $x_2$ is used to **compute the gate** via the sigmoid function.

In practice, both $x_1$ and $x_2$ are obtained by **splitting the output of a single linear layer**:

```bash
import torch.nn.functional as F

def forward(self, x):
   x_proj = self.fc1(x)                
   x1, x2 = x_proj.chunk(2, dim=-1)    # x1 = Wx + b, x2 = Vx + c
   output = x1 * torch.sigmoid(x2)     # GLU = x1 · σ(x2)
   return output
```
So GLU can be rewritten as:

$$
\text{GLU}(x) = x_1 \cdot \sigma(x_2)
$$
where:
$$x_1 = W x + b$$
 $$x_2 = V x + c$$


This is a learned, cross-gating mechanism — the model learns different parameters for the signal and the gate.


## SwiGLU

With Swish and GLU out of the way, it becomes very easy to understand **SwiGLU**. It is defined as:

$$
\text{SwiGLU}(x) = x_1 \cdot \text{Swish}(x_2)
$$

Where:
- $x_1, x_2$ are typically obtained by splitting a linear projection of the input (inspired by GLU).

- $\text{Swish}(x_2) = x_2 \cdot \sigma(x_2)$ is the self-gated activation.

So putting it together:

$$
\text{SwiGLU}(x) = x_1 \cdot (x_2 \cdot \sigma(x_2))
$$

This combines the **signal-gate decoupling** of GLU with the **smooth self-gating** of Swish, and is used in the feed-forward blocks of large-scale models like Google's PaLM, Meta's LLaMA.


### Why Does It Work?
> Noam Shazeer, the author in his paper writes: "We offer no explanation as to why these architectures seem to work; we attribute their success, as all else, to divine benevolence."

The improvement in performance have only been proven *emprically* by observing faster convergence during training

### Verified Pure NumPy Solution (From Scratch)
```python
import numpy as np

def SwiGLU(x: np.ndarray) -> np.ndarray:
    """
    Args:
        x: np.ndarray of shape (batch_size, 2d)

    Returns:
        np.ndarray of shape (batch_size, d)
    """
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    
    d = x.shape[1] // 2
    x1 = x[:, :d]
    x2 = x[:, d:]
    return x1 * (x2 * sigmoid(x2))
```

### Verified Sample Trace Data
```json
{
  "test": "print(np.round(SwiGLU(np.zeros((1, 4))), 4))",
  "expected_output": "[[0., 0.]]"
}
```

---

## Problem 174: Train a Simple GAN on 1D Gaussian Data
- **Difficulty**: `hard`
- **Category**: Deep Learning

### Description & Spec
In this task, you will train a Generative Adversarial Network (GAN) to learn a one-dimensional Gaussian distribution. The GAN consists of a generator that produces samples from latent noise and a discriminator that estimates the probability that a given sample is real. Both networks should have one hidden layer with ReLU activation in the hidden layer. The generator’s output layer is linear, while the discriminator's output layer uses a sigmoid activation.

You must train the GAN using the standard non-saturating GAN loss for the generator and binary cross-entropy loss for the discriminator. In the NumPy version, parameters should be updated using vanilla gradient descent. In the PyTorch version, parameters should be updated using stochastic gradient descent (SGD) with the specified learning rate. The training loop should alternate between updating the discriminator and the generator each iteration.

Your function must return the trained generator forward function `gen_forward(z)`, which produces generated samples given latent noise.

### Mathematical Derivation & Intuition
## Understanding GANs for 1D Gaussian Data
A Generative Adversarial Network (GAN) consists of two neural networks - a **Generator** $G_\theta$ and a **Discriminator** $D_\phi$ - trained in a minimax game.

### 1. The Roles
- **Generator** $G_\theta(z)$: Takes a latent noise vector $z \sim \mathcal{N}(0, I)$ and outputs a sample intended to resemble the real data.
- **Discriminator** $D_\phi(x)$: Outputs a probability $p \in (0, 1)$ that the input $x$ came from the real data distribution rather than the generator.

### 2. The Objective
The classical GAN objective is:
$$
\min_{\theta} \; \max_{\phi} \; \mathbb{E}_{x \sim p_{\text{data}}} [\log D_\phi(x)] + \mathbb{E}_{z \sim p(z)} [\log (1 - D_\phi(G_\theta(z)))]
$$
Here:
- $p_{\text{data}}$ is the real data distribution.
- $p(z)$ is the prior distribution for the latent noise (often standard normal).

### 3. Practical Losses
In implementation, we minimize:
- **Discriminator loss**:
$$
\mathcal{L}_D = - \left( \frac{1}{m} \sum_{i=1}^m \log D(x^{(i)}_{\text{real}}) + \log(1 - D(x^{(i)}_{\text{fake}})) \right)
$$
- **Generator loss** (non-saturating form):
$$
\mathcal{L}_G = - \frac{1}{m} \sum_{i=1}^m \log D(G(z^{(i)}))
$$

### 4. Forward/Backward Flow
1. **Discriminator step**: Real samples $x_{\text{real}}$ and fake samples $x_{\text{fake}} = G(z)$ are passed through $D$, and $\mathcal{L}_D$ is minimized w.r.t. $\phi$.
2. **Generator step**: Fresh $z$ is sampled, $x_{\text{fake}} = G(z)$ is passed through $D$, and $\mathcal{L}_G$ is minimized w.r.t. $\theta$ while keeping $\phi$ fixed.

### 5. Architecture for This Task
- **Generator**: Fully connected layer ($\mathbb{R}^{\text{latent\_dim}} \to \mathbb{R}^{\text{hidden\_dim}}$) -> ReLU -> Fully connected layer ($\mathbb{R}^{\text{hidden\_dim}} \to \mathbb{R}^1$).
- **Discriminator**: Fully connected layer ($\mathbb{R}^1 \to \mathbb{R}^{\text{hidden\_dim}}$) → ReLU → Fully connected layer ($\mathbb{R}^{\text{hidden\_dim}} \to \mathbb{R}^1$) → Sigmoid.

### 6. Numerical Tips
- Initialize weights with a small Gaussian ($\mathcal{N}(0, 0.01)$).
- Add $10^{-8}$ to logs for numerical stability.
- Use a consistent batch size $m$ for both real and fake samples.
- Always sample fresh noise for the generator on each update.

**Your Task**: Implement the training loop to learn the parameters $\theta$ and $\phi$, and return the trained `gen_forward(z)` function. The evaluation (mean/std of generated samples) will be handled in the test cases.

### Verified Pure NumPy Solution (From Scratch)
```python
import numpy as np

def relu(x):
    return np.maximum(0, x)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def train_gan(mean_real: float, std_real: float, latent_dim: int = 1, hidden_dim: int = 16, learning_rate: float = 0.001, epochs: int = 5000, batch_size: int = 128, seed: int = 42):
    np.random.seed(seed)
    data_dim = 1

    # Initialize generator weights
    w1_g = np.random.normal(0, 0.01, (latent_dim, hidden_dim))
    b1_g = np.zeros(hidden_dim)
    w2_g = np.random.normal(0, 0.01, (hidden_dim, data_dim))
    b2_g = np.zeros(data_dim)

    # Initialize discriminator weights
    w1_d = np.random.normal(0, 0.01, (data_dim, hidden_dim))
    b1_d = np.zeros(hidden_dim)
    w2_d = np.random.normal(0, 0.01, (hidden_dim, 1))
    b2_d = np.zeros(1)

    def disc_forward(x):
        h1 = np.dot(x, w1_d) + b1_d
        a1 = relu(h1)
        logit = np.dot(a1, w2_d) + b2_d
        p = sigmoid(logit)
        return p, logit, a1, h1

    def gen_forward(z):
        h1 = np.dot(z, w1_g) + b1_g
        a1 = relu(h1)
        x_gen = np.dot(a1, w2_g) + b2_g
        return x_gen, a1, h1

    for epoch in range(epochs):
        # Sample real data
        x_real = np.random.normal(mean_real, std_real, batch_size)[:, None]
        z = np.random.normal(0, 1, (batch_size, latent_dim))
        x_fake, _, _ = gen_forward(z)

        # Discriminator forward
        p_real, _, a1_real, h1_real = disc_forward(x_real)
        p_fake, _, a1_fake, h1_fake = disc_forward(x_fake)

        # Discriminator gradients
        grad_logit_real = - (1 - p_real) / batch_size
        grad_a1_real = grad_logit_real @ w2_d.T
        grad_h1_real = grad_a1_real * (h1_real > 0)
        grad_w1_d_real = x_real.T @ grad_h1_real
        grad_b1_d_real = np.sum(grad_h1_real, axis=0)
        grad_w2_d_real = a1_real.T @ grad_logit_real
        grad_b2_d_real = np.sum(grad_logit_real, axis=0)

        grad_logit_fake = p_fake / batch_size
        grad_a1_fake = grad_logit_fake @ w2_d.T
        grad_h1_fake = grad_a1_fake * (h1_fake > 0)
        grad_w1_d_fake = x_fake.T @ grad_h1_fake
        grad_b1_d_fake = np.sum(grad_h1_fake, axis=0)
        grad_w2_d_fake = a1_fake.T @ grad_logit_fake
        grad_b2_d_fake = np.sum(grad_logit_fake, axis=0)

        grad_w1_d = grad_w1_d_real + grad_w1_d_fake
        grad_b1_d = grad_b1_d_real + grad_b1_d_fake
        grad_w2_d = grad_w2_d_real + grad_w2_d_fake
        grad_b2_d = grad_b2_d_real + grad_b2_d_fake

        w1_d -= learning_rate * grad_w1_d
        b1_d -= learning_rate * grad_b1_d
        w2_d -= learning_rate * grad_w2_d
        b2_d -= learning_rate * grad_b2_d

        # Generator update
        z = np.random.normal(0, 1, (batch_size, latent_dim))
        x_fake, a1_g, h1_g = gen_forward(z)
        p_fake, _, a1_d, h1_d = disc_forward(x_fake)

        grad_logit_fake = - (1 - p_fake) / batch_size
        grad_a1_d = grad_logit_fake @ w2_d.T
        grad_h1_d = grad_a1_d * (h1_d > 0)
        grad_x_fake = grad_h1_d @ w1_d.T

        grad_a1_g = grad_x_fake @ w2_g.T
        grad_h1_g = grad_a1_g * (h1_g > 0)
        grad_w1_g = z.T @ grad_h1_g
        grad_b1_g = np.sum(grad_h1_g, axis=0)
        grad_w2_g = a1_g.T @ grad_x_fake
        grad_b2_g = np.sum(grad_x_fake, axis=0)

        w1_g -= learning_rate * grad_w1_g
        b1_g -= learning_rate * grad_b1_g
        w2_g -= learning_rate * grad_w2_g
        b2_g -= learning_rate * grad_b2_g

    return gen_forward
```

### Verified Sample Trace Data
```json
{
  "test": "gen_forward = train_gan(4.0, 1.25, epochs=1000, seed=42)\nz = np.random.normal(0, 1, (500, 1))\nx_gen, _, _ = gen_forward(z)\nprint((round(np.mean(x_gen), 4), round(np.std(x_gen), 4)))",
  "expected_output": "(0.0004, 0.0002)"
}
```

---

