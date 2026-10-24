# Deep-ML Knowledge Base: Computer Vision

Total verified problems: 2

Source: [Deep-ML (Open-Deep-ML)](https://www.deep-ml.com/)

---

## Problem 70: Calculate Image Brightness
- **Difficulty**: `easy`
- **Category**: Computer Vision

### Description & Spec
## Task: Image Brightness Calculator

In this task, you will implement a function `calculate_brightness(img)` that calculates the average brightness of a grayscale image. The image is represented as a 2D matrix, where each element represents a pixel value between 0 (black) and 255 (white).

### **Your Task**:
Implement the function `calculate_brightness(img)` to:
1. Return the average brightness of the image rounded to two decimal places.
2. Handle edge cases:
   - If the image matrix is empty.
   - If the rows in the matrix have inconsistent lengths.
   - If any pixel values are outside the valid range (0-255).

For any of these edge cases, the function should return `-1`.

### Mathematical Derivation & Intuition
# Image Brightness Calculator

Consider a grayscale image represented as a 2D matrix where each element represents a pixel value between 0 (black) and 255 (white):

$$
Image = \begin{pmatrix}
p_{11} & p_{12} \\
p_{21} & p_{22}
\end{pmatrix}
$$

The average brightness is calculated as:

$$
Brightness = \frac{\sum_{i=1}^{m} \sum_{j=1}^{n} p_{ij}}{m \times n}
$$

Where:

1) $p_{ij}$ is the pixel value at position $(i,j)$  
2) $m$ is the number of rows  
3) $n$ is the number of columns  

### Things to Note:

1) All pixel values must be between 0 and 255  
2) The image matrix must be well-formed (all rows same length)  
3) Empty or invalid images return -1

### Verified Pure NumPy Solution (From Scratch)
```python
def calculate_brightness(img):
    # Check if image is empty or has no columns
    if not img or not img[0]:
        return -1

    rows, cols = len(img), len(img[0])

    # Check if all rows have same length and values are valid
    for row in img:
        if len(row) != cols:
            return -1
        for pixel in row:
            if not 0 <= pixel <= 255:
                return -1

    # Calculate average brightness
    total = sum(sum(row) for row in img)
    return round(total / (rows * cols), 2)
```

### Verified Sample Trace Data
```json
{
  "test": "\n# Test empty image\nprint(calculate_brightness([]))\n",
  "expected_output": "-1"
}
```

---

## Problem 82: Grayscale Image Contrast Calculator
- **Difficulty**: `easy`
- **Category**: Computer Vision

### Description & Spec
Write a Python function to calculate the contrast of a grayscale image using the difference between the maximum and minimum pixel values.

### Mathematical Derivation & Intuition
## Calculating Contrast of a Grayscale Image

Contrast in a grayscale image refers to the difference in luminance or color that makes an object distinguishable. Here are methods to calculate contrast:

### 1. Basic Contrast Calculation

The simplest way to define the contrast of a grayscale image is by using the difference between the maximum and minimum pixel values:

$$
\text{Contrast} = \max(I) - \min(I)
$$

### 2. RMS Contrast

Root Mean Square (RMS) contrast considers the standard deviation of pixel intensities:

$$
\text{RMS Contrast} = \frac{\sigma}{\mu}
$$

### 3. Michelson Contrast

Michelson contrast is defined as:

$$
C = \frac{I_{\text{max}} - I_{\text{min}}}{I_{\text{max}} + I_{\text{min}}}
$$

### Example Calculation

For a grayscale image with pixel values ranging from 50 to 200:

1. **Maximum Pixel Value**: 200  
2. **Minimum Pixel Value**: 50  
3. **Contrast Calculation**:

$$
\text{Contrast} = 200 - 50 = 150
$$

### Applications

Calculating contrast is crucial in:

- Image quality assessment
- Preprocessing in computer vision
- Enhancing visibility in images
- Object detection and analysis

### Verified Pure NumPy Solution (From Scratch)
```python
import numpy as np

def calculate_contrast(img):
    """
    Calculate the contrast of a grayscale image.
    Args:
        img (numpy.ndarray): 2D array representing a grayscale image with pixel values between 0 and 255.
    Returns:
        float: Contrast value rounded to 3 decimal places.
    """
    # Find the maximum and minimum pixel values
    max_pixel = np.max(img)
    min_pixel = np.min(img)

    # Calculate contrast
    contrast = max_pixel - min_pixel

    return round(float(contrast), 3)
```

### Verified Sample Trace Data
```json
{
  "test": "print(calculate_contrast(np.array([[0, 50], [200, 255]])))",
  "expected_output": "255"
}
```

---

