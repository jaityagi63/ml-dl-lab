## **Tensor:** 
Tensors represent the fundamental data structure in machine learning and deep learning applications.

- A tensor is simply a **multidimensional array** used to store data
- **Scaler**(0-dim tensors): Single numerical value 
- **Vectors**(1-dim tensors):  Arrays of numbers with directional properties
- **Matrices**(2-dim tensors):  Rectangular arrays for linear transformations
- **Higher-order tensors**(n-dim tensors): for storing complex data

## **Introduction to Tensors: Mathematical Foundations**

In the context of deep learning, tensors serve as the primary data structure for representing:

- **Input data:** Images, text sequences, audio signals, and tabular data
- **Model parameters:** weight and biases in neural network
- **Intermediate computations:** Activations and gradients during training
- **Output predictions:** Classification probabilities and regression values

## **Mathematical Definition**

A tensor **T** of rank **n** (or order **n**) is a mathematical object with **n** indices, where each index can range over a specific dimension. Formally:   
**T** ∈ ℝᵈ¹ˣᵈ²ˣ...ˣᵈⁿ    
Where:
- **d₁, d₂, ..., dₙ** represent the size of each dimension
- **n** is the rank/order of the tensor
- **ℝ** denotes the set of real numbers (though tensors can contain other data types)

### **Computational Significance**
Tensors enable efficient computation through:

1. **Vectorization**: Operations on entire arrays rather than individual elements
2. **Parallelization**: GPU acceleration for simultaneous computations
3. **Automatic Differentiation**: Gradient computation for optimization algorithms
4. **Memory Efficiency**: Optimized storage and access patterns

## **Data Representation Through Tensors**    
**Tensors provide a unified mathematical framework for representing diverse data types in numerical form**
- ### **Image Representation Example :**
    Consider a color image representation as a 3-dimensional tensor with shape  `[channels, height, width]`:
    - **Channels (C)**: Color information (typically 3 for RGB: Red, Green, Blue)
    - **Height (H)**: Vertical resolution in pixels  
    - **Width (W)**: Horizontal resolution in pixels

    For example, a standard image might have shape `[3, 224, 224]`, representing:
    - **3 color channels** (RGB)
    - **224 pixels** in height
    - **224 pixels** in width
    - **Total elements**: 3 × 224 × 224 = 150,528 numerical values

![Tensor representation of an image showing the breakdown into color channels and spatial dimensions](https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/images/00-tensor-shape-example-of-image.png)

## **Data Type Implications**
The choice of tensor dimensions and data organization significantly impacts:
- **Memory consumption**: Higher dimensions require more storage
- **Computational complexity**: More dimensions increase operation costs
- **Model architecture design**: Network layers must match tensor shapes
- **Training efficiency**: Optimal batch sizes depend on tensor dimensions

## **Tensor Creation and Initialization Methods**
Understanding tensor creation methods is crucial for:
- **Data preprocessing**: Converting raw data into tensor format
- **Model initialization**: Creating parameter tensors with appropriate shapes
- **Experimentation**: Generating synthetic data for testing and valid

### **Scalar Tensors (Rank 0)**

A **scalar** represents a single numerical value and constitutes a **zero-dimensional tensor**. In mathematical notation:
- **Mathematical representation**: *s* ∈ ℝ
- **PyTorch shape**: `torch.Size([])`
- **Dimensions**: 0

### **Properties:**
- Contains exactly one element
- No directional information
- Often used for loss values, learning rates, and single metrics

```python
import torch

# They are the simplest type of tensor and are often used to represent loss or an accuracy score.

scalar = torch.tensor(10)

print(f"Scalar tensor: {scalar}")
print(f"Scalar value: {scalar.item()}")   # Convert tensor to a Python number
print(f"Data type: {scalar.dtype}")       
print(f"Shape: {scalar.shape}")           
print(f"Size: {scalar.size()}")           

# Tensor metadata:
# - dtype: Data type (e.g., int64, float32)
# - device: CPU or GPU where the tensor is stored
# - requires_grad: Whether PyTorch tracks operations for backpropagation
scalar
``` 
### **Tensor dimensionality** 

**ndim** indicates the number of indices required to specify an element within the tensor.

###  **Value Extraction from Tensors**