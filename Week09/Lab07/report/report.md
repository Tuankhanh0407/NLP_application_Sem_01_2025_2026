# Introduction to PyTorch lab report

## 1. Implementation steps:

### 1.1. Project structure:
The program is organized into a modular structure with the following components:

```
root/
│
├── main.py (Entry point)
└── src/
    ├── tensor_operations.py (Tensor manipulation tasks)
    ├── autograd_demo.py (Gradient computation tasks)
    └── model_builder.py (Neural network building tasks)
```

### 1.2. Code implementation details:

#### 1.2.1. Main program:
- Serve as the entry point and orchestrate all lab tasks.
- Use a main method to call specialized classes.
- Each component is instantiated and executed independently.

#### 1.2.2. Tensor operations:
Class: `TensorOperations`
- `demo_tensor_creation()`: Create tensors from various sources (lists, NumPy arrays).
- `demo_tensor_operations()`: Perform arithmetic and matrix operations.
- `demo_indexing_slicing()`: Demonstrate tensor indexing and slicing.
- `demo_reshape_operations()`: Show tensor reshaping capabilities.

#### 1.2.3. Autograd demo:
Class: `AutogradDemo`
- `demo_basic_autograd()`: Demonstrate automatic gradient computation.
- `demo_multiple_backward()`: Show gradient accumulation behavior.

#### 1.2.4. Model builder:
Class: `ModelBuilder`
- `demo_linear_layer()`: Implement linear transformation layer.
- `demo_embedding_layer()`: Create embedding lookup table.
- `demo_custom_model()`: Build complete neural network model.

## 2. Code execution tutorials:

### 2.1. Prerequisites:

```bash
pip install torch numpy
```

### 2.2. Running the program:

```bash
python main.py
```

### 2.3. Expected output structure:
The program executes sequentially through three main parts:
- Tensor operations and manipulations.
- Automatic gradient computation.
- Neural network model building.

## 3. Results analysis and explanation:

### 3.1. Part 1 - Tensor operations:

**Task 1.1:** Tensor creation

**Results:**

```
Tensor from list: [[1,2],[3,4]]
Tensor from NumPy array: [[1,2],[3,4]]
Ones tensor: [[1,1],[1,1]]
Random tensor: [[0.4253,0.3224],[0.1148,0.4028]]
```

**Explanation:**
- Successfully created tensors from Python lists and NumPy arrays.
- `one_like()` creates a tensor of ones with the same shape as input.
- `rand_like()` generates random values with the specified data type.
- All tensors are stored on CPU by default.

**Task 1.2:** Tensor operations

**Results:**

```
Addition: [[2,4],[6,8]]
Scalar multiplication: [[5,10],[15,20]]
Matrix multiplication: [[5,11],[11,25]]
```

**Explanation:**
- **Addition:** Element-wise addition of tensors.
- **Scalar multiplication:** Each element multiplied by scalar value 5.
- **Matrix multiplication:** Using `@` operator for matrix multiplication `x_data @ x_data.T`

**Task 1.3:** Indexing and scaling

**Results:**

```
First row: [1,2]
Second column: [2,4]
Element at [1,1]: 4.0
```

**Explanation:**
- **First row:** `x_data[0, :]` selects all elements in row 0.
- **Second column:** `x_data[:, 1]` selects all elements in column 1.
- **Specific element:** `x_data[1, 1]` selects element at position (1, 1).

**Task 1.4:** Tensor reshaping

**Results:**
- Original shape: (4, 4) -> Reshaped: (16, 1)

**Explanation:**
- `torch.rand(4, 4)` creates a 4 x 4 tensor with random values.
- `view(16, 1)` reshapes the tensor without copying data.
- The total number of elements (16) remains unchanged.

### 3.2. Part 2 - Autograd demo:

**Task 2.1:** Basic autograd

**Results:**

```
x: tensor([1.], requires_grad=True)
y = x + 2: tensor([3.], grad_fn=<AddBackward0>)
z = y * y * 3: tensor([27.], grad_fn=<MulBackward0>)
Gradient dz/dx: tensor([18.])
```

**Mathematical verification:**

```
z = 3 × (x + 2)²
dz/dx = 6 × (x + 2)
When x = 1: dz/dx = 6 × (1 + 2) = 18
```

**Explanation:**
- `requires_grad = True` enables gradient tracking.
- PyTorch builds computation graph automatically.
- `backward()` computes gradients using chain rule.
- Gradients are stored in `.grad` attribute.

**Multiple backward calls investigation:**

**Results:**

```
First backward(): gradient = 18
Second backward(): gradient = 36
```

**Explanation:**
- Gradients accumulate when `backward()` is called multiple times.
- `retain_graph = True` preserves computation graph for repeated backward passes.
- This behavior is useful for certain optimization algorithms.

### 3.3. Part 3 - Model building:

**Task 3.1:** Linear layer

**Results:**

```
Input shape: (3,5) → Output shape: (3,2)
```

**Explanation:**
- `nn.Linear(5, 2)` creates a linear layer mapping 5 input features to 2 output features.
- The layer applies transformation: `y = x * A^T + b`.
- Input: 3 samples with 5 features each.
- Output: 3 samples with 2 features each.

**Task 3.2:** Embedding layer

**Results:**

```
Input shape: (4,) → Output shape: (4,3)
```

**Explanation:**
- `nn.Embedding(10, 3)` creates embedding table for 10 words with 3-dimensional vectors.
- Input: Indices [1, 5, 0, 8] representing word IDs.
- Output: Corresponding 3D embedding vectors for each word.

**Task 3.3:** Custom model

**Results:**

```
Input: (1,4) → Output: (1,4,2)
```

**Model architecture:**

```
Embedding(100,16) → Linear(16,8) → ReLU() → Linear(8,2)
```

**Explanation:**
- **Embedding layer:** Map 100 vocabulary items to 16D vectors.
- **First linear layer:** Transform 16D embeddings to 8D hidden representation.
- **ReLU activation:** Introduce non-linearity.
- **Output layer:** Produce 2D output for each input token.
- The model processes sequences of 4 tokens and produces 2D outputs for each token.

## 4. References:
- **PyTorch official documentation:** https://pytorch.org/docs/stable/index.html
- **NumPy documentation:** https://numpy.org/doc/