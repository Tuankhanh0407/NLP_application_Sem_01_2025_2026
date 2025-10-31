""" 
(i) Problem description:
This program implements a PyTorch introduction lab covering:
- Tensor operations and manipulations.
- Automatic gradient computation with autograd.
- Building neural netword models with torch.nn.

Input:
- Various tensor creation methods and model paramters.

Output:
- Tensor operations results, gradient computations, and model outputs.

(ii) Approach:
The program is divided into three main parts:
1. Tensor exploration - creating, manipulating, and operating on tensors.
2. Autograd - understanding automatic gradient computation.
3. Model building - creating neural network models using nn.Module.
"""

# Import necessary libraries
from src.tensor_operations import TensorOperations
from src.autograd_demo import AutogradDemo
from src.model_builder import ModelBuilder

def main():
    """ 
    Main method to execute all PyTorch lab tasks. 
    """
    print("=== PyTorch introduction lab ===\n")

    print("--- Part 1: Tensor operations ---")
    # Part 1: Tensor operations
    tensor_ops = TensorOperations()
    tensor_ops.demo_tensor_creation()
    tensor_ops.demo_tensor_operations()
    tensor_ops.demo_indexing_slicing()
    tensor_ops.demo_reshape_operations()

    print("\n--- Part 2: Autograd demo ---")
    # Part 2: Autograd
    autograd_demo = AutogradDemo()
    autograd_demo.demo_basic_autograd()
    autograd_demo.demo_multiple_backward()

    print("\n--- Part 3: Model building ---")
    # Part 3: Model building
    model_builder = ModelBuilder()
    model_builder.demo_linear_layer()
    model_builder.demo_embedding_layer()
    model_builder.demo_custom_model()

if __name__ == "__main__":
    main()