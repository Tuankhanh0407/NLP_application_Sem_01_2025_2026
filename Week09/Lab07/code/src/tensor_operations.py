""" 
(i) Problem description:
Implement tensor creation, operations, indexing, slicing, and reshaping operations using PyTorch.

Input:
- Various data sources (list, NumPy arrays) and tensor shapes.

Output:
- Created tensors and results of operations.

(ii) Approach:
Use PyTorch tensor creation methods and operations to manipulate tensor data structures.
"""

# Import necessary libraries
import numpy as np 
import torch 

class TensorOperations:
    """ 
    Implement various tensor operations including creation, manipulation, and transformation. 
    """

    def demo_tensor_creation(self):
        """ 
        Demonstrate different methods of tensor creation. 
        """
        print("\n--- Task 1.1: Tensor creation ---")

        # Creation tensor from list 
        data = [[1, 2], [3, 4]]
        x_data = torch.tensor(data)
        print(f"Tensor from list:\n{x_data}\n")

        # Create tensor from NumPy array 
        np_array = np.array(data)
        x_np = torch.from_numpy(np_array)
        print(f"Tensor from NumPy array:\n{x_np}\n")

        # Create tensors with specific values
        x_ones = torch.ones_like(x_data)
        print(f"Ones tensor:\n{x_ones}\n")

        x_rand = torch.rand_like(x_data, dtype = torch.float)
        print(f"Random tensor:\n{x_rand}\n")

        # Print tensor properties
        print(f"Tensor shape: {x_rand.shape}")
        print(f"Tensor datatype: {x_rand.dtype}")
        print(f"Tensor device: {x_rand.device}")

    def demo_tensor_operations(self):
        """ 
        Demonstrate various tensor operations. 
        """
        print("\n--- Task 1.2: Tensor operations ---")

        data = [[1, 2], [3, 4]]
        x_data = torch.tensor(data, dtype = torch.float)

        # Addition
        addition_result = x_data + x_data 
        print(f"Addition (x_data + x_data):\n{addition_result}\n")

        # Multiplication with scalar 
        scalar_multiplication = x_data * 5 
        print(f"Scalar multiplication (x_data * 5):\n{scalar_multiplication}\n")

        # Matrix multiplication
        matrix_multiplication = x_data @ x_data.T 
        print(f"Matrix multiplication (x_data @ x_data.T):\n{matrix_multiplication}\n")

    def demo_indexing_slicing(self):
        """ 
        Demonstrate tensor indexing and slicing operations.
        @return (Tuple): Contain first row, second column, and specific element. 
        """
        print("\n--- Task 1.3: Indexing and slicing ---")

        data = [[1, 2], [3, 4]]
        x_data = torch.tensor(data, dtype = torch.float)

        # First row
        first_row = x_data[0, :]
        print(f"First row: {first_row}")

        # Second column
        second_column = x_data[:, 1]
        print(f"Second column: {second_column}")

        # Specific element 
        specific_element = x_data[1, 1]
        print(f"Element at [1, 1]: {specific_element}")

        return first_row, second_column, specific_element 
    
    def demo_reshape_operations(self):
        """ 
        Demonstrate tensor reshaping operations.
        @return reshaped tensor. 
        """
        print("\n--- Task 1.4: Tensor reshaping ---")

        # Create 4 x 4 random tensor
        original_tensor = torch.rand(4, 4)
        print(f"Original tensor shape (4, 4): {original_tensor.shape}")
        print(f"Original tensor:\n{original_tensor}\n")

        # Reshape to 16 x 1
        reshaped_tensor = original_tensor.view(16, 1)
        print(f"Reshaped tensor shape (16, 1): {reshaped_tensor.shape}")
        print(f"Reshaped tensor:\n{reshaped_tensor}")

        return reshaped_tensor