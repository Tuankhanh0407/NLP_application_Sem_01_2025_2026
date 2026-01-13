# Import necessary libraries.
import numpy as np
import torch
from typing import List, Tuple

class TensorCreator:
    """ 
    Demonstrate creating tensors from Python lists, NumPy arrays, and generating constant/random tensors, then reporting basic metadata.

    Attributes:
    - device (str): The compute device string ('cpu' or CUDA device if available) to place tensors on. 
    """
    def __init__(self) -> None:
        """ 
        Initialize the TensorCreator class. 
        """
        self.device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

    def create_from_list_and_numpy(self, data: List[List[int]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """ 
        Create tensors from a Python list and a NumPy array.
        @param data (List[List[int]]): A 2D list of integers to be converted into tensors.
        @return (Tuple[torch.Tensor, torch.Tensor]): A tuple containing tensors created from list and NumPy array respectively.
        """
        x_data = torch.tensor(data, device = self.device)
        np_array = np.array(data)
        x_np = torch.from_numpy(np_array).to(self.device)
        return x_data, x_np
    
    def create_constant_and_random_like(self, reference: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """ 
        Create a ones tensor and a random tensor with the same shape as the reference.
        @param reference (torch.Tensor): The reference tensor whose shape will be used.
        @return (Tuple[torch.Tensor, torch.Tensor]): A tuple of (ones_like_tensor, rand_like_tensor_float).
        """
        x_ones = torch.ones_like(reference, device = self.device)
        x_rand = torch.rand_like(reference, dtype = torch.float, device = self.device)
        return x_ones, x_rand
    
    def describe_tensor(self, t: torch.Tensor) -> str:
        """ 
        Produce a textual description of a tensor's shape, dtype, and device.
        @param t (torch.Tensor): The tensor to describe.
        @return (str): A formatted string describing shape, dtype, and device. 
        """
        return f"Shape: {tuple(t.shape)}, dtype: {t.dtype}, device: {t.device}"
    
def main() -> None:
    """ 
    Execute the task 1.1 demonstration: Create tensors and print details.  
    """
    creator = TensorCreator()
    data = [[1, 2], [3, 4]]
    x_data, x_np = creator.create_from_list_and_numpy(data)
    print(f"Tensor from list:\n{x_data}\n")
    print(f"Tensor from NumPy array:\n{x_np}\n")
    x_ones, x_rand = creator.create_constant_and_random_like(x_data)
    print(f"Ones tensor:\n{x_ones}\n")
    print(f"Random tensor:\n{x_rand}\n")
    print(f"Tensor metadata -> {creator.describe_tensor(x_rand)}")

if __name__ == "__main__":
    main()