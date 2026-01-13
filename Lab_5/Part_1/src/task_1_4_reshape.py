# Import necessary libraries.
import torch
from typing import Tuple

class TensorReshape:
    """ 
    Demonstrate reshaping tensors using view or reshape.

    Attributes:
    - device (str): Compute device string where tensors are allocated. 
    """
    def __init__(self) -> None:
        """ 
        Initialize the TensorReshape class. 
        """
        self.device: str = "cuda" if torch.cuda.is_available() else 'cpu'

    def random_tensor(self, shape: Tuple[int, int]) -> torch.Tensor:
        """ 
        Create a random tensor with the given 2D shape.
        @param shape (Tuple[int, int]): The (rows, cols) shape of the random tensor.
        @return (torch.Tensor): A random float tensor of the requested shape. 
        """
        return torch.rand(shape, device = self.device)
    
    def reshape_to(self, x: torch.Tensor, new_shape: Tuple[int, int]) -> torch.Tensor:
        """ 
        Reshape a tensor to a new shape using reshape (safe for potential non-contiguity).
        @param x (torch.Tensor): Input tensor.
        @param new_shape (Tuple[int, int]): The target shape.
        @return (torch.Tensor): The reshaped tensor. 
        """
        return x.reshape(new_shape)
    
def main() -> None:
    """ 
    Execute the task 1.4 demonstration: Create (4, 4) random tensor and reshape to (16, 1). 
    """
    rs = TensorReshape()
    x = rs.random_tensor((4, 4))
    y = rs.reshape_to(x, (16, 1))
    print(f"Original (4, 4) random tensor:\n{x}\n")
    print(f"Reshaped to (16, 1):\n{y}\n")

if __name__ == "__main__":
    main()