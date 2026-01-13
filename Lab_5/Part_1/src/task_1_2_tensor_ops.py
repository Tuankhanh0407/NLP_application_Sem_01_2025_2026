# Import necessary libraries.
import torch
from typing import List

class TensorOps:
    """ 
    Perform basic tensor arithmetic operations: Addition, scalar multiplication, and matrix multiplication.

    Attributes:
    - device (str): Selected compute device where tensors are allocated.
    """
    def __init__(self) -> None:
        """ 
        Initialize the TensorOps class. 
        """
        self.device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

    def to_tensor(self, data: List[List[int]]) -> torch.Tensor:
        """ 
        Convert a 2D list to a tensor on the configured device.
        @param data (List[List[int]]): The numeric 2D list.
        @return (torch.Tensor): The created tensor. 
        """
        return torch.tensor(data, device = self.device)
    
    def add_self(self, x: torch.Tensor) -> torch.Tensor:
        """ 
        Add a tensor to itself element-wise.
        @param x (torch.Tensor): Input tensor.
        @return (torch.Tensor): Result tensor of (x + x). 
        """
        return x + x
    
    def multiply_scalar(self, x: torch.Tensor, scalar: int) -> torch.Tensor:
        """ 
        Multiply tensor by a scalar.
        @param x (torch.Tensor): Input tensor.
        @param scaler (int): The scalar value to multiply by.
        @return (torch.Tensor): Result tensor of (x * scalar). 
        """
        return x * scalar
    
    def matmul_with_transpose(self, x: torch.Tensor) -> torch.Tensor:
        """ 
        Compute matrix multiplication x @ x.T.
        @param x (torch.Tensor): Input 2D tensor.
        @return (torch.Tensor): Result tensor of x @ x.T. 
        """
        return x @ x.T
    
def main() -> None:
    """ 
    Execute the task 1.2 demonstration: Use provided operations and print results. 
    """
    ops = TensorOps()
    data = [[1, 2], [3, 4]]
    x = ops.to_tensor(data)
    print(f"x:\n{x}\n")
    print(f"x + x:\n{ops.add_self(x)}\n")
    print(f"x * 5:\n{ops.multiply_scalar(x, 5)}\n")
    print(f"x @ x.T:\n{ops.matmul_with_transpose(x)}\n")

if __name__ == "__main__":
    main()