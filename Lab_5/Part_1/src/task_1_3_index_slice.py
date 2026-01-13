# Import necessary libraries.
import torch
from typing import List, Tuple

class TensorIndexSlice:
    """ 
    Demonstrate indexing and slicing operations on tensors.

    Attributes:
    - device (str): Compute device string where tensors are placed.
    """
    def __init__(self) -> None:
        """ 
        Initialize the TensorIndexSlice class. 
        """
        self.device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

    def to_tensor(self, data: List[List[int]]) -> torch.Tensor:
        """ 
        Convert a 2D list to a tensor on the configured device.
        @param data (List[List[int]]): Numeric 2D list.
        @return (torch.Tensor): The created tensor. 
        """
        return torch.tensor(data, device = self.device)
    
    def first_row(self, x: torch.Tensor) -> torch.Tensor:
        """ 
        Extract the first row.
        @param x (torch.Tensor): Input 2D tensor.
        @return (torch.Tensor): The first row as a 1D tensor. 
        """
        return x[0]
    
    def second_column(self, x: torch.Tensor) -> torch.Tensor:
        """ 
        Extract the second column.
        @param x (torch.Tensor): Input 2D tensor.
        @return (torch.Tensor): The second column as a 1D tensor. 
        """
        return x[:, 1]
    
    def value_at(self, x: torch.Tensor, row: int, col: int) -> torch.Tensor:
        """ 
        Extract a single value at the given row and column.
        @param x (torch.Tensor): Input 2D tensor.
        @param row (int): Row index (0-based).
        @param col (int): Column index (0-based).
        @return (torch.Tensor): The single-element tensor containing the value. 
        """
        return x[row, col]
    
def main() -> None:
    """ 
    Execute the task 1.3 demonstration: Indexing and slicing operations. 
    """
    idx = TensorIndexSlice()
    data = [[1, 2], [3, 4]]
    x = idx.to_tensor(data)
    print(f"Tensor:\n{x}\n")
    print(f"First row:\n{idx.first_row(x)}\n")
    print(f"Second column:\n{idx.second_column(x)}\n")
    print(f"Value at (row = 1, col = 1):\n{idx.value_at(x, 1, 1)}\n")

if __name__ == "__main__":
    main()