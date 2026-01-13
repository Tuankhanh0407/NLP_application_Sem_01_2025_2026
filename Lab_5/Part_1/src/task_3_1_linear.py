# Import necessary libraries.
import torch
from torch import nn as tnn
from typing import Tuple

class LinearDemo:
    """ 
    Demonstrate nn.Linear usage: y = x * (A ^ T) + b.

    Attributes:
    - layer (tnn.Linear): The linear transformation layer from 5 input features to 2 outputs. 
    """
    def __init__(self) -> None:
        """ 
        Initialize the LinearDemo class. 
        """
        self.layer: tnn.Linear = tnn.Linear(in_features = 5, out_features = 2)

    def forward_random_input(self, batch_shape: Tuple[int, int]) -> torch.Tensor:
        """ 
        Generate random input and pass through the linear layer.
        @param batch_shape (Tuple[int, int]): The (batch_size, input_dim) shape for random input, should be (3, 5).
        @return (torch.Tensor): The output tensor after linear transformation. 
        """
        input_tensor = torch.randn(*batch_shape)
        output = self.layer(input_tensor)
        print(f"Input shape: {tuple(input_tensor.shape)}")
        print(f"Output shape: {tuple(output.shape)}")
        print(f"Output:\n{output}")
        return output
    
def main() -> None:
    """ 
    Execute the task 3.1 demonstration: nn.Linear forward pass. 
    """
    demo = LinearDemo()
    _ = demo.forward_random_input((3, 5))

if __name__ == "__main__":
    main()