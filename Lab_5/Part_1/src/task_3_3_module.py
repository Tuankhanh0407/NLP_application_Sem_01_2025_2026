# Import necessary libraries.
import torch
from torch import nn as tnn
from typing import List

class MyFirstModel(tnn.Module):
    """ 
    A simple neural network combining Embedding, Linear, ReLU, and Output Linear.

    Attributes:
    - embedding (tnn.Embedding): Embedding layer mapping vocab indices to vectors of size embedding_dim.
    - linear (tnn.Linear): Linear layer projecting embedding_dim to hidden_dim.
    - activation (tnn.ReLU): Non-linear activation function applied after the first linear layer.
    - output_layer (tnn.Linear): Final linear layer projecting hidden_dim to output_dim. 
    """
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int, output_dim: int) -> None:
        """ 
        Initialize the MyFirstModel class.
        @param vocab_size (int): Number of distinct tokens in the vocabulary.
        @param embedding_dim (int): Dimension of each token embedding vector.
        @param hidden_dim (int): Size of the hidden representation.
        @param output_dim (int): Number of output units. 
        """
        super(MyFirstModel, self).__init__()
        self.embedding = tnn.Embedding(vocab_size, embedding_dim)
        self.linear = tnn.Linear(embedding_dim, hidden_dim)
        self.activation = tnn.ReLU()
        self.output_layer = tnn.Linear(hidden_dim, output_dim)

    def forward(self, indices: torch.LongTensor) -> torch.Tensor:
        """ 
        Define forward pass: Embedding -> Linear -> ReLU -> Output Linear.
        @param indices (torch.LongTensor): A 1D tensor of token indices representing a sentence.
        @return (torch.Tensor): The output tensor (sequence_length, output_dim). 
        """
        embeds = self.embedding(indices)
        hidden = self.activation(self.linear(embeds))
        output = self.output_layer(hidden)
        return output
    
class ModuleDemo:
    """ 
    A wrapper class to instantiate and run the MyFirstModel class with sample input.
    
    Attributes:
    - model (MyFirstModel): The instantiated toy neural network model.
    """
    def __init__(self) -> None:
        """ 
        Initialize the ModuleDemo class. 
        """
        self.model = MyFirstModel(vocab_size = 100, embedding_dim = 16, hidden_dim = 8, output_dim = 2)

    def run_with_input(self, input_indices: List[int]) -> torch.Tensor:
        """ 
        Convert indices to tensor and run the model forward pass, printing shapes.
        @param input_indices (List[int]): A list of token indices representing a sentence (in example, [1, 2, 5, 9]).
        @return (torch.Tensor): The output tensor from the model forward pass.
        """
        input_tensor = torch.LongTensor(input_indices)
        output = self.model(input_tensor)
        print(f"Model output shape: {tuple(output.shape)}")
        print(f"Output:\n{output}")
        return output
    
def main() -> None:
    """ 
    Execute the task 3.3 demonstration: Define a module and run a sample forward pass. 
    """
    demo = ModuleDemo()
    _ = demo.run_with_input([1, 2, 5, 9])

if __name__ == "__main__":
    main()