# Import necessary libraries.
import torch
from torch import nn as tnn
from typing import List

class EmbeddingDemo:
    """ 
    Demonstrate nn.Embedding usage for mapping token indices to vectors.

    Attributes:
    - embedding (tnn.Embedding): The embedding layer with a vocabulary size of 10 and embedding dimension of 3. 
    """
    def __init__(self) -> None:
        """ 
        Initialize the EmbeddingDemo class. 
        """
        self.embedding: tnn.Embedding = tnn.Embedding(num_embeddings = 10, embedding_dim = 3)

    def embed_indices(self, input_indices: List[int]) -> torch.Tensor:
        """ 
        Convert integer indices to a tensor and lookup embeddings.
        @param input_indices (List[int]): The list of token indices (each must be in [0, 9]).
        @return (torch.Tensor): The resulting embeddings tensor of shape (len(input_indices), 3). 
        """
        idx_tensor = torch.LongTensor(input_indices)
        embs = self.embedding(idx_tensor)
        print(f"Input shape: {tuple(idx_tensor.shape)}")
        print(f"Output shape: {tuple(embs.shape)}")
        print(f"Embeddings:\n{embs}")
        return embs
    
def main() -> None:
    """ 
    Execute the task 3.2 demonstration: nn.Embedding demo. 
    """
    demo = EmbeddingDemo()
    _ = demo.embed_indices([1, 5, 0, 8])

if __name__ == "__main__":
    main()