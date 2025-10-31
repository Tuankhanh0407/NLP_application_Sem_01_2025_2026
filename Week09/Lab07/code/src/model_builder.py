""" 
(i) Problem description:
Build neural network models using PyTorch's nn.Module, including linear and embedding layers.

Input: 
- Model paramters and input data.

Output:
- Model architectures and their outputs.

(ii) Approach:
Create custom neural network models by inheriting from nn.Module and combining different layers like linear and embedding.
"""

# Import necessary libraries
import torch 
from torch import nn 

class ModelBuilder:
    """ 
    Build and demonstrate neural network models using PyTorch. 
    """

    def demo_linear_layer(self):
        """ 
        Demonstrate the usage of nn.Linear layer.
        @return (Tuple): Contain input and output tensors. 
        """
        print("\n--- Task 3.1: Linear layer ---")

        # Initialize linear layer (5 features -> 2 features)
        linear_layer = nn.Linear(in_features = 5, out_features = 2)

        # Create sample input (3 samples, 5 features each)
        input_tensor = torch.randn(3, 5)

        # Forward pass 
        output = linear_layer(input_tensor)

        print(f"Input shape: {input_tensor.shape}")
        print(f"Output shape: {output.shape}")
        print(f"Output:\n{output}")

        return input_tensor, output 
    
    def demo_embedding_layer(self):
        """ 
        Demonstrate the usage of nn.Embedding layer.
        @return (Tuple): Contain input indices and embedding vectors. 
        """
        print("\n--- Task 3.2: Embedding layer ---")

        # Initialize embedding layer (10 words, 3-dimensional embeddings)
        embedding_layer = nn.Embedding(num_embeddings = 10, embedding_dim = 3)

        # Create input indices (a sentence with 4 words)
        input_indices = torch.LongTensor([1, 5, 0, 8])

        # Get embeddings
        embeddings = embedding_layer(input_indices)

        print(f"Input shape: {input_indices.shape}")
        print(f"Output shape: {embeddings.shape}")
        print(f"Embeddings:\n{embeddings}")

        return input_indices, embeddings 
    
    def demo_custom_model(self):
        """ 
        Demonstrate building a custom model by inheriting from nn.Module.
        @return model output.         
        """
        print("\n--- Task 3.3: Custom model ---")

        class MyFirstModel(nn.Module):
            """ 
            A simple neural network model for demonstration.

            Attributes:
            - embedding (nn.Embedding): Embedding layer.
            - linear (nn.Linear): First linear layer.
            - activation (nn.ReLU): Activation function.
            - output_layer (nn.Linear): Output layer. 
            """

            def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
                """ 
                Initialize the model architecture.
                @param vocab_size (int): Size of vocabulary.
                @param embedding_dim (int): Dimension of embedding vectors.
                @param hidden_dim (int): Dimension of hidden layer.
                @param output_dim (int): Dimension of output layer. 
                """
                super(MyFirstModel, self).__init__()
                self.embedding = nn.Embedding(vocab_size, embedding_dim)
                self.linear = nn.Linear(embedding_dim, hidden_dim)
                self.activation = nn.ReLU()
                self.output_layer = nn.Linear(hidden_dim, output_dim)

            def forward(self, indices):
                """ 
                Define the forward pass of the model.
                @param indices (torch.Tensor): Input indices.
                @return (torch.Tensor): Model output.
                """
                # 1. Get embeddings
                embeds = self.embedding(indices)

                # 2. Pass through linear layer and activation
                hidden = self.activation(self.linear(embeds))

                # 3.Pass through output layer 
                output = self.output_layer(hidden)
                return output 
            
        # Initialize and test the model
        model = MyFirstModel(vocab_size = 100, embedding_dim = 16, hidden_dim = 8, output_dim = 2)
        input_data = torch.LongTensor([[1, 2, 5, 9]]) # A sequence with 4 words

        # Forward pass
        output_data = model(input_data)

        print(f"Model output shape: {output_data.shape}")
        print(f"Model output:\n{output_data}")

        return output_data 