# Import necessary libraries.
import torch

class AutogradDemo:
    """ 
    Demonstrate fundamental PyTorch autograd operations and gradients.

    Attributes:
    - device (str): Compute device where tensors are placed. 
    """
    def __init__(self) -> None:
        """ 
        Initialize the AutogradDemo class. 
        """
        self.device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

    def compute(self) -> torch.Tensor:
        """ 
        Build a simple computation graph: x -> y = x + 2 -> z = y * y * 3, then backprop.
        @return (torch.Tensor): The gradient of z with respect to x after backward(). 
        """
        x = torch.ones(1, requires_grad = True, device = self.device)
        print(f"x: {x}")
        y = x + 2
        print(f"y: {y}")
        print(f"grad_fn of y: {y.grad_fn}")
        z = y * y * 3
        z.backward() # Equivalent to z.backward(torch.tensor(1.0))
        print(f"dz / dx: {x.grad}")
        return x.grad
    
def main() -> None:
    demo = AutogradDemo()
    _ = demo.compute()

if __name__ == "__main__":
    main()