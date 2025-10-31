""" 
(i) Problem description:
Demonstrate PyTorch's automatic gradient computation (autograd) functionality.

Input:
- Tensors with requires_grad set to True.

Output:
- Computed gradients and gradient functions.

(ii) Approach:
Use PyTorch's autograd system to automatically compute gradients and demonstrate the computation graph functionality.
"""

# Import necessary libraries
import torch 

class AutogradDemo:
    """ 
    Demonstrate PyTorch's automatic gradient computation. 
    """

    def demo_basic_autograd(self):
        """ 
        Demonstrate basic autograd functionality.
        @return (Tuple): Contain input tensor and computed gradient. 
        """
        print("\n--- Task 2.1: Basic autograd ---")

        # Create tensor with gradient tracking 
        x = torch.ones(1, requires_grad = True)
        print(f"Input tensor x: {x}")

        # Perform operations
        y = x + 2 
        print(f"y = x + 2: {y}")
        print(f"y.grad_fn: {y.grad_fn}")

        # More operations
        z = y * y * 3
        print(f"z = y * y * 3: {z}")

        # Compute gradient
        z.backward()
        print(f"Gradient dz/dx: {x.grad}")

        # Mathematical verification
        # z = 3 * (x + 2) ^ 2, dz / dx = 6 * (x + 2) = 6 * (1 + 2) = 18
        expected_grad = 6 * (x + 2)
        print(f"Expected gradient: {expected_grad}")

        return x, x.grad 
    
    def demo_multiple_backward(self):
        """ 
        Demonstrate what happens when backward() is called multiple times.
        @return gradient after second backward call. 
        """
        print("\n--- Multiple backward calls investigation ---")

        # Create new tensor for demonstration
        x = torch.ones(1, requires_grad = True)
        y = x + 2
        z = y * y * 3 

        # First backward call
        z.backward(retain_graph = True) # retain_graph to allow multiple backward
        first_grad = x.grad.clone()
        print(f"Gradient after first backward(): {first_grad}")

        # Second backward call 
        z.backward()
        second_grad = x.grad 
        print(f"Gradient after second backward(): {second_grad}")
        print("Note: Gradients accumulate when backward() is called multiple times!")

        return second_grad