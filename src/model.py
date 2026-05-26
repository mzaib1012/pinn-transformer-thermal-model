import torch
import torch.nn as nn

class TransformerPINN(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=64, output_dim=1):
        super(TransformerPINN, self).__init__()
        
        # Define a fully connected network block with Tanh activations
        # Tanh is strongly preferred for PINNs because it is infinitely differentiable,
        # which is crucial for computing smooth partial derivatives in the physics loss.
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, x):
        """
        Inputs 'x' must be a tensor containing:
        [normalized_time, normalized_height, normalized_ambient_temp, normalized_load_current]
        """
        return self.network(x)
