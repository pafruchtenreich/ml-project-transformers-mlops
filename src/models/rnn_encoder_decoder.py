import torch
from torch import nn


class RNN(nn.Module):
    def __init__(
        self, input_size: int, hidden_size: int, output_size: int, num_layers: int
    ):
        """
        A simple RNN implementation with multiple layers.

        Args:
            input_size (int): Dimension of the input at each time step.
            hidden_size (int): Dimension of the hidden states.
            output_size (int): Dimension of the output.
            num_layers (int): Number of layers in the RNN.
        """
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        # Input to the first hidden layer
        self.i2h = nn.Linear(input_size, hidden_size)

        # Hidden-to-hidden connections for intermediate layers
        self.h2h_layers = nn.ModuleList(
            [nn.Linear(hidden_size, hidden_size) for _ in range(num_layers - 1)]
        )

        # Hidden-to-output layer
        self.h2o = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the RNN.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_size).
            hidden_states (torch.Tensor): Hidden states of shape (num_layers, batch_size, hidden_size).

        Returns:
            output (torch.Tensor): Output tensor of shape (batch_size, output_size).
            hidden_states (torch.Tensor): Updated hidden states of shape (num_layers, batch_size, hidden_size).
        """
        # Update the first hidden state
        hidden_states[0] = torch.tanh(self.i2h(x) + hidden_states[0])

        # Propagate through subsequent hidden layers
        for layer_index, h2h in enumerate(self.h2h_layers, start=1):
            hidden_states[layer_index] = torch.tanh(
                h2h(hidden_states[layer_index - 1]) + hidden_states[layer_index]
            )

        # Compute the output
        output = self.h2o(hidden_states[-1])
        output = torch.softmax(output, dim=-1)

        return output, hidden_states
