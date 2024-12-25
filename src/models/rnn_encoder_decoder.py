import torch
import torch.nn as nn


class RNN(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        dropout_prob: float,
    ):
        """
        RNN with 3 hidden layers, dropout, and LayerNorm.
        """
        super().__init__()
        self.hidden_size = hidden_size

        # 1) Input -> first hidden layer
        self.i2h = nn.Linear(input_size, hidden_size)

        # 2) Second hidden layer
        self.h2h1 = nn.Linear(hidden_size, hidden_size)

        # 3) Third hidden layer
        self.h2h2 = nn.Linear(hidden_size, hidden_size)

        # Output layer
        self.h2o = nn.Linear(hidden_size, output_size)

        # LayerNorm
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.norm3 = nn.LayerNorm(hidden_size)

        # Dropout
        self.dropout1 = nn.Dropout(p=dropout_prob)
        self.dropout2 = nn.Dropout(p=dropout_prob)
        self.dropout3 = nn.Dropout(p=dropout_prob)

    def forward(self, x: torch.Tensor, hidden_states: torch.Tensor):
        """
        Args:
            x (torch.Tensor): Input of shape (batch_size, input_size).
            hidden_states (torch.Tensor): Hidden states of shape (num_layers, batch_size, hidden_size).

        Returns:
            output (torch.Tensor): Output of shape (batch_size, output_size).
            hidden_states (torch.Tensor): Updated hidden states of shape (num_layers, batch_size, hidden_size).
        """
        # --- 1st layer ---
        hidden_states[0] = torch.tanh(self.i2h(x) + hidden_states[0])
        hidden_states[0] = self.dropout1(self.norm1(hidden_states[0]))

        # --- 2nd layer ---
        hidden_states[1] = torch.tanh(self.h2h1(hidden_states[0]) + hidden_states[1])
        hidden_states[1] = self.dropout2(self.norm2(hidden_states[1]))

        # --- 3rd layer ---
        hidden_states[2] = torch.tanh(self.h2h2(hidden_states[1]) + hidden_states[2])
        hidden_states[2] = self.dropout3(self.norm3(hidden_states[2]))

        # --- Output layer ---
        output = self.h2o(hidden_states[2])
        output = torch.softmax(output, dim=-1)

        return output, hidden_states
