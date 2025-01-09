import torch
import torch.nn as nn


class RNN(nn.Module):
    """
    A multi-layer vanilla RNN cell:
      - Each layer has learned i2h (input->hidden) and h2h (hidden->hidden) weights.
      - We apply tanh, LayerNorm, and dropout per layer.
      - The final layer also produces some output logits (size = output_size).
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        output_size: int,
        dropout_prob: float = 0.1,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Each layer gets an i2h, h2h, and per-layer norm+dropout
        self.i2h = nn.ModuleList()
        self.h2h = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        for layer_idx in range(num_layers):
            if layer_idx == 0:
                curr_input_size = input_size
            else:
                curr_input_size = hidden_size

            self.i2h.append(nn.Linear(curr_input_size, hidden_size))
            self.h2h.append(nn.Linear(hidden_size, hidden_size))

            self.norms.append(nn.LayerNorm(hidden_size))
            self.dropouts.append(nn.Dropout(dropout_prob))

        # Output layer from the top hidden state
        self.h2o = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor, hidden_states: torch.Tensor):
        """
        Args:
          x: (batch_size, input_size) at the current time step
          hidden_states: (num_layers, batch_size, hidden_size)

        Returns:
          logits: (batch_size, output_size)
          new_hidden_states: (num_layers, batch_size, hidden_size)
        """
        new_hidden_states = []

        # The "current_input" to layer 0 is 'x'
        current_input = x

        for layer_idx in range(self.num_layers):
            # Previous hidden state for this layer
            h_prev = hidden_states[layer_idx]  # (batch_size, hidden_size)

            # Compute next hidden state
            h_in = self.i2h[layer_idx](current_input) + self.h2h[layer_idx](h_prev)
            h_out = torch.tanh(h_in)

            # LayerNorm + Dropout
            h_out = self.norms[layer_idx](h_out)
            h_out = self.dropouts[layer_idx](h_out)

            current_input = h_out
            new_hidden_states.append(h_out)

        new_hidden_states = torch.stack(new_hidden_states, dim=0)

        # The final layer's hidden state is "current_input"
        logits = self.h2o(current_input)  # (batch_size, output_size)

        return logits, new_hidden_states


class Encoder(nn.Module):
    """
    An encoder that processes (batch_size, 512) token IDs with a custom multi-layer RNN cell.
    """

    def __init__(self, vocab_size, embed_dim, hidden_size, num_layers, dropout_prob):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Embed the tokens
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        # Our custom multi-layer RNN cell
        # - input_size=embed_dim
        # - output_size can be anything, but we won't really use it in the encoder
        #   so let's set output_size=hidden_size (or 1 if we truly don’t care).
        self.rnn_cell = MultiLayerVanillaRNN(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            output_size=hidden_size,  # for convenience
            dropout_prob=dropout_prob,
        )

    def forward(self, src, hidden=None):
        """
        src: (batch_size, 512) of integer token IDs
        hidden: Optional initial hidden state of shape (num_layers, batch_size, hidden_size)

        Returns:
          outputs: (batch_size, 512, hidden_size) top-layer hidden states at each step
          hidden:  (num_layers, batch_size, hidden_size) final hidden states
        """
        if hidden is None:
            # Initialize hidden to zeros
            batch_size = src.size(0)
            hidden = torch.zeros(
                self.num_layers, batch_size, self.hidden_size, device=src.device
            )

        batch_size, seq_len = src.shape  # seq_len should be 512

        # Embedding => (batch_size, 512, embed_dim)
        embedded = self.embedding(src)

        outputs = []
        for t in range(seq_len):
            # Extract the t-th token embedding => (batch_size, embed_dim)
            x_t = embedded[:, t, :]

            # Pass through our custom RNN cell
            # logits: (batch_size, hidden_size)  [since output_size=hidden_size in the encoder]
            # hidden: (num_layers, batch_size, hidden_size)
            logits, hidden = self.rnn_cell(x_t, hidden)

            # We'll store the top hidden layer. (hidden[-1] is the top layer’s hidden state.)
            # But "logits" itself here is effectively from self.h2o. We can store whichever is more convenient.
            outputs.append(hidden[-1])

        # Stack all steps => (batch_size, seq_len, hidden_size)
        outputs = torch.stack(outputs, dim=1)

        return outputs, hidden


class Decoder(nn.Module):
    """
    A decoder that generates a summary token-by-token using the custom RNN cell.
    """

    def __init__(self, vocab_size, embed_dim, hidden_size, num_layers, dropout_prob):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Embedding for target tokens
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        # Our custom multi-layer RNN cell
        # - input_size=embed_dim
        # - output_size=vocab_size (since we want logits over the summary vocabulary)
        self.rnn_cell = MultiLayerVanillaRNN(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            output_size=vocab_size,
            dropout_prob=dropout_prob,
        )

    def forward(self, input_step, hidden):
        """
        input_step: (batch_size,) containing the current token IDs to decode
        hidden: (num_layers, batch_size, hidden_size)

        Returns:
          logits: (batch_size, vocab_size) raw logits for the next token
          hidden: updated hidden state
        """
        # Expand (batch_size,) -> (batch_size, 1), embed => (batch_size, 1, embed_dim)
        input_step = input_step.unsqueeze(1)
        embedded = self.embedding(input_step)  # => (batch_size, 1, embed_dim)

        # Squeeze back to (batch_size, embed_dim)
        embedded = embedded.squeeze(1)

        # Pass through custom RNN cell
        # logits: (batch_size, vocab_size)
        # hidden: (num_layers, batch_size, hidden_size)
        logits, hidden = self.rnn_cell(embedded, hidden)

        return logits, hidden


class Seq2Seq(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, device: torch.device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, tgt, teacher_forcing_ratio=0.5):
        """
        src: (batch_size, 512)  => news articles tokens
        tgt: (batch_size, tgt_len) => reference summary tokens
        teacher_forcing_ratio: float in [0..1]

        Returns:
          outputs: (batch_size, tgt_len-1, vocab_size)
                   raw logits at each decoder step
        """
        batch_size, tgt_len = tgt.shape

        # 1) Encode
        #  - we only need the final hidden state for initialization
        #  - outputs can be used for attention if you implement it
        _, encoder_hidden = self.encoder(src)
        # encoder_hidden => (num_layers, batch_size, hidden_size)

        # 2) Initialize decoder hidden with encoder's final hidden
        decoder_hidden = encoder_hidden

        # We'll store decoder outputs at each time step
        outputs = []

        # Typically, tgt[:, 0] is an <SOS> token
        decoder_input = tgt[:, 0]  # shape: (batch_size,)

        # 3) Unroll the decoder for each time step
        for t in range(1, tgt_len):
            # Pass the current token + hidden state to the decoder
            logits, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
            # logits: (batch_size, vocab_size)

            # Store
            outputs.append(logits.unsqueeze(1))  # => (batch_size, 1, vocab_size)

            # Decide whether to use teacher forcing
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            next_token = tgt[:, t] if teacher_force else logits.argmax(dim=1)

            # Next input
            decoder_input = next_token

        # Concatenate along time: (batch_size, tgt_len-1, vocab_size)
        outputs = torch.cat(outputs, dim=1)

        return outputs
