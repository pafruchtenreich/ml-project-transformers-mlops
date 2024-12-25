import torch
import torch.nn.functional as F
from torch import nn


class AttentionLayer(nn.Module):
    """
    Multi-head attention mechanism class.

    This layer computes attention scores between query (q), key (k), and value (v) tensors,
    then applies a linear projection to the output.
    """

    def __init__(self, hidden_size, n_head):
        """
        Initialize the AttentionLayer.

        Args:
            hidden_size (int): The dimensionality of the hidden representations.
            n_head (int): The number of attention heads.
        """
        super(AttentionLayer, self).__init__()
        self.hidden_size = hidden_size
        self.n_head = n_head
        self.w_q = nn.Linear(hidden_size, hidden_size)
        self.w_k = nn.Linear(hidden_size, hidden_size)
        self.w_v = nn.Linear(hidden_size, hidden_size)
        self.w_o = nn.Linear(hidden_size, hidden_size)

    def forward(self, q, k, v, mask=None):
        """
        Forward pass for the multi-head attention layer.

        Args:
            q (torch.Tensor): Query tensor of shape (batch_size, seq_len, hidden_size).
            k (torch.Tensor): Key tensor of shape (batch_size, seq_len, hidden_size).
            v (torch.Tensor): Value tensor of shape (batch_size, seq_len, hidden_size).
            mask (torch.Tensor, optional): Attention mask to prevent attention to certain positions.

        Returns:
            torch.Tensor: The output of the attention layer of shape (batch_size, seq_len, hidden_size).
        """
        batch_size, length, _ = q.shape
        # apply projections
        q, k, v = self.w_q(q), self.w_k(v), self.w_v(v)

        d_head = self.hidden_size // self.n_head
        # reshape into (batch_size, n_head, seq_len, d_head)
        q = q.view(batch_size, length, self.n_head, d_head).transpose(1, 2)
        k = k.view(batch_size, length, self.n_head, d_head).transpose(1, 2)
        v = v.view(batch_size, length, self.n_head, d_head).transpose(1, 2)

        # compute attention scores
        kt = k.transpose(2, 3)
        y = q @ kt
        y = y * ((d_head) ** -0.5)  # scale by sqrt(d_head)

        # apply mask if provided
        if mask is not None:
            y = y - 10000 * (mask == 0)

        # softmax over the last dimension
        y = F.softmax(y, dim=-1)

        # multiply by values
        y = y @ v

        # restore shape
        y = y.transpose(1, 2).contiguous()
        y = y.view(batch_size, length, self.hidden_size)

        # final linear projection
        y = self.w_o(y)
        return y


class EncoderLayer(nn.Module):
    """
    A single Transformer encoder layer.

    This layer includes multi-head self-attention and a feedforward network.
    """

    def __init__(self, hidden_size, ffn_hidden, n_head, drop_prob):
        """
        Initialize an EncoderLayer.

        Args:
            hidden_size (int): The dimensionality of the hidden representations.
            ffn_hidden (int): The hidden size of the feed-forward network.
            n_head (int): The number of attention heads.
            drop_prob (float): Probability of dropout.
        """
        super(EncoderLayer, self).__init__()
        self.attention = AttentionLayer(hidden_size=hidden_size, n_head=n_head)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.dropout1 = nn.Dropout(p=drop_prob)

        self.linear1 = nn.Linear(hidden_size, ffn_hidden)
        self.linear2 = nn.Linear(ffn_hidden, hidden_size)
        self.dropout2 = nn.Dropout(p=drop_prob)

        self.norm2 = nn.LayerNorm(hidden_size)
        self.dropout3 = nn.Dropout(p=drop_prob)

    def forward(self, x, src_mask):
        """
        Forward pass for the encoder layer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, hidden_size).
            src_mask (torch.Tensor): Attention mask for the source sequence.

        Returns:
            torch.Tensor: Output of the encoder layer of shape (batch_size, seq_len, hidden_size).
        """
        # Self-attention
        _x = x
        x = self.attention(q=x, k=x, v=x, mask=src_mask)
        x = self.dropout1(x)
        x = self.norm1(x + _x)

        # Feed-forward network
        _x = x
        x = self.linear1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.linear2(x)
        x = self.dropout3(x)
        x = self.norm2(x + _x)
        return x


class DecoderLayer(nn.Module):
    """
    A single Transformer decoder layer.

    This layer includes self-attention, encoder-decoder attention, and a feedforward network.
    """

    def __init__(self, hidden_size, ffn_hidden, n_head, drop_prob):
        """
        Initialize a DecoderLayer.

        Args:
            hidden_size (int): The dimensionality of the hidden representations.
            ffn_hidden (int): The hidden size of the feed-forward network.
            n_head (int): The number of attention heads.
            drop_prob (float): Probability of dropout.
        """
        super(DecoderLayer, self).__init__()
        self.self_attention = AttentionLayer(hidden_size=hidden_size, n_head=n_head)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.dropout1 = nn.Dropout(p=drop_prob)

        self.enc_dec_attention = AttentionLayer(hidden_size=hidden_size, n_head=n_head)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.dropout2 = nn.Dropout(p=drop_prob)

        self.linear1 = nn.Linear(hidden_size, ffn_hidden)
        self.linear2 = nn.Linear(ffn_hidden, hidden_size)
        self.dropout3 = nn.Dropout(p=drop_prob)

        self.norm3 = nn.LayerNorm(hidden_size)
        self.dropout4 = nn.Dropout(p=drop_prob)

    def forward(self, dec, enc, trg_mask, src_mask):
        """
        Forward pass for the decoder layer.

        Args:
            dec (torch.Tensor): Decoder input tensor of shape (batch_size, seq_len, hidden_size).
            enc (torch.Tensor): Encoder output tensor of shape (batch_size, seq_len, hidden_size).
            trg_mask (torch.Tensor): Attention mask for the target sequence.
            src_mask (torch.Tensor): Attention mask for the source sequence.

        Returns:
            torch.Tensor: Output of the decoder layer of shape (batch_size, seq_len, hidden_size).
        """
        # Self-attention (decoder)
        _x = dec
        x = self.self_attention(q=dec, k=dec, v=dec, mask=trg_mask)
        x = self.dropout1(x)
        x = self.norm1(x + _x)

        # Encoder-decoder attention
        if enc is not None:
            _x = x
            x = self.enc_dec_attention(q=x, k=enc, v=enc, mask=src_mask)
            x = self.dropout2(x)
            x = self.norm2(x + _x)

        # Feed-forward network
        _x = x
        x = self.linear1(x)
        x = F.relu(x)
        x = self.dropout3(x)
        x = self.linear2(x)
        x = self.dropout4(x)
        x = self.norm3(x + _x)
        return x


class PositionalEncoding(nn.Module):
    """
    Positional encoding for Transformer models.

    This module computes sine and cosine positional embeddings for each position
    in the sequence, which are then added to token embeddings.
    """

    def __init__(self, hidden_size, max_len):
        """
        Initialize PositionalEncoding.

        Args:
            hidden_size (int): The dimensionality of the hidden representations.
            max_len (int): Maximum length of the input sequence.
        """
        super(PositionalEncoding, self).__init__()
        # 'device' should be defined outside or passed in; assuming it is defined globally
        self.encoding = torch.zeros(max_len, hidden_size, device=device)
        self.encoding.requires_grad = False
        pos = torch.arange(0, max_len, device=device).float().unsqueeze(dim=1)
        _2i = torch.arange(0, hidden_size, step=2, device=device).float()
        self.encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / hidden_size)))
        self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / hidden_size)))

    def forward(self, x):
        """
        Forward pass for positional encoding.

        Args:
            x (torch.Tensor): A tensor indicating the sequence input shape,
                              usually (batch_size, seq_len).

        Returns:
            torch.Tensor: The positional encoding of shape (seq_len, hidden_size).
        """
        batch_size, seq_len = x.size()
        return self.encoding[:seq_len, :]


class TransformerEmbedding(nn.Module):
    """
    Embedding module that combines token embeddings with positional encodings.
    """

    def __init__(self, vocab_size, hidden_size, max_len, drop_prob):
        """
        Initialize TransformerEmbedding.

        Args:
            vocab_size (int): Size of the vocabulary.
            hidden_size (int): The dimensionality of the hidden representations.
            max_len (int): Maximum length of the input sequence.
            drop_prob (float): Probability of dropout.
        """
        super(TransformerEmbedding, self).__init__()
        self.tok_emb = nn.Embedding(vocab_size, hidden_size, padding_idx=1)
        self.pos_emb = PositionalEncoding(hidden_size, max_len)
        self.drop_out = nn.Dropout(p=drop_prob)

    def forward(self, x):
        """
        Forward pass to generate combined embeddings.

        Args:
            x (torch.Tensor): Input token IDs of shape (batch_size, seq_len).

        Returns:
            torch.Tensor: Token embedding plus positional encoding, of shape (batch_size, seq_len, hidden_size).
        """
        tok_emb = self.tok_emb(x)
        pos_emb = self.pos_emb(x)
        return self.drop_out(tok_emb + pos_emb)


class Transformer(nn.Module):
    """
    A Transformer model consisting of an encoder and a decoder.

    This model handles source and target inputs, applies attention, and outputs
    logits over the vocabulary for each position in the target sequence.
    """

    def __init__(
        self,
        pad_idx,
        voc_size,
        hidden_size,
        n_head,
        max_len,
        dec_max_len,
        ffn_hidden,
        n_layers,
        drop_prob=0.1,
    ):
        """
        Initialize the Transformer.

        Args:
            pad_idx (int): The padding token index.
            voc_size (int): The size of the vocabulary.
            hidden_size (int): The dimensionality of the hidden representations.
            n_head (int): The number of attention heads.
            max_len (int): Maximum length for the encoder.
            dec_max_len (int): Maximum length for the decoder (unused in this shared embedding example).
            ffn_hidden (int): Hidden size in the feed-forward networks.
            n_layers (int): Number of encoder and decoder layers.
            drop_prob (float): Probability of dropout.
        """
        super().__init__()
        self.pad_idx = pad_idx
        self.embedding = TransformerEmbedding(
            hidden_size=hidden_size,
            max_len=max_len,
            vocab_size=voc_size,
            drop_prob=drop_prob,
        )
        self.encoder_layers = nn.ModuleList(
            [
                EncoderLayer(
                    hidden_size=hidden_size,
                    ffn_hidden=ffn_hidden,
                    n_head=n_head,
                    drop_prob=drop_prob,
                )
                for _ in range(n_layers)
            ]
        )
        self.decoder_layers = nn.ModuleList(
            [
                DecoderLayer(
                    hidden_size=hidden_size,
                    ffn_hidden=ffn_hidden,
                    n_head=n_head,
                    drop_prob=drop_prob,
                )
                for _ in range(n_layers)
            ]
        )
        self.linear = nn.Linear(hidden_size, voc_size)

    def forward(self, src, trg):
        """
        Forward pass for the Transformer.

        Args:
            src (torch.Tensor): Source token IDs of shape (batch_size, src_seq_len).
            trg (torch.Tensor): Target token IDs of shape (batch_size, trg_seq_len).

        Returns:
            torch.Tensor: Logits over the vocabulary for each position in the target sequence,
                          of shape (batch_size, trg_seq_len, voc_size).
        """
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)

        enc_src = self.embedding(src)
        for layer in self.encoder_layers:
            enc_src = layer(enc_src, src_mask)

        out = self.embedding(trg)
        for layer in self.decoder_layers:
            out = layer(out, enc_src, trg_mask, src_mask)

        out = self.linear(out)
        return out

    def make_src_mask(self, src):
        """
        Create a masking tensor for the source sequence to avoid attending to padding tokens.

        Args:
            src (torch.Tensor): Source token IDs of shape (batch_size, src_seq_len).

        Returns:
            torch.Tensor: A mask of shape (batch_size, 1, 1, src_seq_len).
        """
        src_mask = (src != self.pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask

    def make_trg_mask(self, trg):
        """
        Create a masking tensor for the target sequence to avoid attending to padding tokens
        and to maintain auto-regressive property (no access to future tokens).

        Args:
            trg (torch.Tensor): Target token IDs of shape (batch_size, trg_seq_len).

        Returns:
            torch.Tensor: A mask of shape (batch_size, 1, trg_seq_len, trg_seq_len).
        """
        trg_pad_mask = (trg != self.pad_idx).unsqueeze(1).unsqueeze(3)
        trg_len = trg.shape[1]
        trg_sub_mask = (
            torch.tril(torch.ones(trg_len, trg_len)).type(torch.ByteTensor).to(device)
        )
        trg_mask = trg_pad_mask & trg_sub_mask
        return trg_mask
