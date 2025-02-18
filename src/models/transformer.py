import torch
import torch.nn.functional as F
from torch import nn

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


class AttentionLayer(nn.Module):
    """Multi-head attention layer."""

    def __init__(self, hidden_size, n_head):
        """
        Initialize the AttentionLayer.

        Args:
            hidden_size (int): Dimension of the hidden state.
            n_head (int): Number of attention heads.
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
        Forward pass for attention.

        Args:
            q (Tensor): Query tensor.
            k (Tensor): Key tensor.
            v (Tensor): Value tensor.
            mask (Tensor, optional): Attention mask.

        Returns:
            Tensor: Output after attention.
        """
        batch_size, length, _ = q.shape
        # apply projections
        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)

        # split heads
        # dims: (N,h,L,d)
        d_head = self.hidden_size // self.n_head
        q = q.view(batch_size, length, self.n_head, d_head).transpose(1, 2)
        k = k.view(batch_size, k.shape[1], self.n_head, d_head).transpose(1, 2)
        v = v.view(batch_size, v.shape[1], self.n_head, d_head).transpose(1, 2)

        # calculating attention
        kt = k.transpose(2, 3)

        # (N,h,L,d)@(N,h,d,L)->(N,h,L,L)
        y = q @ kt
        y = y * ((d_head) ** (-0.5))
        if mask is not None:
            y = y - 10000 * (mask == 0)
        y = F.softmax(y, dim=-1)

        # (N,h,L,L)@(N,h,L,d)->(N,h,L,d)
        y = y @ v

        # reconcatenating heads
        y = y.transpose(1, 2).contiguous()  # (N,L,h,d)
        y = y.view(batch_size, length, self.hidden_size)
        y = self.w_o(y)
        return y


class EncoderLayer(nn.Module):
    """Transformer encoder layer."""

    def __init__(self, hidden_size, ffn_hidden, n_head, drop_prob):
        """
        Initialize the EncoderLayer.

        Args:
            hidden_size (int): Dimension of the hidden state.
            ffn_hidden (int): Dimension of the feed-forward network.
            n_head (int): Number of attention heads.
            drop_prob (float): Dropout probability.
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
            x (Tensor): Input tensor.
            src_mask (Tensor): Source mask.

        Returns:
            Tensor: Output tensor.
        """
        # 1. compute self attention
        _x = x
        x = self.attention(q=x, k=x, v=x, mask=src_mask)

        x = self.dropout1(x)
        x = self.norm1(x + _x)

        # 2. feed forward network
        _x = x
        x = self.linear1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.linear2(x)

        x = self.dropout3(x)
        x = self.norm2(x + _x)
        return x


class DecoderLayer(nn.Module):
    """Transformer decoder layer."""

    def __init__(self, hidden_size, ffn_hidden, n_head, drop_prob):
        """
        Initialize the DecoderLayer.

        Args:
            hidden_size (int): Dimension of the hidden state.
            ffn_hidden (int): Dimension of the feed-forward network.
            n_head (int): Number of attention heads.
            drop_prob (float): Dropout probability.
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
            dec (Tensor): Decoder input tensor.
            enc (Tensor): Encoder output tensor.
            trg_mask (Tensor): Target mask.
            src_mask (Tensor): Source mask.

        Returns:
            Tensor: Output tensor.
        """
        # 1. compute self attention
        _x = dec
        x = self.self_attention(q=dec, k=dec, v=dec, mask=trg_mask)

        x = self.dropout1(x)
        x = self.norm1(x + _x)
        if enc is not None:
            # 2. compute encoder - decoder attention
            _x = x
            x = self.enc_dec_attention(q=x, k=enc, v=enc, mask=src_mask)
            # 4. add and norm
            x = self.dropout2(x)
            x = self.norm2(x + _x)

        # 3. simple feed forward network
        _x = x
        x = self.linear1(x)
        x = F.relu(x)
        x = self.dropout3(x)
        x = self.linear2(x)

        x = self.dropout4(x)
        x = self.norm3(x + _x)
        return x


class PositionalEncoding(nn.Module):
    """Positional encoding module."""

    def __init__(self, hidden_size, max_len):
        """
        Initialize the PositionalEncoding.

        Args:
            hidden_size (int): Dimension of the hidden state.
            max_len (int): Maximum sequence length.
        """
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_len, hidden_size, device=device)
        self.encoding.requires_grad = False
        pos = torch.arange(0, max_len, device=device)
        pos = pos.float().unsqueeze(dim=1)
        _2i = torch.arange(0, hidden_size, step=2, device=device).float()
        self.encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / hidden_size)))
        self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / hidden_size)))

    def forward(self, x):
        """
        Forward pass for positional encoding.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Positional encoded tensor.
        """
        batch_size, seq_len = x.size()
        return self.encoding[:seq_len, :]


class TransformerEmbedding(nn.Module):
    """Embedding layer with positional encoding for Transformer."""

    def __init__(self, vocab_size, hidden_size, max_len, drop_prob):
        """
        Initialize the TransformerEmbedding.

        Args:
            vocab_size (int): Size of the vocabulary.
            hidden_size (int): Dimension of the embeddings.
            max_len (int): Maximum sequence length.
            drop_prob (float): Dropout probability.
        """
        super(TransformerEmbedding, self).__init__()
        self.tok_emb = nn.Embedding(vocab_size, hidden_size, padding_idx=1)
        self.pos_emb = PositionalEncoding(hidden_size, max_len)
        self.drop_out = nn.Dropout(p=drop_prob)

    def forward(self, x):
        """
        Forward pass for embedding.

        Args:
            x (Tensor): Input tensor of token indices.

        Returns:
            Tensor: Embedded tensor with positional encoding.
        """
        tok_emb = self.tok_emb(x)
        pos_emb = self.pos_emb(x)
        return self.drop_out(tok_emb + pos_emb)


class Transformer(nn.Module):
    """Transformer model consisting of encoder and decoder."""

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
            pad_idx (int): Padding index.
            voc_size (int): Vocabulary size.
            hidden_size (int): Dimension of the hidden state.
            n_head (int): Number of attention heads.
            max_len (int): Maximum source sequence length.
            dec_max_len (int): Maximum target sequence length.
            ffn_hidden (int): Dimension of the feed-forward network.
            n_layers (int): Number of encoder and decoder layers.
            drop_prob (float, optional): Dropout probability. Defaults to 0.1.
        """
        super().__init__()
        self.pad_idx = pad_idx
        self.enc_embedding = TransformerEmbedding(
            hidden_size=hidden_size,
            max_len=max_len,
            vocab_size=voc_size,
            drop_prob=drop_prob,
        )

        self.dec_embedding = TransformerEmbedding(
            hidden_size=hidden_size,
            max_len=dec_max_len,
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

        self.linear = nn.Linear(hidden_size, voc_size, bias=False)
        self.linear.weight = self.dec_embedding.tok_emb.weight

    def forward(self, src, trg):
        """
        Forward pass for the Transformer.

        Args:
            src (Tensor): Source input tensor.
            trg (Tensor): Target input tensor.

        Returns:
            Tensor: Output logits.
        """
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)

        enc_src = self.enc_embedding(src)
        for layer in self.encoder_layers:
            enc_src = layer(enc_src, src_mask)

        out = self.dec_embedding(trg)
        for layer in self.decoder_layers:
            out = layer(out, enc_src, trg_mask, src_mask)
        out = self.linear(out)
        return out

    def make_src_mask(self, src):
        """
        Create source mask.

        Args:
            src (Tensor): Source input tensor.

        Returns:
            Tensor: Source mask.
        """
        src_mask = (src != self.pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask

    def make_trg_mask(self, trg):
        """
        Create target mask.

        Args:
            trg (Tensor): Target input tensor.

        Returns:
            Tensor: Target mask.
        """
        trg_pad_mask = (trg != self.pad_idx).unsqueeze(1).unsqueeze(3)
        trg_len = trg.shape[1]
        trg_sub_mask = (
            torch.tril(torch.ones(trg_len, trg_len)).type(torch.ByteTensor).to(device)
        )
        trg_mask = trg_pad_mask & trg_sub_mask
        return trg_mask
