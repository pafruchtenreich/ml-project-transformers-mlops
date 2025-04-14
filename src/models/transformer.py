import torch
import torch.nn.functional as F
from torch import nn


class AttentionLayer(nn.Module):
    """Multi-head attention layer."""

    def __init__(self, hidden_size, n_head):
        super(AttentionLayer, self).__init__()
        self.hidden_size = hidden_size
        self.n_head = n_head
        self.w_q = nn.Linear(hidden_size, hidden_size)
        self.w_k = nn.Linear(hidden_size, hidden_size)
        self.w_v = nn.Linear(hidden_size, hidden_size)
        self.w_o = nn.Linear(hidden_size, hidden_size)

    def forward(self, q, k, v, mask=None):
        batch_size, length, _ = q.shape
        # linear projections
        q = self.w_q(q)
        k = self.w_k(k)
        v = self.w_v(v)

        d_head = self.hidden_size // self.n_head
        # reshape to (batch_size, n_head, seq_len, d_head)
        q = q.view(batch_size, length, self.n_head, d_head).transpose(1, 2)
        k = k.view(batch_size, k.shape[1], self.n_head, d_head).transpose(1, 2)
        v = v.view(batch_size, v.shape[1], self.n_head, d_head).transpose(1, 2)

        # scaled dot-product attention
        scores = q @ k.transpose(-2, -1)
        scores = scores * (d_head**-0.5)

        if mask is not None:
            # mask shape must be broadcastable to (batch_size, n_head, seq_len, seq_len)
            scores = scores.masked_fill(mask == 0, -1e4)

        attn = F.softmax(scores, dim=-1)
        out = attn @ v

        # combine heads
        out = (
            out.transpose(1, 2).contiguous().view(batch_size, length, self.hidden_size)
        )
        out = self.w_o(out)
        return out


class EncoderLayer(nn.Module):
    """Transformer encoder layer with Pre-LayerNorm and GELU."""

    def __init__(self, hidden_size, ffn_hidden, n_head, drop_prob):
        super(EncoderLayer, self).__init__()
        # Pre-norm
        self.norm1 = nn.LayerNorm(hidden_size, eps=1e-5)
        self.attention = AttentionLayer(hidden_size=hidden_size, n_head=n_head)
        self.dropout1 = nn.Dropout(p=drop_prob)

        self.norm2 = nn.LayerNorm(hidden_size, eps=1e-5)
        self.linear1 = nn.Linear(hidden_size, ffn_hidden)
        self.linear2 = nn.Linear(ffn_hidden, hidden_size)
        self.dropout2 = nn.Dropout(p=drop_prob)

    def forward(self, x, src_mask):
        # ----- Self-Attention with Pre-LayerNorm -----
        _x = self.norm1(x)
        attn_out = self.attention(q=_x, k=_x, v=_x, mask=src_mask)
        attn_out = self.dropout1(attn_out)
        x = x + attn_out  # residual connection

        # ----- Feed Forward with Pre-LayerNorm & GELU -----
        _x = self.norm2(x)
        ffn_out = self.linear1(_x)
        ffn_out = F.gelu(ffn_out)
        ffn_out = self.linear2(ffn_out)
        ffn_out = self.dropout2(ffn_out)
        x = x + ffn_out  # residual connection

        return x


class DecoderLayer(nn.Module):
    """Transformer decoder layer with Pre-LayerNorm and GELU."""

    def __init__(self, hidden_size, ffn_hidden, n_head, drop_prob):
        super(DecoderLayer, self).__init__()
        # Pre-norm
        self.norm1 = nn.LayerNorm(hidden_size, eps=1e-5)
        self.self_attention = AttentionLayer(hidden_size=hidden_size, n_head=n_head)
        self.dropout1 = nn.Dropout(p=drop_prob)

        self.norm2 = nn.LayerNorm(hidden_size, eps=1e-5)
        self.enc_dec_attention = AttentionLayer(hidden_size=hidden_size, n_head=n_head)
        self.dropout2 = nn.Dropout(p=drop_prob)

        self.norm3 = nn.LayerNorm(hidden_size, eps=1e-5)
        self.linear1 = nn.Linear(hidden_size, ffn_hidden)
        self.linear2 = nn.Linear(ffn_hidden, hidden_size)
        self.dropout3 = nn.Dropout(p=drop_prob)

    def forward(self, dec, enc, trg_mask, src_mask):
        # ----- Self-Attention (Decoder) -----
        _x = self.norm1(dec)
        self_attn_out = self.self_attention(q=_x, k=_x, v=_x, mask=trg_mask)
        self_attn_out = self.dropout1(self_attn_out)
        dec = dec + self_attn_out

        # ----- Encoder-Decoder Cross Attention -----
        _x = self.norm2(dec)
        enc_dec_attn_out = self.enc_dec_attention(q=_x, k=enc, v=enc, mask=src_mask)
        enc_dec_attn_out = self.dropout2(enc_dec_attn_out)
        dec = dec + enc_dec_attn_out

        # ----- Feed Forward with GELU -----
        _x = self.norm3(dec)
        ffn_out = self.linear1(_x)
        ffn_out = F.gelu(ffn_out)
        ffn_out = self.linear2(ffn_out)
        ffn_out = self.dropout3(ffn_out)
        dec = dec + ffn_out

        return dec


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding module"""

    def __init__(self, hidden_size, max_len):
        super(PositionalEncoding, self).__init__()
        encoding = torch.zeros(max_len, hidden_size)
        pos = torch.arange(0, max_len).float().unsqueeze(1)
        _2i = torch.arange(0, hidden_size, step=2).float()
        encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / hidden_size)))
        encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / hidden_size)))
        self.encoding = encoding.unsqueeze(0)

    def forward(self, x):
        batch_size, seq_len = x.size()
        return self.encoding[:, :seq_len, :].to(x.device)


class TransformerEmbedding(nn.Module):
    """Embedding layer with positional encoding."""

    def __init__(self, vocab_size, hidden_size, max_len, drop_prob, pad_idx):
        super(TransformerEmbedding, self).__init__()
        self.tok_emb = nn.Embedding(vocab_size, hidden_size, padding_idx=pad_idx)
        self.pos_emb = PositionalEncoding(hidden_size, max_len)
        self.drop_out = nn.Dropout(p=drop_prob)

    def forward(self, x):
        tok_emb = self.tok_emb(x)
        pos_emb = self.pos_emb(x)
        return self.drop_out(tok_emb + pos_emb)


class Transformer(nn.Module):
    """Full Transformer model"""

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
        super().__init__()
        self.pad_idx = pad_idx

        # ----- Embeddings -----
        self.enc_embedding = TransformerEmbedding(
            hidden_size=hidden_size,
            max_len=max_len,
            vocab_size=voc_size,
            drop_prob=drop_prob,
            pad_idx=pad_idx,
        )
        self.dec_embedding = TransformerEmbedding(
            hidden_size=hidden_size,
            max_len=dec_max_len,
            vocab_size=voc_size,
            drop_prob=drop_prob,
            pad_idx=pad_idx,
        )

        # ----- Encoder & Decoder Stacks -----
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

        # ----- Final Linear Projection -----
        # Tie this weight with the decoder token embedding
        self.linear = nn.Linear(hidden_size, voc_size, bias=False)
        # Tie weights
        self.linear.weight = self.dec_embedding.tok_emb.weight

    def forward(self, src, trg):
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)

        # ----- Encoder -----
        enc_src = self.enc_embedding(src)
        for layer in self.encoder_layers:
            enc_src = layer(enc_src, src_mask)

        # ----- Decoder -----
        dec_tgt = self.dec_embedding(trg)
        out = dec_tgt
        for layer in self.decoder_layers:
            out = layer(out, enc_src, trg_mask, src_mask)

        # Final projection to vocabulary
        out = self.linear(out)  # shape: (batch_size, seq_len, voc_size)
        return out

    def make_src_mask(self, src):
        # shape: (batch_size, 1, 1, seq_len)
        src_mask = (src != self.pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask

    def make_trg_mask(self, trg):
        # pad mask
        trg_pad_mask = (trg != self.pad_idx).unsqueeze(1).unsqueeze(3)
        seq_len = trg.shape[1]
        # lower-triangular causal mask
        causal_mask = torch.tril(
            torch.ones((seq_len, seq_len), device=trg.device)
        ).bool()
        trg_mask = trg_pad_mask & causal_mask
        return trg_mask
