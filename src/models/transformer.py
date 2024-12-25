class AttentionLayer(nn.Module):
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
        # apply projections
        q, k, v = self.w_q(q), self.w_k(v), self.w_v(v)

        # split heads
        # dims: (N,h,L,d)
        d_head = self.hidden_size // self.n_head
        q = q.view(batch_size, length, self.n_head, d_head).transpose(1, 2)
        k = k.view(batch_size, length, self.n_head, d_head).transpose(1, 2)
        v = v.view(batch_size, length, self.n_head, d_head).transpose(1, 2)

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
    def __init__(self, hidden_size, ffn_hidden, n_head, drop_prob):
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
        # 1. compute self attention
        _x = x
        x = self.attention(q=x, k=x, v=x, mask=src_mask)

        x = self.dropout1(x)
        x = self.norm1(x + _x)

        # 2. feed forward network
        _x = x
        x = self.linear1(x)
        x = F.ReLU(x)
        x = self.dropout2(x)
        x = self.linear2(x)

        x = self.dropout3(x)
        x = self.norm2(x + _x)
        return x


class DecoderLayer(nn.Module):
    def __init__(self, hidden_size, ffn_hidden, n_head, drop_prob):
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
        x = F.ReLU(x)
        x = self.dropout3(x)
        x = self.linear2(x)

        x = self.dropout4(x)
        x = self.norm3(x + _x)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, hidden_size, max_len):
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_len, hidden_size, device=device)
        self.encoding.requires_grad = False
        pos = torch.arange(0, max_len, device=device)
        pos = pos.float().unsqueeze(dim=1)
        _2i = torch.arange(0, hidden_size, step=2, device=device).float()
        self.encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / hidden_size)))
        self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / hidden_size)))

    def forward(self, x):
        batch_size, seq_len = x.size()
        return self.encoding[:seq_len, :]


class TransformerEmbedding(nn.Module):
    def __init__(self, vocab_size, hidden_size, max_len, drop_prob):
        super(TransformerEmbedding, self).__init__()
        self.tok_emb = nn.Embedding(vocab_size, hidden_size, padding_idx=1)
        self.pos_emb = PositionalEncoding(hidden_size, max_len)
        self.drop_out = nn.Dropout(p=drop_prob)

    def forward(self, x):
        tok_emb = self.tok_emb(x)
        pos_emb = self.pos_emb(x)
        return self.drop_out(tok_emb + pos_emb)


class Transformer(nn.Module):
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
        self.embedding = TransformerEmbedding(
            hidden_size=hidden_size,
            max_len=max_len,
            vocab_size=voc_size,
            drop_prob=drop_prob,
        )  # duplicate for enc/dec as in original if we want two maxlen (and maybe we should)

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
        src_mask = (src != self.pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask

    def make_trg_mask(self, trg):
        trg_pad_mask = (trg != self.pad_idx).unsqueeze(1).unsqueeze(3)
        trg_len = trg.shape[1]
        trg_sub_mask = (
            torch.tril(torch.ones(trg_len, trg_len)).type(torch.ByteTensor).to(device)
        )
        trg_mask = trg_pad_mask & trg_sub_mask
        return trg_mask
