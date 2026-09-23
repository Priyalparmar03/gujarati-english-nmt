"""
From-scratch Transformer encoder-decoder (Vaswani et al., 2017), no pretrained
weights, no library model classes. Every module here — attention, positional
encoding, encoder/decoder layers — is hand-written in PyTorch.
"""
import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_k = d_model // n_heads
        self.n_heads = n_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        B = q.size(0)

        def split_heads(x):
            return x.view(B, -1, self.n_heads, self.d_k).transpose(1, 2)  # (B, h, T, d_k)

        q = split_heads(self.q_proj(q))
        k = split_heads(self.k_proj(k))
        v = split_heads(self.v_proj(v))

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)  # (B, h, T, d_k)

        out = out.transpose(1, 2).contiguous().view(B, -1, self.n_heads * self.d_k)
        return self.out_proj(out)


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x):
        return self.net(x)


class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, src_mask):
        attn_out = self.self_attn(x, x, x, src_mask)
        x = self.norm1(x + self.dropout(attn_out))
        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout(ff_out))
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, src_mask, tgt_mask):
        attn_out = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_out))
        cross_out = self.cross_attn(x, enc_out, enc_out, src_mask)
        x = self.norm2(x + self.dropout(cross_out))
        ff_out = self.ff(x)
        x = self.norm3(x + self.dropout(ff_out))
        return x


class Seq2SeqTransformer(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model=512,
        n_heads=8,
        n_layers=6,
        d_ff=2048,
        dropout=0.1,
        max_len=512,
        pad_id=0,
    ):
        super().__init__()
        self.pad_id = pad_id
        self.d_model = d_model

        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos_enc = PositionalEncoding(d_model, max_len)
        self.dropout = nn.Dropout(dropout)

        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )
        self.decoder_layers = nn.ModuleList(
            [DecoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )

        self.out_proj = nn.Linear(d_model, vocab_size)
        # tie input/output embeddings — halves param count, standard practice
        self.out_proj.weight = self.embed.weight

    def make_src_mask(self, src):
        # (B, 1, 1, T_src)
        return (src != self.pad_id).unsqueeze(1).unsqueeze(2)

    def make_tgt_mask(self, tgt):
        B, T = tgt.shape
        pad_mask = (tgt != self.pad_id).unsqueeze(1).unsqueeze(2)  # (B,1,1,T)
        causal = torch.tril(torch.ones((T, T), device=tgt.device)).bool()  # (T,T)
        return pad_mask & causal  # (B,1,T,T)

    def encode(self, src, src_mask):
        x = self.embed(src) * math.sqrt(self.d_model)
        x = self.dropout(self.pos_enc(x))
        for layer in self.encoder_layers:
            x = layer(x, src_mask)
        return x

    def decode(self, tgt, enc_out, src_mask, tgt_mask):
        x = self.embed(tgt) * math.sqrt(self.d_model)
        x = self.dropout(self.pos_enc(x))
        for layer in self.decoder_layers:
            x = layer(x, enc_out, src_mask, tgt_mask)
        return x

    def forward(self, src, tgt):
        src_mask = self.make_src_mask(src)
        tgt_mask = self.make_tgt_mask(tgt)
        enc_out = self.encode(src, src_mask)
        dec_out = self.decode(tgt, enc_out, src_mask, tgt_mask)
        return self.out_proj(dec_out)

    @torch.no_grad()
    def greedy_translate(self, src, bos_id, eos_id, max_len=128):
        self.eval()
        src_mask = self.make_src_mask(src)
        enc_out = self.encode(src, src_mask)
        B = src.size(0)
        ys = torch.full((B, 1), bos_id, dtype=torch.long, device=src.device)
        finished = torch.zeros(B, dtype=torch.bool, device=src.device)
        for _ in range(max_len - 1):
            tgt_mask = self.make_tgt_mask(ys)
            dec_out = self.decode(ys, enc_out, src_mask, tgt_mask)
            logits = self.out_proj(dec_out[:, -1])
            next_tok = logits.argmax(-1, keepdim=True)
            ys = torch.cat([ys, next_tok], dim=1)
            finished |= next_tok.squeeze(1) == eos_id
            if finished.all():
                break
        return ys

    @torch.no_grad()
    def beam_translate(self, src, bos_id, eos_id, beam_size=4, max_len=128, len_penalty=0.6):
        """Batch size 1 beam search — simple and correctness-focused."""
        self.eval()
        assert src.size(0) == 1, "beam_translate expects batch size 1"
        src_mask = self.make_src_mask(src)
        enc_out = self.encode(src, src_mask)

        beams = [(torch.tensor([[bos_id]], device=src.device), 0.0, False)]
        for _ in range(max_len - 1):
            candidates = []
            for seq, score, done in beams:
                if done:
                    candidates.append((seq, score, done))
                    continue
                tgt_mask = self.make_tgt_mask(seq)
                dec_out = self.decode(seq, enc_out, src_mask, tgt_mask)
                logits = self.out_proj(dec_out[:, -1])
                log_probs = torch.log_softmax(logits, dim=-1).squeeze(0)
                topk = torch.topk(log_probs, beam_size)
                for tok, lp in zip(topk.indices, topk.values):
                    new_seq = torch.cat([seq, tok.view(1, 1)], dim=1)
                    new_done = tok.item() == eos_id
                    candidates.append((new_seq, score + lp.item(), new_done))

            def rank_key(c):
                seq, score, _ = c
                length = seq.size(1)
                return score / (length ** len_penalty)

            candidates.sort(key=rank_key, reverse=True)
            beams = candidates[:beam_size]
            if all(done for _, _, done in beams):
                break

        beams.sort(key=lambda c: c[1] / (c[0].size(1) ** len_penalty), reverse=True)
        return beams[0][0]
