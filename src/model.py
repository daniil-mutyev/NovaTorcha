# src/model.py
import torch
import torch.nn as nn

class TinyGPT(nn.Module):
    def __init__(self, vocab_size, d_model=512, n_layers=6, n_heads=8, d_ff=2048, max_len=512, dropout=0.1):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model, n_heads, d_ff, dropout, batch_first=True)
            for _ in range(n_layers)
        ])
        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, x):
        B, T = x.size()
        pos = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
        h = self.drop(self.tok_emb(x) + self.pos_emb(pos))
        for blk in self.blocks:
            h = blk(h)
        h = self.ln(h)
        return self.head(h)  # logits (B, T, V)
