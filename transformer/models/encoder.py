import math
import torch
import torch.nn as nn
from models.transformer import TransformerBlock, PositionalEncoding

class TransformerEncoder(nn.Module):
    def __init__(self, vocab, d_model, n_head, n_layer, d_ff,
                 dropout=0.1, use_pos=True, use_res=True, use_ln=True):
        super().__init__()
        self.embed = nn.Embedding(vocab, d_model)
        self.use_pos = use_pos
        if use_pos:
            self.pos_enc = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_head, d_ff, dropout, use_res, use_ln)
            for _ in range(n_layer)
        ])
        self.final_ln = nn.LayerNorm(d_model) if use_ln else None
        self.fc_out = nn.Linear(d_model, vocab)

    def forward(self, x):
        x = self.embed(x) * math.sqrt(self.embed.embedding_dim)
        if self.use_pos: x = self.pos_enc(x)
        for layer in self.layers:
            x = layer(x)
        if self.final_ln is not None:
            x = self.final_ln(x)
        return self.fc_out(x)
