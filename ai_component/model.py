"""iTransformerWide: one iTransformer over many series at once: events and occupancy of every sensor (wide) or of every
cluster (cluster models) in, the same out.

Time-Series-Library iTransformer + one embedding added to every data column: sensor id + type + location + target
(events or occupancy). Without it the 2 x S columns would have no identity (iTransformer tokens have no order).
"""
import sys
from pathlib import Path
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Time-Series-Library"))
from models.iTransformer import Model


class ColumnEmbedding(nn.Module):
    """One vector per column: id + type + location of its sensor + its target (events / occupancy)."""
    def __init__(self, sensor_type, location, n_types, d_model):      # sensor_type (S,) int, location (S, 3) float
        super().__init__()
        S = len(sensor_type)
        self.id_emb = nn.Embedding(S, d_model)
        self.type_emb = nn.Embedding(n_types, d_model)
        self.loc_proj = nn.Linear(3, d_model)                           # (lat, lon, missing)
        self.target_emb = nn.Embedding(2, d_model)                      # 0 = events, 1 = occupancy
        col_sensor = torch.arange(S).repeat(2)                          # columns: events of the S sensors, then occupancy
        self.register_buffer("c_sensor", col_sensor)
        self.register_buffer("c_type", torch.as_tensor(sensor_type)[col_sensor])
        self.register_buffer("c_loc", torch.as_tensor(location, dtype=torch.float32)[col_sensor])
        self.register_buffer("c_target", torch.arange(2).repeat_interleave(S))

    def forward(self):                                                  # (2S, d_model)
        return self.id_emb(self.c_sensor) + self.type_emb(self.c_type) + self.loc_proj(self.c_loc) + self.target_emb(self.c_target)


class iTransformerWide(nn.Module):
    """The library's iTransformer forecast() step by step, with one added line: the column embedding."""
    def __init__(self, cfg, sensor_type, location):
        super().__init__()
        self.base = Model(cfg)                                          # library: enc_embedding, encoder, projection
        self.columns = ColumnEmbedding(sensor_type, location, cfg.n_types, cfg.d_model)

    def forward(self, x, x_mark):                                       # x (B, 96, 2S), x_mark (B, 96, 5)
        b = self.base
        means = x.mean(1, keepdim=True).detach()                        # 1. same normalisation as the library
        x = x - means
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x = x / stdev
        N = x.shape[2]
        tokens = b.enc_embedding(x, x_mark)                             # 2. (B, 2S + 5, d_model): data + calendar tokens
        tokens = torch.cat([tokens[:, :N] + self.columns(), tokens[:, N:]], dim=1)   # + column embedding (the added line)
        out, _ = b.encoder(tokens, attn_mask=None)                      # 3. encoder + final LayerNorm
        dec = b.projection(out).permute(0, 2, 1)[:, :, :N]              # 4. keep only the data tokens
        return dec * stdev[:, 0, :].unsqueeze(1) + means[:, 0, :].unsqueeze(1)   # 5. de-normalise
