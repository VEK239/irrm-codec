"""Shared models and datasets for the frozen residue-state benchmark.

The public encoders remain frozen.  Only these task heads are trained.  Every
residue-level condition uses the same head topology; the sole shape-dependent
component is the initial linear projection from the public encoder width.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset


class ResidueArrayDataset(Dataset):
    """Index a memory-mapped frozen residue matrix without loading it eagerly."""

    def __init__(self, features_path, mask_path, rows, targets):
        self.features = np.load(Path(features_path), mmap_mode="r")
        self.mask = np.load(Path(mask_path), mmap_mode="r")
        self.rows = np.asarray(rows, dtype=np.int64)
        self.targets = np.asarray(targets)
        if self.features.shape[:2] != self.mask.shape:
            raise ValueError("Frozen feature and mask shapes do not agree.")
        if len(self.rows) != len(self.targets):
            raise ValueError("Rows and targets must have identical lengths.")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        return (
            torch.from_numpy(np.asarray(self.features[row], dtype=np.float32)),
            torch.from_numpy(np.asarray(self.mask[row], dtype=np.bool_)),
            torch.as_tensor(self.targets[index]),
        )


class FrozenResidueDecoder(nn.Module):
    """Position-aware decoder over frozen per-residue states.

    Learned output queries cross-attend to every residue state.  Unlike global
    mean pooling, this interface never averages away residue order before the
    task head sees it.
    """

    def __init__(
        self,
        input_dim,
        vocab_size=25,
        max_len=40,
        hidden_dim=512,
        num_layers=3,
        nhead=8,
        ff_mult=4,
        dropout=0.2,
    ):
        super().__init__()
        self.max_len = max_len
        self.input_norm = nn.LayerNorm(input_dim)
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.source_position = nn.Parameter(torch.randn(max_len, hidden_dim) * 0.02)
        self.output_queries = nn.Parameter(torch.randn(max_len, hidden_dim) * 0.02)
        layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * ff_mult,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=num_layers)
        self.output = nn.Linear(hidden_dim, vocab_size)

    def forward(self, features, residue_mask):
        if features.ndim != 3 or residue_mask.shape != features.shape[:2]:
            raise ValueError("Expected features [B,L,D] and mask [B,L].")
        memory = self.input_projection(self.input_norm(features))
        memory = memory + self.source_position[: memory.shape[1]].unsqueeze(0)
        queries = self.output_queries.unsqueeze(0).expand(features.shape[0], -1, -1)
        decoded = self.decoder(
            queries,
            memory,
            memory_key_padding_mask=~residue_mask.bool(),
        )
        return self.output(decoded)


class FrozenResiduePgenHead(nn.Module):
    """Trainable position-aware pgen probe over frozen residue states."""

    def __init__(
        self,
        input_dim,
        max_len=40,
        hidden_dim=256,
        num_layers=2,
        nhead=8,
        ff_mult=4,
        dropout=0.2,
    ):
        super().__init__()
        self.input_norm = nn.LayerNorm(input_dim)
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.source_position = nn.Parameter(torch.randn(max_len, hidden_dim) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * ff_mult,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.context = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.query = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        self.pool = nn.MultiheadAttention(hidden_dim, nhead, dropout=dropout, batch_first=True)
        self.regressor = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
        )

    def forward(self, features, residue_mask):
        if features.ndim != 3 or residue_mask.shape != features.shape[:2]:
            raise ValueError("Expected features [B,L,D] and mask [B,L].")
        hidden = self.input_projection(self.input_norm(features))
        hidden = hidden + self.source_position[: hidden.shape[1]].unsqueeze(0)
        hidden = self.context(hidden, src_key_padding_mask=~residue_mask.bool())
        query = self.query.expand(features.shape[0], -1, -1)
        pooled, _ = self.pool(
            query,
            hidden,
            hidden,
            key_padding_mask=~residue_mask.bool(),
            need_weights=False,
        )
        return self.regressor(pooled[:, 0]).squeeze(-1)


def parameter_count(model):
    return int(sum(parameter.numel() for parameter in model.parameters()))
