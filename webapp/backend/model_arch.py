"""Self-contained MLP architectures and preprocessing helpers for inference.

These are intentionally duplicated from the research repo so that the production
service has zero import dependency on Viiraa-Prediction at runtime.  The contract
is the checkpoint file format (pure numpy/dict), not shared Python classes.

When the research repo changes architectures or preprocessing, update this file
and bump the bundle_version in export_bundle.py.
"""

from __future__ import annotations

import math
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


# ── Neural network architectures ─────────────────────────────────────────────

class ScalarMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: Sequence[int], dropout: float, output_dim: int = 1) -> None:
        super().__init__()
        dims = [int(input_dim)] + [int(v) for v in hidden_dims]
        self.output_dim = int(output_dim)
        layers: List[nn.Module] = []
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            layers.extend([
                nn.Linear(in_dim, out_dim),
                nn.ReLU(),
                nn.BatchNorm1d(out_dim),
                nn.Dropout(float(dropout)),
            ])
        layers.append(nn.Linear(dims[-1], int(output_dim)))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        return out.squeeze(-1) if self.output_dim == 1 else out


class ResidualBlock(nn.Module):
    def __init__(self, dim: int, dropout: float) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.bn1 = nn.BatchNorm1d(dim)
        self.fc2 = nn.Linear(dim, dim)
        self.bn2 = nn.BatchNorm1d(dim)
        self.dropout = nn.Dropout(float(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.fc1(x)
        out = torch.relu(out)
        out = self.bn1(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.bn2(out)
        return torch.relu(x + self.dropout(out))


class ResidualScalarMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: Sequence[int], dropout: float, output_dim: int = 1) -> None:
        super().__init__()
        width = int(hidden_dims[0])
        depth = max(1, len(hidden_dims))
        self.output_dim = int(output_dim)
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, width),
            nn.ReLU(),
            nn.BatchNorm1d(width),
            nn.Dropout(float(dropout)),
        )
        self.blocks = nn.Sequential(*[ResidualBlock(width, dropout=float(dropout)) for _ in range(depth)])
        self.head = nn.Linear(width, int(output_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        x = self.blocks(x)
        out = self.head(x)
        return out.squeeze(-1) if self.output_dim == 1 else out


class GatedBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float) -> None:
        super().__init__()
        self.value = nn.Linear(in_dim, out_dim)
        self.gate = nn.Linear(in_dim, out_dim)
        self.bn = nn.BatchNorm1d(out_dim)
        self.dropout = nn.Dropout(float(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        value = torch.relu(self.value(x))
        gate = torch.sigmoid(self.gate(x))
        out = value * gate
        out = self.bn(out)
        return self.dropout(out)


class GatedScalarMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: Sequence[int], dropout: float, output_dim: int = 1) -> None:
        super().__init__()
        dims = [int(input_dim)] + [int(v) for v in hidden_dims]
        self.output_dim = int(output_dim)
        self.blocks = nn.ModuleList([
            GatedBlock(in_dim, out_dim, dropout=float(dropout))
            for in_dim, out_dim in zip(dims[:-1], dims[1:])
        ])
        self.head = nn.Linear(dims[-1], int(output_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        out = self.head(x)
        return out.squeeze(-1) if self.output_dim == 1 else out


def build_model(arch: str, input_dim: int, hidden_dims: List[int], dropout: float) -> nn.Module:
    if str(arch) == "residual":
        return ResidualScalarMLP(input_dim=input_dim, hidden_dims=hidden_dims, dropout=dropout)
    if str(arch) == "gated":
        return GatedScalarMLP(input_dim=input_dim, hidden_dims=hidden_dims, dropout=dropout)
    return ScalarMLP(input_dim=input_dim, hidden_dims=hidden_dims, dropout=dropout)


# ── Preprocessing helpers ─────────────────────────────────────────────────────

def transform_numeric(
    df: pd.DataFrame,
    cols: Sequence[str],
    med: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    if not cols:
        return np.zeros((len(df), 0), dtype=np.float32)
    arr = df[list(cols)].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
    arr = np.where(np.isfinite(arr), arr, med[None, :])
    return ((arr - mean[None, :]) / std[None, :]).astype(np.float32)


def transform_categorical(
    df: pd.DataFrame,
    cols: Sequence[str],
    levels: Dict[str, List[str]],
) -> np.ndarray:
    if not cols:
        return np.zeros((len(df), 0), dtype=np.float32)
    blocks: List[np.ndarray] = []
    for col in cols:
        vals = df[col].where(df[col].notna(), "__MISSING__").astype(str)
        known = set(levels[col])
        vals = vals.where(vals.isin(known), "__UNK__")
        cats = pd.Categorical(vals, categories=levels[col])
        oh = pd.get_dummies(cats, prefix=col, dtype=np.float32)
        blocks.append(oh.to_numpy(dtype=np.float32))
    return np.concatenate(blocks, axis=1).astype(np.float32)


def inverse_transform_target(
    y: np.ndarray,
    preprocess: Dict[str, object],
) -> np.ndarray:
    mode = str(preprocess["mode"])
    if mode == "none":
        return np.asarray(y, dtype=np.float64)
    if mode == "log1p":
        return np.expm1(np.asarray(y, dtype=np.float64)).astype(np.float64)
    if mode == "log1p_zscore":
        y_log = np.asarray(y, dtype=np.float64) * float(preprocess["std"]) + float(preprocess["mean"])
        return np.expm1(y_log).astype(np.float64)
    # zscore
    return (np.asarray(y, dtype=np.float64) * float(preprocess["std"]) + float(preprocess["mean"])).astype(np.float64)
