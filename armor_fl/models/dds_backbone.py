"""SE-1DSqueezeNet backbone, reimplemented from FedSE-1DSqueezeNet (Cluster
Computing, 2026), Table 1 / Fig. 2 / Sec. 3.2, in PyTorch.

Layer-for-layer spec (Table 1):
    Conv1D (Initial)                 k=3            ReLU     -> 32
    PWConv1D (Squeeze)               k=1            ReLU     -> 16
    PWConv1D (Expand1)               k=1            ReLU     -> 32
    DWConv1D (Expand2, dilation=2)   k=3            ReLU     -> 32 (depthwise)
    PWConv1D (Expand2)               k=1            ReLU     -> 32
    Concatenate(Expand1, Expand2)    -              BN       -> 64
    GAP1D (SE squeeze)                                       -> 64
    Dense (SE excitation 1)          -              ReLU     -> 16
    Dense (SE excitation 2)          -              Sigmoid  -> 64
    Multiply (SE scale)                                      -> 64
    GAP1D                            -              Dropout  -> 64
    FC                               -              Softmax  -> classes

This is the DDS module: DSConv (Expand2's DW+PW) + DilaConv (dilation=2 on the
depthwise conv) + SE attention, fused under a squeeze-expand pattern.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class SqueezeExcite1D(nn.Module):
    def __init__(self, channels: int, reduction_channels: int = 16):
        super().__init__()
        self.fc1 = nn.Linear(channels, reduction_channels)
        self.fc2 = nn.Linear(reduction_channels, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, L)
        z = x.mean(dim=-1)                     # squeeze: GAP over length -> (B, C)
        u = torch.relu(self.fc1(z))            # excitation layer 1
        s = torch.sigmoid(self.fc2(u))         # excitation layer 2 -> (B, C)
        return x * s.unsqueeze(-1)             # scale


class DDSModule(nn.Module):
    """Depthwise-Dilated-SE module (squeeze -> expand1 / expand2 -> concat -> SE)."""

    def __init__(self, in_channels: int = 32, squeeze_channels: int = 16,
                 expand_channels: int = 32, dilation: int = 2):
        super().__init__()
        self.squeeze = nn.Sequential(
            nn.Conv1d(in_channels, squeeze_channels, kernel_size=1), nn.ReLU(),
        )
        # Expand1: plain pointwise conv branch.
        self.expand1 = nn.Sequential(
            nn.Conv1d(squeeze_channels, expand_channels, kernel_size=1), nn.ReLU(),
        )
        # Expand2: DSConv1D (depthwise, dilated) + pointwise -- the DDS core.
        dw_padding = (3 - 1) * dilation // 2  # 'same'-style padding for k=3
        self.expand2 = nn.Sequential(
            nn.Conv1d(squeeze_channels, squeeze_channels, kernel_size=3,
                      dilation=dilation, padding=dw_padding,
                      groups=squeeze_channels),  # depthwise
            nn.ReLU(),
            nn.Conv1d(squeeze_channels, expand_channels, kernel_size=1),  # pointwise
            nn.ReLU(),
        )
        concat_channels = expand_channels * 2
        self.bn = nn.BatchNorm1d(concat_channels)
        self.se = SqueezeExcite1D(concat_channels, reduction_channels=16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = self.squeeze(x)
        e1 = self.expand1(s)
        e2 = self.expand2(s)
        out = torch.cat([e1, e2], dim=1)
        out = self.bn(out)
        out = self.se(out)
        return out


class SE1DSqueezeNet(nn.Module):
    """Full IDS backbone: Conv1D -> DDS -> AAP -> FC -> Softmax (Fig. 2)."""

    def __init__(self, num_features: int, num_classes: int,
                 init_channels: int = 32, squeeze_channels: int = 16,
                 expand_channels: int = 32, dropout: float = 0.5,
                 dilation: int = 2):
        super().__init__()
        self.num_features = num_features
        self.init_conv = nn.Sequential(
            nn.Conv1d(1, init_channels, kernel_size=3, padding=1), nn.ReLU(),
        )
        self.dds = DDSModule(init_channels, squeeze_channels, expand_channels,
                              dilation=dilation)
        concat_channels = expand_channels * 2
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(concat_channels, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, F) tabular flow features -> treat as a length-F, 1-channel sequence.
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (B, 1, F)
        x = self.init_conv(x)
        x = self.dds(x)
        x = x.mean(dim=-1)  # global average pool over sequence length
        x = self.dropout(x)
        return self.fc(x)  # logits; softmax applied via CrossEntropyLoss

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
