"""8-bit uniform symmetric post-training quantization (paper Eqs. 8-10)."""
from __future__ import annotations

import copy

import torch
import torch.nn as nn


def quantize_tensor(w: torch.Tensor) -> tuple[torch.Tensor, float]:
    """Symmetric uniform quantization to signed int8, per-tensor scale."""
    max_abs = torch.max(w.abs().max(), torch.tensor(1e-12, device=w.device))
    scale = float(max_abs / 127.0)
    q = torch.clamp(torch.round(w / scale), -128, 127).to(torch.int8)
    return q, scale


def dequantize_tensor(q: torch.Tensor, scale: float) -> torch.Tensor:
    return q.to(torch.float32) * scale


def quantize_model_weights(model: nn.Module) -> dict[str, tuple[torch.Tensor, float]]:
    """Returns {param_name: (int8_tensor, scale)} for every weight/bias tensor."""
    quantized = {}
    for name, param in model.state_dict().items():
        if param.dtype in (torch.float32, torch.float64):
            quantized[name] = quantize_tensor(param.detach().cpu())
    return quantized


def dequantized_model(model: nn.Module) -> nn.Module:
    """Round-trips every float tensor through int8 quantization and back --
    simulates deploying the quantized model for inference/latency measurement,
    matching the paper's "Un/Quantized" evaluation pairs (Table 3/4)."""
    qmodel = copy.deepcopy(model)
    quantized = quantize_model_weights(model)
    state = qmodel.state_dict()
    for name, (q, scale) in quantized.items():
        state[name] = dequantize_tensor(q, scale)
    qmodel.load_state_dict(state)
    return qmodel


def quantized_size_bytes(model: nn.Module) -> int:
    """Size of the model if stored as int8 weights + one float32 scale per tensor."""
    quantized = quantize_model_weights(model)
    total = 0
    for q, _scale in quantized.values():
        total += q.numel() * 1  # int8 = 1 byte
        total += 4  # per-tensor scale, float32
    return total


def fp32_size_bytes(model: nn.Module) -> int:
    return sum(p.numel() * 4 for p in model.parameters())
