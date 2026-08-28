"""Client-side local training (paper Algorithm 2): train up to E epochs with
early stopping on patience P, using Adam + categorical cross-entropy, matching
Table 2's FL hyperparameters."""
from __future__ import annotations

import copy
from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class LocalTrainConfig:
    max_epochs: int = 20
    patience: int = 5
    lr: float = 0.01
    batch_size: int = 32
    device: str = "cpu"


def local_train(model: nn.Module, X: torch.Tensor, y: torch.Tensor,
                 X_val: torch.Tensor, y_val: torch.Tensor,
                 cfg: LocalTrainConfig) -> tuple[nn.Module, list[float], int]:
    """Returns (trained model, per-epoch validation loss trajectory, epochs_run).

    Batches manually via index permutation rather than DataLoader/TensorDataset:
    same shuffle-every-epoch, fixed-batch-size semantics (no change to what's
    being reproduced), but avoids per-batch DataLoader/collate overhead that
    dominates wall-clock when data is already fully GPU-resident (this is
    called ~num_clients x num_rounds times per run_id, so the per-batch Python
    overhead compounds heavily -- see PROGRESS.md's wall-clock finding).
    """
    model = model.to(cfg.device)
    X, y = X.to(cfg.device), y.to(cfg.device)
    X_val, y_val = X_val.to(cfg.device), y_val.to(cfg.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    criterion = nn.CrossEntropyLoss()
    n = X.shape[0]

    best_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    patience_counter = 0
    loss_trajectory: list[float] = []
    epochs_run = 0

    for epoch in range(cfg.max_epochs):
        model.train()
        perm = torch.randperm(n, device=X.device)
        for start in range(0, n, cfg.batch_size):
            batch_idx = perm[start:start + cfg.batch_size]
            if batch_idx.numel() < 2:
                continue  # BatchNorm1d requires >1 sample per channel in train mode
            xb, yb = X[batch_idx], y[batch_idx]
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_val), y_val).item()
        loss_trajectory.append(val_loss)
        epochs_run = epoch + 1

        if val_loss < best_loss:
            best_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= cfg.patience:
                break

    model.load_state_dict(best_state)
    return model, loss_trajectory, epochs_run


@torch.no_grad()
def evaluate(model: nn.Module, X: torch.Tensor, y: torch.Tensor,
             device: str = "cpu") -> dict[str, float]:
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

    model = model.to(device)
    model.eval()
    X, y = X.to(device), y.to(device)
    logits = model(X)
    preds = logits.argmax(dim=1).cpu().numpy()
    y_np = y.cpu().numpy()
    return {
        "accuracy": accuracy_score(y_np, preds),
        "f1": f1_score(y_np, preds, average="weighted", zero_division=0),
        "precision": precision_score(y_np, preds, average="weighted", zero_division=0),
        "recall": recall_score(y_np, preds, average="weighted", zero_division=0),
        "f1_macro": f1_score(y_np, preds, average="macro", zero_division=0),
    }
