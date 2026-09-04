"""Full per-round diagnostic trace for one real-data ARMOR run: prints every
client's weight, z-score, and e-process log-values each round, plus which
client IDs are actually malicious. Used to find exactly why ARMOR still
underperforms fedavg under label_flip even after the shift-gate fix.

Usage:
    python scripts/trace_armor_run.py
"""
import sys

sys.path.insert(0, ".")

from armor_fl.data.preprocessing import load_cicids2017, train_test_split_stratified

import numpy as np
import torch

from armor_fl.data.partition import dirichlet_partition
from armor_fl.fl import attacks as atk
from armor_fl.fl.armor import ArmorAggregator, ArmorConfig, ClientUpdate
from armor_fl.fl.client import LocalTrainConfig, local_train
from armor_fl.fl.robust_stats import flatten_state_dict, unflatten_to_state_dict
from armor_fl.models.dds_backbone import SE1DSqueezeNet

SEED = 0
NUM_CLIENTS = 10
CLIENT_FRACTION = 0.6
NUM_ROUNDS = 15
LOCAL_EPOCHS = 10
MAL_FRACTION = 0.4
ALPHA = 0.5
DEVICE = "cuda"

bundle = load_cicids2017("data_raw/cicids2017", sample_frac=0.1, seed=SEED)
X_train, X_test, y_train, y_test = train_test_split_stratified(bundle.X, bundle.y, seed=SEED)
n_features, n_classes = X_train.shape[1], len(bundle.classes)
print(f"train={X_train.shape[0]} classes={n_classes}")

rng = np.random.default_rng(SEED)
torch.manual_seed(SEED)
client_partitions = dirichlet_partition(y_train, NUM_CLIENTS, alpha=ALPHA, seed=SEED)
malicious_ids = atk.assign_malicious_clients(NUM_CLIENTS, MAL_FRACTION, seed=SEED)
print(f"malicious_ids = {sorted(malicious_ids)}")

def model_factory():
    return SE1DSqueezeNet(num_features=n_features, num_classes=n_classes)

global_model = model_factory().to(DEVICE)
template_state = {k: v.clone() for k, v in global_model.state_dict().items()}
global_vector = flatten_state_dict(template_state)
local_model = model_factory().to(DEVICE)

armor_cfg = ArmorConfig(burn_in_rounds=3)
aggregator = ArmorAggregator(NUM_CLIENTS, armor_cfg)

X_t = torch.tensor(X_train, dtype=torch.float32, device=DEVICE)
y_t = torch.tensor(y_train, dtype=torch.long, device=DEVICE)

for round_idx in range(1, NUM_ROUNDS + 1):
    n_select = max(1, int(round(CLIENT_FRACTION * NUM_CLIENTS)))
    selected = rng.choice(NUM_CLIENTS, size=n_select, replace=False).tolist()
    attack_active = round_idx >= 3

    updates = []
    for cid in selected:
        idx = client_partitions[cid]
        if len(idx) < 4:
            continue
        n_val = max(1, int(0.2 * len(idx)))
        val_idx, train_idx = idx[:n_val], idx[n_val:]
        if len(train_idx) == 0:
            continue
        train_idx_t = torch.as_tensor(train_idx, dtype=torch.long, device=DEVICE)
        val_idx_t = torch.as_tensor(val_idx, dtype=torch.long, device=DEVICE)
        Xc, yc = X_t[train_idx_t], y_t[train_idx_t]
        Xv, yv = X_t[val_idx_t], y_t[val_idx_t]

        is_malicious = attack_active and cid in malicious_ids
        if is_malicious:
            yc = torch.tensor(
                atk.label_flip(yc.cpu().numpy(), n_class := n_classes, seed=SEED + round_idx),
                dtype=torch.long, device=DEVICE,
            )

        local_model.load_state_dict(unflatten_to_state_dict(global_vector, template_state))
        train_cfg = LocalTrainConfig(max_epochs=LOCAL_EPOCHS, patience=3, lr=0.01,
                                      batch_size=256, device=DEVICE)
        local_model, loss_traj, _ = local_train(local_model, Xc, yc, Xv, yv, train_cfg)
        local_vector = flatten_state_dict(local_model.state_dict())
        delta = local_vector - global_vector
        updates.append(ClientUpdate(client_id=cid, vector=global_vector + delta,
                                     n_k=len(train_idx),
                                     local_val_loss=loss_traj[-1] if loss_traj else float("nan")))

    if not updates:
        continue
    result = aggregator.step(updates)
    global_vector = result.aggregated_vector

    print(f"\n--- round {round_idx} (attack_active={attack_active}, "
          f"burn_in={result.burn_in_active}) ---")
    for u in updates:
        tag = "MAL" if u.client_id in malicious_ids else "hon"
        print(f"  client {u.client_id} [{tag}]: w={result.weights.get(u.client_id, 0):.1f} "
              f"z={result.z_scores.get(u.client_id, float('nan')):.2f} "
              f"log_e_pop={result.log_e_pop.get(u.client_id, float('nan')):.2f} "
              f"log_e_drift={result.log_e_drift.get(u.client_id, float('nan')):.2f} "
              f"log_e_shift={result.log_e_shift.get(u.client_id, float('nan')):.2f} "
              f"status={result.status.get(u.client_id)}")
    if result.excluded_this_round:
        print(f"  EXCLUDED this round: {result.excluded_this_round}")
    if result.reinstated_this_round:
        print(f"  REINSTATED this round: {result.reinstated_this_round}")

print(f"\nFinal malicious_ids: {sorted(malicious_ids)}")
