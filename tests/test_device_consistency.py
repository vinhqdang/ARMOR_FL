"""Regression tests for a real bug found while benchmarking on GPU (2026-08-27):
every existing config/test used device="cpu", so several places that create a
fresh CPU tensor and then combine it with a client-update vector (which lives
on whatever `cfg.device` local training used) had never been exercised on a
non-CPU device. Concretely: `flatten_state_dict(global_model.state_dict())`
stayed CPU-resident because `global_model` was never moved off the device
`model_factory()` defaults to, while local updates were moved to `cfg.device`
inside `local_train`; and `weighted_average` / `geometric_median` / attack
noise generators created their weight/noise tensors on the default (CPU)
device regardless of the vectors they were being combined with.

These tests are skipped when no CUDA device is available (e.g. the Mac this
project was started on) -- they exist to catch a regression on whichever
machine picks this project back up with a GPU.
"""
import pytest
import torch

from armor_fl.fl.armor import ArmorConfig
from armor_fl.fl.attacks import free_rider, gaussian_noise
from armor_fl.fl.robust_stats import geometric_median, krum_select, weighted_average
from armor_fl.fl.simulate import SimulationConfig, run_simulation
from armor_fl.models.dds_backbone import SE1DSqueezeNet

cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA device")


def make_synthetic(n=400, n_features=20, n_classes=3, seed=0):
    import numpy as np
    rng = np.random.default_rng(seed)
    centers = rng.normal(0, 3, size=(n_classes, n_features))
    y = rng.integers(0, n_classes, size=n)
    X = centers[y] + rng.normal(0, 1.0, size=(n, n_features))
    X = (X - X.min(0)) / (X.max(0) - X.min(0) + 1e-8)
    return X.astype("float32"), y.astype("int64")


@cuda_only
def test_robust_stats_helpers_accept_cuda_vectors():
    vectors = [torch.randn(10, device="cuda") for _ in range(5)]
    weighted_average(vectors, [1.0, 2.0, 3.0, 4.0, 5.0])
    geometric_median(vectors, weights=[1.0] * 5)
    krum_select(vectors, num_byzantine_assumed=1)


@cuda_only
def test_attacks_accept_cuda_delta():
    delta = torch.randn(50, device="cuda")
    assert gaussian_noise(delta).device.type == "cuda"
    assert free_rider(delta).device.type == "cuda"


@cuda_only
@pytest.mark.parametrize("aggregator", ["fedavg", "krum", "foolsgold", "armor"])
def test_run_simulation_on_cuda_does_not_crash(aggregator):
    """End-to-end regression check: this is exactly the path that crashed
    with 'Expected all tensors to be on the same device' before the fix."""
    X, y = make_synthetic()
    X_test, y_test = make_synthetic(n=100, seed=1)

    def model_factory():
        return SE1DSqueezeNet(num_features=20, num_classes=3)

    cfg = SimulationConfig(
        num_clients=6, client_fraction=0.6, num_rounds=2, local_epochs=2,
        patience=1, aggregator=aggregator,
        armor_config=ArmorConfig(burn_in_rounds=1) if aggregator == "armor" else None,
        aggregator_kwargs={"num_byzantine_assumed": 1} if aggregator == "krum" else {},
        attack_type="gaussian_noise", malicious_fraction=0.3,
        device="cuda", seed=0,
    )
    result = run_simulation(model_factory, X, y, X_test, y_test, 3, cfg)
    assert len(result.per_round_metrics) > 0
    assert torch.isfinite(torch.tensor(result.per_round_metrics[-1]["accuracy"]))
