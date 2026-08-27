import torch

from armor_fl.fl.aggregators import SimpleUpdate, build_aggregator

NAMES = ["fedavg", "trimmed_mean", "coordinate_median", "krum", "multi_krum", "foolsgold"]


def make_updates(n_honest=8, n_byzantine=2, d=20, offset=10.0, seed=0):
    g = torch.Generator().manual_seed(seed)
    updates = []
    for i in range(n_honest):
        updates.append(SimpleUpdate(client_id=i, vector=torch.randn(d, generator=g), n_k=50))
    for i in range(n_byzantine):
        updates.append(SimpleUpdate(client_id=n_honest + i,
                                     vector=torch.randn(d, generator=g) + offset, n_k=50))
    return updates


def test_all_aggregators_run_without_error():
    updates = make_updates()
    for name in NAMES:
        agg = build_aggregator(name, num_byzantine_assumed=2)
        out = agg.aggregate(updates)
        assert out.shape == (20,)
        assert torch.isfinite(out).all()


def test_robust_aggregators_resist_offset_byzantine_more_than_fedavg():
    updates = make_updates(n_honest=8, n_byzantine=2, offset=10.0)
    honest_mean = torch.stack([u.vector for u in updates[:8]], dim=0).mean(dim=0)

    fedavg_out = build_aggregator("fedavg").aggregate(updates)
    fedavg_err = torch.norm(fedavg_out - honest_mean).item()

    for name in ["trimmed_mean", "coordinate_median", "krum"]:
        agg = build_aggregator(name, num_byzantine_assumed=2)
        out = agg.aggregate(updates)
        err = torch.norm(out - honest_mean).item()
        assert err < fedavg_err, f"{name} did not beat plain FedAvg under attack"
