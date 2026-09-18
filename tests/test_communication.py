from fedids_bench.models.mlp import SmallMLP
from fedids_bench.evaluation.communication import CommunicationTracker

def test_communication_tracker():
    model = SmallMLP(n_features=10, n_classes=2, hidden_dims=[5], seed=42)
    # (10*5+5) + (5*2+2) = 55 + 12 = 67 params
    # 67 * 4 bytes = 268 bytes
    assert model.parameter_bytes() == 268

    tracker = CommunicationTracker(model)
    # Record 2 rounds with 3 participating clients per round
    tracker.record_round(round_num=1, num_clients=3)
    tracker.record_round(round_num=2, num_clients=3)

    summary = tracker.get_summary()
    assert summary["parameter_bytes"] == 268
    # Per client per round: upload = 268, download = 268 -> round total for 3 clients = 6 * 268 = 1608
    # 2 rounds = 3216 bytes
    assert summary["total_upload_bytes"] == 268 * 3 * 2
    assert summary["total_download_bytes"] == 268 * 3 * 2
    assert summary["total_communication_bytes"] == 3216
