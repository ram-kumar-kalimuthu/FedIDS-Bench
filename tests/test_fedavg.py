import torch
from fedids_bench.federated.algorithms.fedavg import fedavg_aggregate

def test_fedavg_weighted_math():
    w1 = {"weight": torch.tensor([1.0, 2.0]), "bias": torch.tensor([0.0])}
    w2 = {"weight": torch.tensor([3.0, 6.0]), "bias": torch.tensor([10.0])}

    # Client 1 has 100 samples, Client 2 has 300 samples -> weights 0.25 and 0.75
    client_updates = [w1, w2]
    sample_counts = [100, 300]

    agg = fedavg_aggregate(client_updates, sample_counts)

    # weight: 0.25*[1, 2] + 0.75*[3, 6] = [0.25+2.25, 0.5+4.5] = [2.5, 5.0]
    expected_weight = torch.tensor([2.5, 5.0])
    # bias: 0.25*0 + 0.75*10 = 7.5
    expected_bias = torch.tensor([7.5])

    torch.testing.assert_close(agg["weight"], expected_weight)
    torch.testing.assert_close(agg["bias"], expected_bias)

def test_fedavg_single_client_identity():
    w1 = {"weight": torch.tensor([1.5, 2.5])}
    agg = fedavg_aggregate([w1], [50])
    torch.testing.assert_close(agg["weight"], w1["weight"])
