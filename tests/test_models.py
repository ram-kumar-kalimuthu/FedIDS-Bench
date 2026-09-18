import torch
from fedids_bench.models.mlp import SmallMLP

def test_mlp_forward_and_params():
    model = SmallMLP(n_features=10, n_classes=3, hidden_dims=[16, 8], seed=42)
    x = torch.randn(5, 10)
    out = model(x)
    assert out.shape == (5, 3)

    num_params = model.num_parameters()
    # (10*16 + 16) + (16*8 + 8) + (8*3 + 3) = 176 + 136 + 27 = 339
    assert num_params == 339
    
    # 339 float32 params * 4 bytes = 1356 bytes
    assert model.parameter_bytes() == 1356

def test_mlp_deterministic_init():
    m1 = SmallMLP(n_features=10, n_classes=3, seed=99)
    m2 = SmallMLP(n_features=10, n_classes=3, seed=99)
    
    p1 = m1.get_parameters()
    p2 = m2.get_parameters()
    
    for k in p1:
        assert torch.equal(p1[k], p2[k])
