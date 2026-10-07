import torch

from flow_matching.models import VectorFieldMLP


def test_vector_field_shape() -> None:
    model = VectorFieldMLP(data_dim=2, hidden_dim=32, time_dim=16)
    x = torch.randn(7, 2)
    t = torch.rand(7)
    assert model(t, x).shape == x.shape
