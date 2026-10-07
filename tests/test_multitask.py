import numpy as np
import torch
from src.models.multitask import SatisfactionMultitask, normalization_fit


def test_multitask_normalization_excludes_application_rows():
    X = np.array([[1., 3.], [3., 7.], [100., 200.]], dtype='float32')
    mean, scale = normalization_fit(X, np.array([0, 1]), chunk=1)
    altered = X.copy()
    altered[2] = -1e9
    m2, s2 = normalization_fit(altered, np.array([0, 1]), chunk=1)
    assert np.array_equal(mean, m2) and np.array_equal(scale, s2)
    np.testing.assert_allclose(mean, [2, 5])
    np.testing.assert_allclose(scale, [1, 2])


def test_auxiliary_loss_trains_shared_encoder_without_satisfaction_head():
    torch.manual_seed(4)
    model = SatisfactionMultitask([6] * 19, 5)
    cat = torch.randint(0, 6, (8, 19))
    numeric, missing = torch.randn(8, 4), torch.zeros(8, 4)
    mask = torch.zeros(8, 21, dtype=torch.bool)
    mask[:, 0] = True
    loss = model.auxiliary_loss(cat, numeric, missing, mask)
    loss.backward()
    assert model.covariate.encoder[0].weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in model.satisfaction.parameters())
    assert model(torch.randn(8, 5), cat, numeric, missing).shape == (8,)
