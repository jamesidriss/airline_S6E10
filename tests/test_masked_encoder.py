import numpy as np
import pandas as pd
import torch

from src.features.view import RAW21
from src.models.masked import MaskedEncoder, prepare_covariates, masked_loss


def test_masking_hides_rating_and_both_numeric_categorical_twins():
    torch.manual_seed(1)
    model = MaskedEncoder([6] * 19, width=16, latent=4)
    cat = torch.zeros((2, 19), dtype=torch.long)
    num, missing = torch.zeros((2, 4)), torch.zeros((2, 4))
    for field in (0, 12, 17, 18):
        mask = torch.zeros((2, 21), dtype=torch.bool)
        mask[:, field] = True
        altered_cat, altered_num = cat.clone(), num.clone()
        altered_cat[:, field] = 5
        if field in (17, 18):
            altered_num[:, field - 17] = 100
        assert torch.equal(model.encode(cat, num, missing, mask),
                           model.encode(altered_cat, altered_num, missing, mask))
        clear = torch.zeros_like(mask)
        assert not torch.equal(model.encode(cat, num, missing, clear),
                               model.encode(altered_cat, altered_num, missing, clear))


def test_ssl_preprocessing_refuses_label_or_id_columns():
    frame = pd.DataFrame(np.tile(np.arange(21), (3, 1)), columns=RAW21)
    prepare_covariates(frame)
    for extra in ("id", "satisfaction", "te_rating"):
        try:
            prepare_covariates(frame.assign(**{extra: 1}))
        except ValueError:
            pass
        else:
            raise AssertionError(f"SSL accepted forbidden input {extra}")


def test_masked_loss_has_finite_gradient_and_no_loss_from_unmasked_targets():
    torch.manual_seed(2)
    model = MaskedEncoder([6] * 19, width=16, latent=4)
    cat = torch.randint(6, (8, 19))
    num, missing = torch.randn(8, 4), torch.zeros(8, 4)
    mask = torch.zeros((8, 21), dtype=torch.bool)
    mask[:, 0] = True
    loss = masked_loss(model, cat, num, missing, mask)
    assert torch.isfinite(loss)
    loss.backward()
    assert model.heads[0].weight.grad.abs().sum() > 0
    assert model.heads[1].weight.grad.abs().sum() == 0
