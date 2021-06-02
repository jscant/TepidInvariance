import numpy as np
import pytest
from tepid_invariance.models.egnn import EGNN
from tepid_invariance.models.lie_transformer import LieTransformer
from tepid_invariance.utils import to_numpy
from test.setup_and_params import MODEL_KWARGS, ORIGINAL_COORDS, FEATS, MASK, \
    EPS, ROTATED_COORDS, setup
from torch import sigmoid

dump_path = setup()


def test_egnn_invariance():
    model = EGNN(dump_path, 0, 0, None, None, **MODEL_KWARGS).cuda().eval()

    unrotated_result = list(to_numpy(sigmoid(
        model((ORIGINAL_COORDS, FEATS, MASK)))).flatten())
    assert sum(abs(np.array(unrotated_result))) != pytest.approx(0, abs=1e-5)

    for rotated_coords in ROTATED_COORDS:
        rotated_result = list(to_numpy(sigmoid(
            model((rotated_coords, FEATS, MASK)))).flatten())
        assert unrotated_result == pytest.approx(rotated_result, abs=EPS)


def est_lie_transformer_invariance():
    model = LieTransformer(
        dump_path, 0, 0, None, None, **MODEL_KWARGS).cuda().eval()

    unrotated_result = list(to_numpy(sigmoid(
        model((ORIGINAL_COORDS, FEATS, MASK)))).flatten())
    assert sum(abs(np.array(unrotated_result))) != pytest.approx(0, abs=1e-5)

    for rotated_coords in ROTATED_COORDS:
        rotated_result = list(to_numpy(sigmoid(
            model((rotated_coords, FEATS, MASK)))).flatten())
        assert unrotated_result == pytest.approx(rotated_result, abs=EPS)


"""
def test_lie_conv_invariance():
    model = LieResNet(dump_path, 0, 0, None, None, **MODEL_KWARGS).eval()

    unrotated_result = float(sigmoid(model((ORIGINAL_COORDS, FEATS, MASK))))
    assert unrotated_result != pytest.approx(0, abs=1e-5)

    for rotated_coords in ROTATED_COORDS:
        rotated_result = float(sigmoid(model((rotated_coords, FEATS, MASK))))

        assert unrotated_result == pytest.approx(rotated_result, abs=EPS)
"""
