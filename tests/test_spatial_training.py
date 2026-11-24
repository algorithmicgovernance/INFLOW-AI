from model.spatial.train_convlstm import _patch_starts, _sample_index
from model.make_spatial_prediction import validate_spatial_model_contract
import pytest


def test_patch_starts_include_grid_edge():
    assert _patch_starts(1125, 64, 32)[-1] == 1061
    assert _patch_starts(1204, 64, 32)[-1] == 1140


def test_sample_index_is_chronological_and_complete():
    samples = _sample_index(range(6, 8), height=96, width=96, patch_size=64, stride=32)
    assert samples.shape == (8, 3)
    assert tuple(samples[0]) == (6, 0, 0)
    assert tuple(samples[-1]) == (7, 32, 32)

    sampled = _sample_index(
        range(6, 8), 96, 96, 64, 32, patches_per_timestep=2, seed=7
    )
    assert sampled.shape == (4, 3)
    assert (sampled[:2, 0] == 6).all()
    assert (sampled[2:, 0] == 7).all()


def test_spatial_model_contract_accepts_six_horizon_output():
    model = type("Model", (), {
        "input_shape": (None, 6, 64, 64, 10),
        "output_shape": (None, 6, 64, 64, 1),
    })()
    validate_spatial_model_contract(model, 6)


def test_spatial_model_contract_rejects_legacy_single_map_output():
    model = type("Model", (), {
        "input_shape": (None, 6, 64, 64, 10),
        "output_shape": (None, 64, 64, 1),
    })()
    with pytest.raises(ValueError, match="not the six-horizon checkpoint"):
        validate_spatial_model_contract(model, 6)
