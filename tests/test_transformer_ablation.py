import numpy as np

from model.ablation.models import MODEL_REGISTRY
from model.ablation.models.transformer_model import TransformerInput
from model.ablation.temporal_transformer_pipeline import _future_tokens, _split_feature_indices


def test_transformer_is_registered_without_importing_torch():
    assert "transformer" in MODEL_REGISTRY
    assert MODEL_REGISTRY["transformer"].is_multi_horizon is True


def test_transformer_partitions_history_and_four_ecmwf_windows():
    names = [
        "rainfall_lag_1",
        "ecmwf_rainfall_ssd_lead_1",
        "ecmwf_rainfall_ssd_lead_2",
        "ecmwf_rainfall_ceq_lead_3",
        "ecmwf_rainfall_ssd_lead_4",
    ]
    history, future = _split_feature_indices(names, np.arange(len(names)))
    assert history == [0]

    matrix = np.arange(15, dtype=np.float32).reshape(3, 5)
    tokens, channels = _future_tokens(matrix, future)
    assert tokens.shape == (3, 4, 2)
    assert channels == ["ecmwf_rainfall_ceq", "ecmwf_rainfall_ssd"]
    assert tokens[0, 0, 1] == matrix[0, 1]
    assert tokens[0, 1, 1] == matrix[0, 2]
    assert tokens[0, 2, 0] == matrix[0, 3]
    assert tokens[0, 3, 1] == matrix[0, 4]


def test_transformer_input_reports_sample_count():
    inputs = TransformerInput(
        history=np.zeros((7, 12, 3), dtype=np.float32),
        future=np.zeros((7, 4, 2), dtype=np.float32),
    )
    assert len(inputs) == 7

