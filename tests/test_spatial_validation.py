import numpy as np

from model.spatial.evaluate_spatial_model import _metrics, _top_n_mask, _update_counts


def test_top_n_mask_uses_exact_observed_counts_per_tile_and_lead():
    scores = np.asarray([[
        [[0.1, 0.9], [0.8, 0.2]],
        [[0.7, 0.4], [0.3, 0.2]],
    ]])
    counts = np.asarray([[2, 1]])
    result = _top_n_mask(scores, counts)
    np.testing.assert_array_equal(result.sum(axis=(-2, -1)), counts)
    np.testing.assert_array_equal(result[0, 0], [[False, True], [True, False]])
    np.testing.assert_array_equal(result[0, 1], [[True, False], [False, False]])


def test_transition_metrics_cover_all_four_change_states():
    origin = np.asarray([[[False, False], [True, True]]])
    actual = np.asarray([[[[False, True], [False, True]]]])
    predicted = actual.copy()
    binary = np.zeros((1, 4), dtype=np.int64)
    transitions = np.zeros((1, 4, 4), dtype=np.int64)
    _update_counts(binary, transitions, actual, predicted, origin)
    result = _metrics(binary[0], transitions[0])
    assert result["accuracy"] == 1.0
    assert result["transition_accuracy"] == 1.0
    assert result["dry_to_flood_recall"] == 1.0
    assert result["flood_to_dry_recall"] == 1.0
    assert transitions[0].diagonal().tolist() == [1, 1, 1, 1]
