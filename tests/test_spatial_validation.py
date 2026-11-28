import numpy as np

from model.spatial.evaluate_spatial_model import (
    _change_detection_rows,
    _metrics,
    _persistence_mask,
    _top_n_mask,
    _update_counts,
)


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


def test_persistence_repeats_origin_mask_and_never_predicts_change():
    origin = np.asarray([[[False, True], [True, False]]])
    prediction = _persistence_mask(origin, horizon=3)

    assert prediction.shape == (1, 3, 2, 2)
    for lead in range(3):
        np.testing.assert_array_equal(prediction[:, lead], origin)

    actual = np.asarray([[[[True, True], [False, False]]]])
    binary = np.zeros((1, 4), dtype=np.int64)
    transitions = np.zeros((1, 4, 4), dtype=np.int64)
    _update_counts(binary, transitions, actual, prediction[:, :1], origin)
    result = _metrics(binary[0], transitions[0])
    assert result["dry_to_flood_predicted"] == 0
    assert result["dry_to_flood_precision"] == 0.0
    assert result["dry_to_flood_recall"] == 0.0
    assert result["dry_to_flood_f1"] == 0.0
    assert result["flood_to_dry_predicted"] == 0
    assert result["flood_to_dry_f1"] == 0.0


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
    assert result["transition_macro_f1"] == 1.0
    for transition in ("stable_dry", "dry_to_flood", "flood_to_dry", "stable_flood"):
        assert result[f"{transition}_precision"] == 1.0
        assert result[f"{transition}_f1"] == 1.0
        assert result[f"{transition}_accuracy"] == 1.0
        assert result[f"{transition}_iou"] == 1.0
    assert transitions[0].diagonal().tolist() == [1, 1, 1, 1]


def test_transition_metrics_measure_onset_and_recession_errors_separately():
    # One error within each origin group: stable dry is predicted as onset and
    # stable flood is predicted as recession.
    transitions = np.asarray([
        [8, 2, 0, 0],
        [1, 9, 0, 0],
        [0, 0, 7, 3],
        [0, 0, 4, 6],
    ])
    result = _metrics(np.asarray([21, 6, 5, 8]), transitions)

    assert result["dry_to_flood_precision"] == 9 / 11
    assert result["dry_to_flood_recall"] == 9 / 10
    assert result["dry_to_flood_iou"] == 9 / 12
    assert result["dry_to_flood_accuracy"] == 37 / 40
    assert result["flood_to_dry_precision"] == 7 / 11
    assert result["flood_to_dry_recall"] == 7 / 10
    assert result["flood_to_dry_iou"] == 7 / 14
    assert result["flood_to_dry_accuracy"] == 33 / 40
    assert result["dry_to_flood_false_positive"] == 2
    assert result["flood_to_dry_false_negative"] == 3


def test_change_detection_rows_emit_one_row_per_transition():
    origin = np.asarray([[[False, False], [True, True]]])
    actual = np.asarray([[[[False, True], [False, True]]]])
    binary = np.zeros((1, 4), dtype=np.int64)
    transitions = np.zeros((1, 4, 4), dtype=np.int64)
    _update_counts(binary, transitions, actual, actual, origin)
    wide = {"model": "convlstm", "lead_period": 1, **_metrics(binary[0], transitions[0])}

    rows = _change_detection_rows([wide])

    assert [row["transition"] for row in rows] == [
        "stable_dry", "dry_to_flood", "flood_to_dry", "stable_flood",
    ]
    assert all(row["f1"] == 1.0 for row in rows)
