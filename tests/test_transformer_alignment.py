from types import SimpleNamespace

import numpy as np
import pandas as pd

from model.ablation.temporal_transformer_pipeline import (
    _aligned_target_dates,
    _aligned_targets,
    _common_origin_dates,
)
from model.ablation.temporal_ablation_runner import _recursive_raw_reconstruction


def _dataset(train_origins, test_origins, offset):
    train_origins = np.asarray(pd.to_datetime(train_origins))
    test_origins = np.asarray(pd.to_datetime(test_origins))
    return SimpleNamespace(
        origin_train_dates=train_origins,
        origin_test_dates=test_origins,
        train_dates=train_origins + np.timedelta64(offset, "D"),
        test_dates=test_origins + np.timedelta64(offset, "D"),
        y_train=np.arange(len(train_origins), dtype=np.float32) + offset,
        y_test=np.arange(len(test_origins), dtype=np.float32) + offset + 100,
    )


def test_transformer_alignment_uses_common_origins_across_leads_and_splits():
    lead_one = _dataset(["2003-05-11", "2003-05-21"], ["2003-06-01"], 10)
    # The same origin may fall on the other side of a lead-specific split.
    lead_two = _dataset(["2003-05-01"], ["2003-05-11", "2003-05-21", "2003-06-01"], 20)

    common = _common_origin_dates(lead_two, [lead_one, lead_two])
    assert pd.to_datetime(common).tolist() == list(pd.to_datetime([
        "2003-05-11", "2003-05-21", "2003-06-01"
    ]))

    targets = _aligned_targets(common, [lead_one, lead_two], "y_train")
    target_dates = _aligned_target_dates(common, [lead_one, lead_two])
    assert targets.shape == (3, 2)
    assert target_dates.shape == (3, 2)
    assert pd.Timestamp(target_dates[0, 0]) == pd.Timestamp("2003-05-21")
    assert pd.Timestamp(target_dates[0, 1]) == pd.Timestamp("2003-05-31")


def test_first_difference_reconstruction_is_recursive_for_points_and_samples():
    origins = np.asarray(pd.to_datetime(["2020-01-21", "2020-02-01"]))
    history_dates = np.asarray(pd.to_datetime(["2020-01-21", "2020-02-01"]))
    history_values = np.asarray([10.0, 20.0])
    transformed = np.asarray([[1.0, 2.0, -4.0], [3.0, -1.0, 2.0]])
    samples = np.stack([transformed, transformed + 1.0], axis=0)
    points, sample_paths = _recursive_raw_reconstruction(
        "first_differenced",
        transformed,
        samples,
        np.zeros_like(transformed),
        origins,
        history_dates,
        history_values,
    )
    np.testing.assert_allclose(points, [[11.0, 13.0, 9.0], [23.0, 22.0, 24.0]])
    np.testing.assert_allclose(sample_paths[0], points)
    np.testing.assert_allclose(sample_paths[1], [[12.0, 15.0, 12.0], [24.0, 24.0, 27.0]])
