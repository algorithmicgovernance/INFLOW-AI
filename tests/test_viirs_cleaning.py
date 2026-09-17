import os
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from processing.data_cleaning.process_inundation_viirs import (
    clean_viirs_temporal_dataframe,
)


def test_viirs_cleaning_applies_centered_three_observation_median():
    dataframe = pd.DataFrame({
        "period_start": pd.date_range("2020-05-01", periods=5, freq="15D"),
        "percent_inundation": [1.0, 10.0, 2.0, 3.0, 4.0],
    })

    cleaned = clean_viirs_temporal_dataframe(dataframe)

    np.testing.assert_allclose(cleaned["percent_inundation"], [5.5, 2.0, 3.0, 3.0, 3.5])


def test_viirs_cleaning_interpolates_drying_period_increases():
    dataframe = pd.DataFrame({
        "period_start": pd.to_datetime([
            "2020-11-16", "2020-12-01", "2020-12-16", "2021-01-01",
            "2021-01-16", "2021-02-01", "2021-02-16",
        ]),
        # The centered median is unchanged. The December-January increase above
        # the last valid December value is replaced down to February.
        "percent_inundation": [6.0, 6.0, 10.0, 10.0, 10.0, 5.0, 5.0],
        "percent_inundation_region": [6.0, 6.0, 10.0, 10.0, 10.0, 5.0, 5.0],
    })

    cleaned = clean_viirs_temporal_dataframe(dataframe)

    expected = [6.0, 6.0, 5.75, 5.5, 5.25, 5.0, 5.0]
    np.testing.assert_allclose(cleaned["percent_inundation"], expected)
    np.testing.assert_allclose(cleaned["percent_inundation_region"], expected)


def test_viirs_cleaning_leaves_no_drying_period_increases_between_anchors():
    dataframe = pd.DataFrame({
        "period_start": pd.to_datetime([
            "2020-11-16", "2020-12-01", "2020-12-16", "2021-01-01",
            "2021-01-16", "2021-02-01", "2021-02-16",
        ]),
        "percent_inundation": [6.0, 6.0, 10.0, 10.0, 10.0, 5.0, 5.0],
    })

    cleaned = clean_viirs_temporal_dataframe(dataframe)
    drying_values = cleaned.loc[cleaned["period_start"].dt.month.isin([12, 1, 2, 3, 4]), "percent_inundation"]

    assert drying_values.diff().dropna().le(0).all()


def test_viirs_cleaning_can_use_post_april_recovery_anchor():
    dataframe = pd.DataFrame({
        "period_start": pd.to_datetime([
            "2020-11-16", "2020-12-01", "2020-12-16", "2021-04-01",
            "2021-04-16", "2021-05-01", "2021-05-16",
        ]),
        "percent_inundation": [6.0, 6.0, 6.0, 8.0, 8.0, 5.0, 5.0],
    })

    cleaned = clean_viirs_temporal_dataframe(dataframe)

    assert cleaned.loc[4, "percent_inundation"] < cleaned.loc[3, "percent_inundation"]
    assert cleaned.loc[5, "percent_inundation"] == 5.0


def test_viirs_cleaning_handles_record_starting_mid_drying_season():
    dataframe = pd.DataFrame({
        "period_start": pd.to_datetime([
            "2020-02-01", "2020-02-16", "2020-03-01", "2020-03-16",
            "2020-04-01", "2020-04-16", "2020-05-01",
        ]),
        "percent_inundation": [6.0, 6.0, 9.0, 9.0, 9.0, 5.0, 5.0],
    })

    cleaned = clean_viirs_temporal_dataframe(dataframe)
    drying_values = cleaned.loc[cleaned["period_start"].dt.month.isin([2, 3, 4]), "percent_inundation"]

    assert drying_values.diff().dropna().le(0).all()
