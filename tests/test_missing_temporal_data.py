import importlib.util
from pathlib import Path


_MAIN_PATH = Path(__file__).resolve().parents[1] / "__main__.py"
_SPEC = importlib.util.spec_from_file_location("inflow_main", _MAIN_PATH)
assert _SPEC is not None and _SPEC.loader is not None
main = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(main)


def test_check_if_new_data_returns_true_when_csv_is_missing(tmp_path):
    missing_path = tmp_path / "nested" / "temporal_data_seasonal_df.csv"

    assert main.check_if_new_data(str(missing_path)) is True
