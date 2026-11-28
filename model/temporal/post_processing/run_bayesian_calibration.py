from __future__ import annotations

import json

from .bayesian_calibration import run_bayesian_calibration


def main() -> None:
    result = run_bayesian_calibration()
    print(json.dumps({
        "comparison_path": result["comparison_path"],
        "source_run_id": result["source_run_id"],
    }, indent=2))


if __name__ == "__main__":
    main()
