"""Fine-tune an existing compatible multi-horizon spatial ConvLSTM.

Run from the project root after preparing updated spatial data:

    python3 -m model.spatial.fine_tune_convlstm

The published output is replaced only if validation improves over the source
model. Architecture or horizon changes require full training instead.
"""
from __future__ import annotations

import argparse
import logging

from processing.config import get_cfg
from .train_convlstm import train


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", help="Existing model to fine-tune")
    parser.add_argument("--output", help="Destination for the improved model")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    source = args.source or get_cfg(
        "spatial_fine_tuning.source_model_path", "model/spatial/spatial_model.keras"
    )
    output = args.output or get_cfg(
        "spatial_fine_tuning.output_model_path", "model/spatial/spatial_model.keras"
    )
    train(output=str(output), fine_tune_from=str(source))


if __name__ == "__main__":
    main()
