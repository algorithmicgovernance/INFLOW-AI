"""Legacy entry point; use ``model.ablation.prepare_temporal_data``."""

from .prepare_temporal_data import main, prepare_temporal_data

prepare_data = prepare_temporal_data

if __name__ == "__main__":
    main()
