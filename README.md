# INFLOW-AI Flood Inundation Prediction

INFLOW-AI downloads and processes hydrological, satellite, rainfall, moisture, lake-level, and teleconnection data for the INFLOW study area. It trains and evaluates temporal ablation models, retrains the selected model on the latest available history, and produces two-month inundation forecasts with uncertainty intervals and explanations.

![Latest year-by-year comparison](predictions/temporal_predictions_percent_inundation_ssd_2026-08-16_to_2026-10-01/prediction_year_by_year_comparison.png)

## Requirements

- Python 3.11 or newer
- macOS, Linux, or Windows
- Network access for the first data download and later source updates

Install dependencies in a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python3 -m pip install -r requirements.txt
```

## First Run

1. Review [config.yaml](config.yaml), especially `runtime.target_product`.
2. Download and process the datasets required for temporal-model ablation testing. This uses
   `runtime.target_product` from `config.yaml` and does not require an existing
   trained model:

```bash
python3 -m model.ablation.prepare_temporal_data
```

3. Run the temporal-model ablation experiments:

```bash
python3 -m model.ablation.run_temporal_ablation
```

4. Run the prediction pipeline:

```bash
python3 __main__.py
```

To retrain the ConvLSTM spatial model from the processed HDF5 grids, run:

```bash
python3 -m model.spatial.train_convlstm
```

Spatial training uses a chronological 80/20 split and streams 64-pixel patches
from the configured target-product inundation, rainfall, and moisture HDF5
files. It samples the configured number of spatial patches per timestep to keep
epochs tractable. The resulting model is saved to `model/spatial_model.keras`, which is
the model loaded by the spatial prediction pipeline. Training requires a
TensorFlow-compatible Python version (currently Python 3.12 or earlier).
The model channels also include all four ECMWF 10-day rainfall windows and the
aligned static GloFAS 10-year return-period raster. When `runtime.target_product`
is `viirs`, VIIRS inundation is used as the prediction target.
Spatial samples remain grouped by timestep so HDF5 chunks are decompressed once
and reused across patches. `spatial_training.hdf5_cache_mb`, `batch_size`,
`patches_per_timestep`, and optional GPU `mixed_precision` can be tuned for the
available hardware.
The model emits every configured future dekad simultaneously. Training has a
maximum epoch ceiling, but early stopping restores the best validation-loss
weights and stops only after validation has failed to improve for the configured
patience.
Only validation-loss improvements replace `model/spatial_model.keras`.
Resumable training state is backed up periodically and is kept separate from
the published best model; rerunning the same command resumes an interrupted fit.
After training, spatial validation compares the ConvLSTM with persistence and
GloFAS 20-year/100-year return-period baselines. For every tile and lead, each
method is assigned the observed number of inundated pixels by selecting its
highest-scoring pixels. Binary flood metrics and dry/flood transition metrics
are written to `model/spatial_model.validation.csv` and `.validation.json`.
The two large GloFAS source rasters are not committed to Git and must exist at
the paths configured under `spatial_validation` (copy them into the same paths
when training in Colab).

The first data run takes substantially longer. It downloads and builds the historic record for the spatial and temporal datasets, including HDF5 files and aligned temporal CSVs. Later runs update only the available period.

No external data folder needs to be downloaded or copied into the repository. The pipeline creates its required data products under `data/`.

## Outputs

The main run writes forecasts under `predictions/`, including:

- `temporal_predictions.csv`: historical values and future predictions
- `model_performance.json`: selected model, test metrics, preprocessing, and runtime retraining metadata
- `prediction_year_by_year_comparison.png` and other forecast plots
- `explanations/shap_waterfall_lead_*.png`: SHAP driver plots for each forecast lead
- `explanations/shap_metadata.json`: explanation metadata and contribution percentages

A cumulative forecast ledger is written to `predictions/forecast_performance.csv`.

## Configuration

The main settings are in [config.yaml](config.yaml).

### Runtime

```yaml
runtime:
  target_product: "viirs"   # viirs or modis
```

The target product controls the inundation source, aligned historic directory, temporal cadence, and forecast target.

The shared horizon setting controls both temporal and spatial forecasts:

```yaml
forecast:
  horizon_periods: 6
```

Temporal models forecast six dekads, while the ConvLSTM emits six matching maps
with output shape `(batch, 6, height, width, 1)`.

### Temporal-model ablation experiments

The `ablation.experiments` section controls the experiment grid:

- `models`
- `training_cutoff_dates`
- `autoregressive_values`
- `target_types`
- `inundation_products`
- `seed`
- `log_path`

With `ablation.staged.enabled`, the experiment runs in two stages. Stage one
varies model, autoregression, and target transformation while holding the
configured preprocessing blocks fixed. The configurations with the highest
configured ranking metric advance to stage two, which evaluates the ECMWF,
target-lag, and historical-feature-lag removals listed under
`ablation.staged.feature_block_ablations`. The stage and block name are stored
in every experiment-log row.

Feature selection is shared across candidate models. The configured
`feature_selection.selection_model_type` computes and caches one selection for
each dataset, lead, target setup, and feature-block combination. Thus all model
architectures are compared on the same lead-specific inputs without repeating
feature-selection cross-validation.

The `ablation.pipeline` section controls shared model preparation:

- `target_column`
- forecast horizon
- calendar features
- target and feature lag blocks
- lag steps such as `[1, 2, 3, 6, 12]`
- feature selection thresholds and candidate feature counts
- optional PCA

The `transformer` model is a joint, non-autoregressive multi-horizon ablation
model. Its encoder receives the configured sequence of historical observations;
its decoder receives four ECMWF tokens for days 1–10, 11–20, 21–30, and 31–40.
Decoder self-attention and encoder-decoder cross-attention feed a head that
predicts every configured target horizon in one pass. Its architecture and
training settings are under `ablation.pipeline.transformer`.

Transformer experiments are recorded and compared in the ablation log. They
are not promoted to `best_temporal_model.json` yet because the operational
retraining and forecast path currently accepts lead-specific tabular models.

Experiment weights, scalers, feature-selection caches, PCA artifacts, and runtime-retrained weights are stored beneath the configured `ablation.artifacts.base_dir`.

## Temporal-model ablation results

The ablation runner records experiment metadata and metrics in:

```text
model/ablation/ablation_experiment_log.csv
```

The `model/ablation` artifact directory and `ablation` configuration key are
retained for compatibility with existing experiment logs and saved models;
canonical Python modules and commands now use explicit `temporal` names.

It stores the selected experiment metadata in:

```text
model/best_temporal_model.json
```

The main pipeline consumes that metadata, uses the selected feature subsets and transformations, retrains the same model architecture on current data, and then forecasts.

## License

This project is licensed under the [MIT License](LICENCE.md).

## Contact

For questions or issues, contact Jessica Rapson at jessica.rapson@algorithmicgovernance.org.
