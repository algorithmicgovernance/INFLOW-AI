# INFLOW-AI Flood Inundation Prediction

INFLOW-AI builds dekadal flood-inundation forecasts for the INFLOW study area.
The repository contains three connected workflows:

1. downloading and processing target and predictor data;
2. selecting a temporal forecasting model through staged ablation experiments;
3. operational temporal and spatial prediction using the selected temporal
   model and a separately trained ConvLSTM.

The configured target product controls the target timeline and alignment. The
current VIIRS source is dekadal; MODIS is also supported.

## Requirements

- Python 3.11 or 3.12 is recommended;
- network access for initial downloads and source updates;
- TensorFlow/Keras for spatial training and inference;
- the geospatial system libraries required by Rasterio and GeoPandas.

Python 3.13 can run much of the temporal pipeline, but current TensorFlow builds
used by this project require Python 3.12 or earlier.

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python3 -m pip install -r requirements.txt
```

Run commands from the repository root.

## Pipeline overview

| Stage | Command | Result |
|---|---|---|
| Prepare data | `python3 -m model.ablation.prepare_temporal_data` | Downloads or updates raw sources and creates target-aligned HDF5 and temporal CSV products |
| Temporal ablation | `python3 -m model.ablation.run_temporal_ablation` | Runs configured experiments and writes the best deployable temporal-model metadata |
| Spatial training | `python3 -m model.spatial.train_convlstm` | Trains the multi-horizon ConvLSTM and optionally runs spatial validation |
| Spatial validation only | `python3 -m model.spatial.evaluate_spatial_model` | Compares a saved spatial model with persistence and GloFAS baselines |
| Deployment | `python3 __main__.py` | Updates operational data, retrains the selected temporal model, and exports temporal and spatial forecasts |

The old `model.ablation.prepare_data` and `model.ablation.run_ablation` commands
remain as compatibility wrappers. New scripts and documentation should use the
explicit `prepare_temporal_data` and `run_temporal_ablation` names.

## 1. Data preparation

Configure `runtime.target_product` and run:

```bash
python3 -m model.ablation.prepare_temporal_data
```

No trained model is required. The command prepares:

- VIIRS or MODIS inundation targets;
- Victoria, Albert and Kyoga lake data;
- gauge rainfall and teleconnections;
- gridded TAMSAT rainfall, cumulative rainfall and soil moisture;
- ECMWF rainfall forecasts;
- optional HydroATLAS basin attributes.

Daily TAMSAT rainfall and moisture are streamed into complete target periods
using precomputed masks. Their HDF5 grids and regional temporal CSV summaries
are created together and appended when possible. Existing valid raw downloads
are reused, so processed outputs can be rebuilt without downloading everything
again.

ECMWF processing combines:

- historical hindcasts from `sources.ecmwf.hindcast_base_url`;
- the archived 2025 operational forecasts;
- the real-time archive used for 2026 and later forecasts.

For each target period, the latest eligible initialization is ensemble-averaged
into four non-overlapping rainfall windows: days 1–10, 11–20, 21–30 and 31–40.
The compact coarse-grid HDF5 is used by spatial training, while regional values
are exported to `ecmwf_hindcasts_temporal.csv` for temporal models. Ensemble
size is stored per initialization because archives can contain different member
counts. Existing initialization selections are checked before expensive
extraction and reprojection.

NetCDF run metadata is cached incrementally at
`paths.downloads.ecmwf_inventory_cache`. Each entry is keyed by file path,
size, modification time and archive type. Unchanged files are not reopened on
later runs, while new or changed files are inspected and added to the cache.
The cache is saved after every inspected file, so interrupting inventory loses
at most the file currently being read.

Processed target-aligned products are written under:

```text
data/historic/<target-product>-aligned/
```

Raw source files remain under `data/downloads/`.

VIIRS has separate historical and operational paths. Only immutable maps from
the configured `final/` archive are written to the historical HDF5 and temporal
CSV used for training and validation. During deployment, the pipeline selects
the highest available version of every non-final dekad from
`prelim/prelim_versions.csv` and adds it to a disposable view under
`data/operational/`. These provisional maps inform the current forecast but
never enter the historical record; a preliminary dekad disappears from the
operational overlay automatically when its final map becomes available.

### Optional basin-attribute regeneration

`processing/data_cleaning/process_basin_attributes.py` reproduces the static
HydroATLAS PCA layers used by the spatial model. It rasterizes basin membership
onto the configured reference grid, imputes and scales numeric attributes,
encodes selected categorical levels, and retains PCA components meeting the
configured variance threshold. It also writes metadata, source hashes and the
fitted preprocessing parameters.

This step is disabled by default because the HydroATLAS GeoPackage is not
downloaded automatically. Set the source path and deliberately enable
replacement before running normal data preparation:

```yaml
basin_attributes_processing:
  enabled: true
  basin_path: "data/maps/hydroatlas/inflow-basin-atlas.gpkg"
  overwrite_existing: true
```

Alternatively, run it explicitly:

```bash
python3 -m processing.data_cleaning.process_basin_attributes --force
```

Spatial training currently consumes the first saved PCA layer; the complete
component stack is retained for reproducibility.

## 2. Temporal ablation

Run the experiment grid defined in `config.yaml`:

```bash
python3 -m model.ablation.run_temporal_ablation
```

Before creating any model, a mandatory data preflight prints the target-series
length and date range plus row coverage and numeric missingness for every
aligned CSV. It also checks the corresponding gridded HDF5 products. Enabled
ECMWF features receive additional availability, consecutive-gap and trailing-
gap checks. Unsafe inputs stop the complete experiment grid before training;
the detailed report is written to `model/ablation/data_preflight.json`.

When `ablation.enabled: true` and complete experiment lists are present, the
configuration takes precedence over command-line defaults.

### Staged experiment design

With `ablation.staged.enabled: true`, ablation has two stages:

1. **Model selection:** vary model, target transformation, autoregression,
   cutoff date and target product while holding feature blocks fixed.
2. **Feature-block ablation:** take the top `top_n_model_configurations` under
   `ranking_metric` and evaluate the configured removal blocks, such as no
   ECMWF, no target lags, no historical feature lags, or no lags.

The current default grid tests random forest, elastic net and Transformer
models on raw, first-differenced and seasonally differenced targets. Edit
`ablation.experiments` to change that grid.

Feature selection is lead-specific but cached across model candidates. For a
given dataset, target setup, lead and feature-block combination, selection is
computed once and reused so model architectures receive comparable inputs.
Near-constant and highly correlated candidates are removed according to
`ablation.pipeline.feature_selection`.

The temporal Transformer is a joint non-autoregressive encoder-decoder model:
the encoder receives historical hydrometeorological sequences, while the
decoder receives the four ECMWF forecast windows. Decoder self-attention and
cross-attention produce all configured target horizons simultaneously.

Transformer results are logged and ranked during ablation, but joint
multi-horizon models are not currently published as the operational best model.
Deployment currently requires a lead-specific tabular model.

### Ablation outputs

Every run, including failures, is appended to:

```text
model/ablation/ablation_experiment_log.csv
```

Weights, scalers, selected-feature caches, PCA objects and runtime weights are
stored beneath `ablation.artifacts.base_dir`. The best successful deployable
row is written to:

```text
model/best_temporal_model.json
```

The deployment pipeline uses the saved per-lead feature names, preprocessing
artifacts, target transformation and model type from that file.

## 3. Spatial model training and validation

Prepare data first, then train:

```bash
python3 -m model.spatial.train_convlstm
```

The ConvLSTM consumes ten channels:

1. inundation;
2. gridded rainfall;
3. gridded soil moisture;
4. elevation;
5. the first basin-attribute PCA layer;
6. ECMWF days 1–10;
7. ECMWF days 11–20;
8. ECMWF days 21–30;
9. ECMWF days 31–40;
10. GloFAS 10-year return-period values.

Training uses a chronological split controlled by
`spatial_training.validation_fraction`. Target-product grids are streamed as
configured patches rather than loading the complete space-time cube. The model
emits all `forecast.horizon_periods` future maps simultaneously, with output
shape `(batch, horizon, patch_height, patch_width, 1)`.

Early stopping restores the best validation weights. `ModelCheckpoint` updates
the published model only when the configured validation metric improves, while
`BackupAndRestore` keeps separate resumable training state. An interrupted run
can therefore resume without replacing the best published model with a worse
epoch.

### Fine-tuning an existing spatial model

Fine-tuning reuses an already trained model and updates its weights at a lower
learning rate; it does not rebuild the ConvLSTM from random initialization:

```bash
python3 -m model.spatial.fine_tune_convlstm
```

The source model must already match the configured 10-channel input and
`forecast.horizon_periods` output architecture. Changing the number of channels,
patch dimensions or forecast horizons requires full training. Fine-tuning uses
the same chronological validation split as training and writes to a temporary
candidate checkpoint. The published model is replaced only when the candidate
improves the configured validation metric beyond `early_stopping_min_delta`.
Interrupted runs retain resumable state while leaving the published model
untouched.

Use `spatial_fine_tuning.recent_training_periods: 0` for all historical
training periods, or set a positive number to focus updates on the most recent
training periods. Source/output paths, learning rate, epoch ceiling, layer
freezing, early stopping, learning-rate reduction, backup frequency and
post-tuning validation are all configured under `spatial_fine_tuning`.

When `spatial_validation.enabled: true`, successful training automatically
compares the ConvLSTM with:

- spatial persistence;
- the GloFAS 20-year return-period map;
- the GloFAS 100-year return-period map.

For each validation tile and lead, all methods are assigned the observed number
of flooded pixels by selecting their highest-scoring cells. Reports include
binary flood metrics and transition metrics for dry-to-flood, flood-to-dry,
stay-dry and stay-wet outcomes.

Validation can also be run separately:

```bash
python3 -m model.spatial.evaluate_spatial_model
```

Outputs are written beside the model as `.validation.csv` and
`.validation.json`. The large GloFAS validation rasters are not committed and
must exist at the configured `spatial_validation` paths.

## 4. Operational deployment

Deployment requires:

- `model/best_temporal_model.json` and every artifact it references;
- a trained multi-horizon spatial model at `spatial_training.model_path` when
  `spatial_prediction.enabled: true`;
- processed spatial HDF5 inputs and configured static maps.

Run:

```bash
python3 __main__.py
```

The operational sequence is:

1. load the selected temporal experiment metadata and report current data
   completeness before attempting updates;
2. update current temporal sources and construct the final-plus-preliminary
   operational inundation view;
3. download missing ECMWF hindcast, 2025 archive and real-time files, then
   update the target-aligned ECMWF HDF5 and temporal CSV;
4. run the strict data preflight and stop if required inputs remain incomplete;
5. rebuild the selected lead-specific preprocessing and retrain that model on
   all currently available labelled history;
6. recursively reconstruct transformed targets into raw inundation fractions;
7. export the configured number of future periods and uncertainty interval;
8. run spatial inference and reporting when enabled.

The selected experiment’s product and target settings are authoritative during
deployment. `main.py` synchronizes the in-memory runtime configuration from the
best-model metadata before retraining.

Spatial inference does not train the ConvLSTM. The saved model must already
emit exactly `forecast.horizon_periods` maps. Each spatial lead is paired with
the matching temporal forecast date and constrained to the temporal predicted
inundated fraction. The configured uncertainty bounds define best, average and
worst spatial scenarios.

Population centres above `spatial_prediction.minimum_population` are displayed
on flood-change maps. Schools and hospitals are classified for each scenario
as `become_wet`, `stay_wet`, `become_dry` or `stay_dry`.

## Outputs

Each deployment creates a temporal forecast directory beneath `outputs.root`:

```text
temporal_predictions_<target>_<start-date>_to_<end-date>/
├── temporal_predictions.csv
├── model_performance.json
├── prediction_*.png
├── prediction_year_by_year_comparison.png
├── explanations/
│   ├── shap_waterfall_lead_*.png
│   └── shap_metadata.json
└── spatial_predictions/
    ├── lead_01_<date>/
    │   ├── flood_change_map.png
    │   └── impacted_facilities/
    │       ├── impacted_hospitals.csv
    │       └── impacted_schools.csv
    └── ... lead_N_<date>/
```

The facility CSVs include every configured facility and its classification
under best, average and worst scenarios. The cumulative temporal forecast
ledger is written to `outputs.forecast_performance_csv`.

## Configuration guide

All operationally tunable paths and parameters are in [config.yaml](config.yaml).

### `sources` and `paths`

- `sources` contains remote VIIRS, MODIS, rainfall, lake, teleconnection and
  ECMWF endpoints.
- `sources.ecmwf` also controls download concurrency, request timeout, member
  window definitions, maximum initialization age and HDF5 compression.
- `paths.downloads` contains reusable raw-download locations.
- `paths.historic` contains processed target-aligned HDF5 and CSV products.
- `paths.maps` contains catchments and administrative boundaries.

### `reference_grid`

`reference_grid.mask_regions` defines the CRS, transform, shape and bounds used
by aligned spatial products and masks. Changing it requires rebuilding spatial
products and retraining the spatial model.

### `runtime`, `forecast`, and `outputs`

```yaml
runtime:
  target_product: "viirs"       # viirs or modis

forecast:
  horizon_periods: 6            # shared temporal/spatial horizon
  interval_percent: 80          # operational uncertainty interval

outputs:
  root: "predictions"
```

`forecast.horizon_periods` must match the output horizon of the saved spatial
model. `runtime.study_start_date` and the source-specific runtime sections
control historical acquisition.

### `ablation`

- `ablation.pipeline` controls the target column, optional feature blocks,
  lag steps, feature selection, optional PCA and Transformer architecture.
- `ablation.experiments` defines the stage-one model grid.
- `ablation.staged` controls ranking, promotion count, fixed preprocessing and
  stage-two feature-block removals.
- `ablation.artifacts.base_dir` controls saved experiment artifacts.

### `data_preflight`

This section controls minimum timeline length, target and predictor missingness,
date coverage, ECMWF availability after its first forecast, allowable trailing
ECMWF gaps, and JSON report paths. Relaxing these thresholds permits more
imputation and should be treated as an explicit modelling decision.

### `spatial_training`

This section controls static input paths, sequence and patch dimensions,
sampling density, batch size, HDF5 cache, architecture, optimizer, loss,
maximum epochs, chronological validation fraction, mixed precision, checkpoint
monitoring, early stopping and learning-rate reduction.

### `spatial_fine_tuning`

This section controls the existing source model, publication path, reduced
learning rate, optional recent-period training window, optional ConvLSTM layer
freezing, epoch ceiling, early stopping, checkpoint backup and validation after
fine-tuning.

### `spatial_validation` and `spatial_prediction`

- `spatial_validation` enables post-training evaluation and configures the
  inundation threshold and GloFAS baseline rasters.
- `spatial_prediction` enables deployment inference and configures map,
  population-centre, facility and catchment paths plus inference batch size.

### `basin_attributes_processing`

This optional section controls HydroATLAS inputs, selected attributes,
rasterization, imputation, PCA retention, output metadata and replacement of an
existing static artifact. It is `false` by default.

## Notes on reproducibility

- Raw downloads are retained under `data/downloads` and reused.
- Do not delete artifacts referenced by `model/best_temporal_model.json` or the
  ablation log if historical experiments must remain reproducible.
- Temporal feature selection is fitted on training data and cached by dataset
  and lead configuration.
- Metrics for transformed targets are reconstructed into raw target space
  before comparison.
- Spatial train/validation splitting is chronological, not random by row.

## License

This project is licensed under the [MIT License](LICENCE.md).

## Contact

For questions or issues, contact Jessica Rapson at
jessica.rapson@algorithmicgovernance.org.
