# INFLOW-AI Standard Operating Procedure

Owner: ICPAC (**To confirm**). This SOP is based on the repository README,
configuration and implementation. Operational decisions not defined by the
repository are marked **Proposed** or **To confirm**.

## 1. Purpose and scope

Produce a dekadal flood-inundation forecast for the configured INFLOW target
product. Each operational run produces:

- a temporal forecast of inundated fraction and area for
  `forecast.horizon_periods` dekads (default 6, approximately two months), with
  uncertainty bounds;
- one spatial flood-change map per lead, constrained to the corresponding
  temporal forecast; and
- hospital and school classifications for best, average and worst scenarios.

Run the operational workflow from the repository root with:

```bash
python3 __main__.py
```

## 2. Roles

| Role | Responsibility | Name |
|---|---|---|
| Operator | Runs or monitors the scheduled job and completes the checklist | **To confirm** |
| Reviewer | Signs off sanity checks and interpretation before release | **To confirm** |
| Technical owner | Resolves failures and owns configuration, data and model artifacts | **To confirm** |
| Releaser | Sends approved outputs to partners | **To confirm** |

## 3. Prerequisites

Check these during initial setup and after any server, environment, data-product
or configuration change.

- Use Python 3.11 or 3.12. Python 3.13 can run much of the temporal workflow,
  but not the repository's current TensorFlow-dependent spatial training and
  inference.
- Install the geospatial system libraries required by Rasterio and GeoPandas
  and provide network access to the configured `sources` endpoints.
- Create and activate the environment:

  ```bash
  python3 -m venv .venv
  source .venv/bin/activate
  python3 -m pip install -r requirements.txt
  ```

- Keep `model/best_temporal_model.json` and every artifact it references under
  `ablation.artifacts.base_dir`.
- When `spatial_prediction.enabled: true`, keep a compatible multi-horizon
  ConvLSTM at `spatial_training.model_path` (default
  `model/spatial_model.keras`). Its input must match the configured sequence,
  patch and 10-channel contract, and its output horizon must equal
  `forecast.horizon_periods`.
- Keep the processed target-aligned HDF5 inputs under
  `data/historic/<target-product>-aligned/` and the configured elevation, basin
  attribute, GloFAS and other static maps.
- Supply non-automated inputs separately. The HydroATLAS GeoPackage is needed
  only to regenerate basin attributes. GloFAS static/training and 20-/100-year
  validation rasters must exist at their configured paths; they are not
  committed with the repository.

For first-time setup, or to update/rebuild aligned products, run:

```bash
python3 -m model.ablation.prepare_temporal_data
```

This prepares both temporal products and the gridded products used by spatial
training. It reuses valid raw downloads in `data/downloads/`. The strict data
preflight also runs during ablation and deployment and writes reports to the
paths under `data_preflight.report_paths`.

## 4. Routine operational run

**Frequency.** Run once per dekad (days 1-10, 11-20 and 21-month end), after
the closing dekad's inundation product is available. Source publication lags
and the exact production day are not enforced by the code. **To confirm:** the
production day and whether ICPAC will rerun after delayed inputs arrive.

**Proposed scheduler example** (set the actual path, account and timezone):

```cron
0 6 1,11,21 * * cd /path/to/INFLOW-AI && .venv/bin/python3 __main__.py >> logs/run_$(date +\%F).log 2>&1
```

Existing valid raw downloads are reused. However, a rerun is not completely
idempotent: the temporal model is retrained, its runtime timestamp forms part
of `model_version`, and another set of ledger rows can therefore be added.
Record which run is released.

### Procedure

1. Activate the environment; confirm disk space, required model artifacts,
   static maps and the previous run's outputs.
2. Run `python3 __main__.py` from the repository root, or monitor the scheduled
   invocation.
3. The command loads the selected experiment, performs an initial non-fatal
   data audit, updates target and predictor sources (including the
   final-plus-preliminary VIIRS operational view when configured), processes
   ECMWF data, and then runs the strict preflight.
4. It rebuilds the selected lead-specific temporal preprocessing, retrains the
   selected deployable temporal model on all currently labelled history,
   exports the temporal forecast, reports and ledger, and finally performs
   spatial inference and facility reporting. Deployment does not train or
   fine-tune the ConvLSTM.
5. Complete the release gate below. Release only the reviewed run and record
   it as described in section 9.

The experiment metadata controls the deployment target product and target
settings. Do not attempt to switch the deployed product only by changing
`runtime.target_product`.

## 5. Sanity checks (release gate)

The run directory is:

```text
<outputs.root>/temporal_predictions_<target>_<start-date>_to_<end-date>/
```

| # | Check | Pass |
|---|---|---|
| 1 | Process exits successfully and the log contains no traceback | Yes |
| 2 | `predictions/data_preflight.json` reports a passing deployment preflight; review `predictions/data_preflight_before_updates.json` for what changed | Yes |
| 3 | `temporal_predictions.csv` contains observed history followed by exactly `forecast.horizon_periods` forecast rows (`actual_fraction` empty), on consecutive target dekads beginning after the latest observation | Yes |
| 4 | Forecast rows contain no empty prediction or interval values; lower bound <= forecast <= upper bound; fractions lie in [0, 1] | Yes |
| 5 | Lead 1 is credible relative to the latest observed extent; any large jump has a defensible hydrometeorological explanation | Yes |
| 6 | `prediction_*.png`, `prediction_year_by_year_comparison.png`, `model_performance.json` and the generated feature-history plots exist and are plausible | Yes |
| 7 | `explanations/shap_waterfall_lead_*.png` and `shap_metadata.json` exist, or the log records the non-fatal SHAP warning and the reviewer accepts the run without explanations | Yes/waiver |
| 8 | `spatial_predictions/` contains one `lead_NN_<date>/` per lead, with `flood_change_map.png` and `impacted_facilities/impacted_hospitals.csv` and `impacted_schools.csv` | Yes |
| 9 | Best/average/worst map areas correspond to the temporal lower/point/upper forecasts; inspect change patterns and facility classifications for plausibility | Yes |
| 10 | Latest input dates are appropriate, including the target mask and an ECMWF initialization within the configured maximum age; investigate any extrapolated or missing recent predictor data | Yes |
| 11 | `outputs.forecast_performance_csv` contains the run's horizon rows and older rows have been backfilled with actual/error values where observations are now available | Yes |

Spatial outputs are constrained to the temporal fractions by selecting the
highest-scoring valid pixels, so the area agreement in check 9 is primarily an
implementation/format check. It does not independently validate spatial skill.

If checks 5, 6 or 9 fail, compare against persistence, inspect rainfall and
lake-level inputs (including possible releases), and escalate before release.

## 6. Spatial model updating by fine-tuning

Use fine-tuning to update the weights of an existing compatible spatial model
at a lower learning rate instead of rebuilding it from random initialization.
Fine-tuning is a deliberate maintenance operation; it is not part of
`python3 __main__.py`.

### When fine-tuning is appropriate

Fine-tune after new finalized, aligned spatial history has accumulated, when
the existing model still has the same:

- target product and input semantics;
- `spatial_training.sequence_length`;
- `spatial_training.patch_size`;
- 10-channel input contract; and
- `forecast.horizon_periods` output contract.

Changing the reference grid, patch dimensions, channel contract, target
product or forecast horizon requires rebuilding the spatial products as
applicable and full training with:

```bash
python3 -m model.spatial.train_convlstm
```

### Fine-tuning procedure

1. Back up the published model and its `.training.json`, validation reports and
   any existing `.fine_tuning.json`. Confirm that the updated aligned HDF5 and
   static inputs pass preparation/preflight checks.
2. Review `spatial_fine_tuning` in `config.yaml`:
   - `source_model_path`: compatible model to load;
   - `output_model_path`: publication destination (default is the same model);
   - `learning_rate`: default `0.0001`, lower than full training's `0.001`;
   - `recent_training_periods`: `0` uses all eligible historical training
     periods; a positive number limits weight updates to that many most-recent
     training periods;
   - `freeze_convlstm_layers`: when `true`, only non-ConvLSTM layers remain
     trainable; default is `false`;
   - epoch, early-stopping, learning-rate-reduction, backup and validation
     controls.
3. Prefer a separate output candidate for an operator-controlled first run:

   ```bash
   python3 -m model.spatial.fine_tune_convlstm \
     --source model/spatial_model.keras \
     --output model/spatial_model_candidate.keras
   ```

   To use the configured in-place paths instead, run:

   ```bash
   python3 -m model.spatial.fine_tune_convlstm
   ```

4. The command first evaluates the source on the chronological validation
   split. Training writes a hidden candidate checkpoint and publishes it to
   the requested output only if the monitored metric improves over that
   baseline by more than `spatial_fine_tuning.early_stopping_min_delta`.
   Otherwise the prior published model is left unchanged.
5. If interrupted, rerun the same command to resume from the configured
   `BackupAndRestore` state. Do not delete the fine-tuning backup directory
   until the run completes.
6. Review `<output>.fine_tuning.json`. Confirm `fine_tuning_published: true`,
   the source, baseline and history. With `run_validation: true`, a published
   improvement is also evaluated and writes `<output>.validation.csv` and
   `<output>.validation.json`.
7. Compare the candidate's spatial validation, transition metrics and visual
   outputs with the currently published model. Only after reviewer approval,
   point `spatial_training.model_path` at the candidate or replace the
   production artifact, then run the operational release gate.

The validation split is chronological and controlled by
`spatial_training.validation_fraction`. Fine-tuning changes the training
window only; it does not move or randomize the validation period.

## 7. Typical failures

The exact library error text can vary. Add confirmed production messages to
the runbook as they occur.

| Symptom | Likely cause | Action |
|---|---|---|
| `pip` reports no compatible TensorFlow distribution | Python 3.13 or unsupported platform | Use the supported Python 3.11/3.12 spatial environment |
| Rasterio/GeoPandas import or build failure | Missing geospatial system libraries | Install the required native libraries and reinstall requirements |
| Missing `best_temporal_model.json` or referenced artifact | Artifact moved or deleted | Restore the exact artifact set; otherwise rerun temporal ablation |
| Strict data preflight fails | Missingness, date coverage, target length or ECMWF availability exceeds configured limits | Review the JSON report and source processing; do not casually relax thresholds |
| Spatial model not found | Wrong `spatial_training.model_path` or no trained model | Correct/restore the model, train it, or deliberately set `spatial_prediction.enabled: false` for a temporal-only run |
| Spatial shape/horizon contract error | Saved model is incompatible with configured sequence, patch, channels or horizon | Restore a compatible checkpoint; use full training for architecture changes |
| Fine-tuning contract error | Source model does not exactly match the current configured input/output shapes | Do not fine-tune; use full spatial training |
| Fine-tuning reports no improving checkpoint | Candidate did not beat the baseline by `early_stopping_min_delta` | Keep the published model; review data/window settings before another attempt |
| Download connection error or timeout | Endpoint, network or proxy problem | Retry; valid raw files are reused; verify configured endpoints |
| ECMWF gaps or no eligible initialization | Archive gap, timeout or initialization older than configured maximum | Review `sources.ecmwf`, the preflight report and logs; retry when data is available |
| Spatial validation cannot find rasters | GloFAS 20-/100-year rasters absent | Supply them at `spatial_validation` paths, or deliberately disable validation—not operational inference |
| Basin-attribute regeneration fails | HydroATLAS GeoPackage absent or replacement not deliberately enabled | Leave regeneration disabled unless inputs are present; use `--force` for an intentional rebuild |
| Editing `runtime.target_product` does not change deployment product | Best-model metadata is authoritative | Run `python3 -m model.ablation.run_temporal_ablation` for the intended product |
| Transformer is not selected for deployment | Joint Transformer models are evaluated but not published as the operational best model | Expected: deployment currently requires lead-specific tabular models |
| Spatial training/fine-tuning interrupted | Session or machine loss | Rerun the same command; backup state supports resumption and validation checkpoints protect the published model |

## 8. Maintenance

| Task | When | Notes |
|---|---|---|
| Review run log and release gate | Every run | Section 5 |
| Check disk space and back up models, metadata, aligned history and ledger | **Proposed:** monthly and before model maintenance | Preserve all artifacts referenced by best-model metadata |
| Verify source endpoints and formats | **Proposed:** each flood season and after failures | Remote products can change |
| Review ledger against newly observed extent | **Proposed:** each season | Analyse error by lead and season/month |
| Rerun temporal ablation | When target product, feature/data sources, experiment grid or cutoff policy changes | A deployable success updates the best-model file |
| Fine-tune ConvLSTM | **To confirm:** after sufficient new finalized spatial history | Use section 6; deployment never performs this step |
| Fully retrain ConvLSTM | When reference grid, target product, spatial contract or forecast horizon changes | Run `python3 -m model.spatial.train_convlstm` |
| Rerun spatial validation | After every published spatial model change | Automatic when enabled, or run `python3 -m model.spatial.evaluate_spatial_model` |
| Regenerate basin attributes | Only when HydroATLAS inputs or processing choices change | Run `python3 -m processing.data_cleaning.process_basin_attributes --force` intentionally |

The selected temporal model is retrained on all currently labelled history at
every deployment run; no separate routine temporal retraining command is
needed.

## 9. Records

For each operational run record: start/end time, operator, target product,
source-data cutoff dates, model version, checks 1-11, waivers, issues and fixes,
reviewer, release decision and release time. Preserve:

- the run log and output directory;
- `predictions/data_preflight_before_updates.json` and
  `predictions/data_preflight.json`;
- `model/ablation/ablation_experiment_log.csv`, including failed experiments;
- `outputs.forecast_performance_csv`; and
- spatial training/fine-tuning metadata and validation reports.

## 10. Escalation

1. The operator retries once where the action in section 7 says a retry is
   safe.
2. The technical owner investigates. Do not release a failed or waived gate
   without reviewer sign-off.
3. If unresolved, contact Jessica Rapson at
   `jessica.rapson@algorithmicgovernance.org`. Code:
   <https://github.com/rapsoj/INFLOW-AI>. Archive DOI:
   <https://doi.org/10.5281/zenodo.18258570>.

## Appendix: key configuration (`config.yaml`)

- `runtime.target_product`: `viirs` or `modis` (best-model metadata is
  authoritative during deployment)
- `forecast.horizon_periods`: default 6; shared temporal/spatial horizon
- `forecast.interval_percent`: default 80
- `outputs.root`, `outputs.forecast_performance_csv`
- `data_preflight.*`: completeness thresholds and report paths
- `spatial_prediction.enabled`, paths, `minimum_population`, patch border and
  inference batch size
- `spatial_training.model_path`, input paths, sequence/patch settings,
  architecture, optimizer, checkpoint and validation split
- `spatial_fine_tuning.*`: source/output paths, lower learning rate, training
  window, layer freezing, stopping, backups and post-tuning validation
- `spatial_validation.*`: threshold and GloFAS baseline paths
- `sources.ecmwf.*`: archives, concurrency, retries, timeout, initialization
  age, forecast windows and HDF5 settings
- `paths.downloads`, `paths.historic`, `paths.maps`
- `ablation.artifacts.base_dir`
