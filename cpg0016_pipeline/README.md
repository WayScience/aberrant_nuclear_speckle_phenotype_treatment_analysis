# cpg0016_morphem_pipeline

Standalone repo-style folder for the CPG0016 single-cell MorphEm and Isolation Forest pipeline.

## What It Includes

- Per-source CPG0016 `Image.csv` and `Nuclei.csv` download plus merge to parquet.
- `load_data_with_illum.csv` download needed by sampling.
- Single-cell sampling with perturbation metadata joins.
- DNA image download, illumination correction, cropping, and normalization to zarr.
- MLflow translation plus MorphEm embedding extraction.
- Isolation Forest training.
- Isolation Forest scoring on the same feature parquet used for training.

## Folder Layout

```text
cpg0016_morphem_pipeline/
  README.md
  pyproject.toml
  justfile
  scripts/
  src/
  tests/
```

## Setup

From inside `cpg0016_morphem_pipeline/`:

```bash
uv sync --group test
```

Or with `just`:

```bash
just setup
```

## External Requirements

- `aws` CLI on `PATH` for `workspace/analysis` CSV downloads.
- Access to the MLflow tracking backend used by the translation model.
- A valid MLflow logged model identifier for the translation model.
- Local cached MorphEm model files because `translate_and_extract_morphem.py` uses `snapshot_download(..., local_files_only=True)`.
- Optional CUDA if you want GPU execution.
- `imagecodecs` may be required at runtime to decode some TIFF compressions.

## Pipeline Order

The supported end-to-end order matches the `justfile` stage names:

1. `download-profiles`
2. `download-load-data`
3. `sample-cells`
4. `make-zarr`
5. `extract-morphem`
6. `train-iforest`
7. `score-iforest`

Those recipes call these scripts in order:

1. `run_cpg0016_source_profiles.sh`
2. `download_cpg0016_load_data_with_illum.py`
3. `sample_anomalous_single_cells_fs.py`
4. `download_correct_crop_normalize_to_zarr.py`
5. `translate_and_extract_morphem.py`
6. `train_isolation_forest.py`
7. `score_isolation_forest.py`

The wrapper `run_cpg0016_source_profiles.sh` already calls `download_and_merge_cpg0016_source_profiles.py` internally, so you do not run both separately in the normal pipeline.

## Default Paths

By default the `justfile` uses:

- `downloaded_profiles_csvs/`
- `downloaded_cpg0016_csvs/`
- `outputs/sampled_anomalous_single_cells.parquet`
- `outputs/normalized_crops.zarr`
- `outputs/morphem_features.parquet`
- `outputs/isolation_forest.joblib`
- `outputs/isolation_forest_feature_columns.json`
- `outputs/scored_cells.parquet`
- `outputs/scored_by_jcp.parquet`
- `temp_single_cell_crop_work/`

Show the currently resolved paths with:

```bash
just show-paths
```

## Overriding Paths And Runtime Settings

You can override output paths and selected runtime settings in two ways.

### Direct script flags

Every script exposes its own output path directly with command-line arguments such as:

- `--output-dir`
- `--output-parquet`
- `--output-zarr`
- `--output-model`
- `--output-feature-columns`
- `--output-cell-parquet`
- `--output-jcp-parquet`
- `--work-dir`

Example:

```bash
uv run python scripts/sample_anomalous_single_cells_fs.py \
  --downloaded-profiles-dir my_profiles \
  --load-data-csv-dir my_load_data \
  --output-parquet custom_outputs/sample.parquet
```

### `just` environment overrides

The `justfile` reads these path and runtime environment variables:

- `OUTPUT_ROOT`
- `DOWNLOADED_PROFILES_DIR`
- `DOWNLOADED_CPG0016_CSV_DIR`
- `CROP_WORK_DIR`
- `MAX_CONCURRENT_REQUESTS`
- `LOAD_DATA_WORKERS`
- `CROP_DOWNLOAD_WORKERS`
- `MODEL_ID`
- `MLFLOW_TRACKING_URI`
- `MORPHEM_BATCH_SIZE`
- `MORPHEM_DEVICE`

Example using custom output locations:

```bash
OUTPUT_ROOT=custom_outputs \
DOWNLOADED_PROFILES_DIR=custom_profiles \
DOWNLOADED_CPG0016_CSV_DIR=custom_load_data \
CROP_WORK_DIR=custom_crop_work \
MODEL_ID=my-logged-model-id \
MLFLOW_TRACKING_URI=sqlite:////path/to/mlflow.db \
just run-all
```

## Common `just` Commands

Inspect the resolved default or overridden paths:

```bash
just show-paths
```

Run stages individually:

```bash
just download-profiles
just download-load-data
just sample-cells
just make-zarr
MODEL_ID=my-logged-model-id just extract-morphem
just train-iforest
just score-iforest
```

Run the full pipeline end to end:

```bash
MODEL_ID=my-logged-model-id just run-all
```

The `run-all` recipe trains and scores on the same `outputs/morphem_features.parquet` input, matching the intended workflow for this standalone folder.

## Direct Script Usage

### 1. Download and merge per-source analysis profiles

```bash
bash scripts/run_cpg0016_source_profiles.sh --output-dir downloaded_profiles_csvs
```

### 2. Download `load_data_with_illum.csv`

```bash
uv run python scripts/download_cpg0016_load_data_with_illum.py \
  --output-dir downloaded_cpg0016_csvs
```

### 3. Sample single cells

```bash
uv run python scripts/sample_anomalous_single_cells_fs.py \
  --downloaded-profiles-dir downloaded_profiles_csvs \
  --load-data-csv-dir downloaded_cpg0016_csvs \
  --output-parquet outputs/sampled_anomalous_single_cells.parquet
```

### 4. Build the normalized crop zarr store

```bash
uv run python scripts/download_correct_crop_normalize_to_zarr.py \
  --input-parquet outputs/sampled_anomalous_single_cells.parquet \
  --output-zarr outputs/normalized_crops.zarr \
  --work-dir temp_single_cell_crop_work
```

### 5. Translate and extract MorphEm embeddings

```bash
uv run python scripts/translate_and_extract_morphem.py \
  --input-zarr outputs/normalized_crops.zarr \
  --input-parquet outputs/sampled_anomalous_single_cells.parquet \
  --output-parquet outputs/morphem_features.parquet \
  --model-id my-logged-model-id \
  --mlflow-tracking-uri sqlite:////path/to/mlflow.db
```

### 6. Train Isolation Forest

```bash
uv run python scripts/train_isolation_forest.py \
  --input-parquet outputs/morphem_features.parquet \
  --output-model outputs/isolation_forest.joblib \
  --output-feature-columns outputs/isolation_forest_feature_columns.json
```

### 7. Score the MorphEm feature parquet used for training

```bash
uv run python scripts/score_isolation_forest.py \
  --input-parquet outputs/morphem_features.parquet \
  --input-model outputs/isolation_forest.joblib \
  --output-cell-parquet outputs/scored_cells.parquet \
  --output-jcp-parquet outputs/scored_by_jcp.parquet
```

## Output Contracts

- `run_cpg0016_source_profiles.sh` is the wrapper stage used by `just download-profiles`; it invokes `download_and_merge_cpg0016_source_profiles.py` for each supported source.
- `download_and_merge_cpg0016_source_profiles.py` writes `Nuclei_Image_merged.parquet` files and deletes the successfully merged `Image.csv` and `Nuclei.csv` inputs.
- `sample_anomalous_single_cells_fs.py` writes one sampled parquet that includes DNA and illumination URLs.
- `download_correct_crop_normalize_to_zarr.py` writes a zarr store with strict row-order metadata in attrs.
- `translate_and_extract_morphem.py` verifies parquet and zarr alignment before writing feature columns named `morphem_feature_####`.
- `train_isolation_forest.py` writes a `.joblib` model and a JSON sidecar listing feature columns.
- `score_isolation_forest.py` writes both per-cell and `Metadata_JCP2022` aggregated outputs.

## Testing

Run the standalone tests with:

```bash
uv run --group test pytest
```
