# Repo Scope

- The actionable project lives in `cpg0016_pipeline/`. The repo root only contains `LICENSE`, `.git/`, and this folder, so run project commands from `cpg0016_pipeline/`.
- There is no repo-local CI, lint, formatter, or typecheck config. Do not invent `ruff`, `mypy`, or GitHub Actions checks that are not present.

# Setup And Verification

- Create the environment with `uv sync --group test` or `just setup` from `cpg0016_pipeline/`.
- The only checked-in automated test coverage is `uv run --group test pytest`, currently `tests/test_isolation_forest_scripts.py`.
- For focused verification of train/score changes, run `uv run --group test pytest tests/test_isolation_forest_scripts.py`.
- There is no lightweight automated coverage for the download, sampling, zarr, or MLflow/MorphEm stages; those stages are integration-heavy and require external services or cached assets.

# Real Entrypoints

- The primary operator interface is `cpg0016_pipeline/justfile`, not a package CLI.
- Supported stage order is fixed in `justfile`: `download-profiles -> download-load-data -> sample-cells -> make-zarr -> extract-morphem -> train-iforest -> score-iforest`.
- `just run-all` executes that exact sequence and trains then scores on the same `outputs/morphem_features.parquet`.
- `just show-paths` is the fastest way to confirm resolved output/input locations before a long run.

# Pipeline Contracts

- `scripts/run_cpg0016_source_profiles.sh` is the normal entrypoint for profile downloads; it hardcodes the supported sources (`source_1`, `source_10`, `source_11`, `source_13`, `source_15`, `source_2`-`source_9`) and calls `download_and_merge_cpg0016_source_profiles.py` per source.
- The profile download/merge stage writes `Nuclei_Image_merged.parquet` files and deletes successfully merged `Image.csv` and `Nuclei.csv` inputs afterward.
- `sample_anomalous_single_cells_fs.py` fetches perturbation metadata live from public JUMP metadata URLs at runtime; sampling is not fully offline or hermetic.
- `download_correct_crop_normalize_to_zarr.py` writes a strict-order zarr store from the sampled parquet and requires `URL_OrigDNA`, `URL_IllumDNA`, and bbox columns.
- `translate_and_extract_morphem.py` requires both an MLflow logged model ID and local cached Hugging Face files for `CaicedoLab/MorphEm`; it uses `snapshot_download(..., local_files_only=True)` and will not fetch the MorphEm model for you.
- Train/score scripts identify model inputs by the `morphem_feature_` prefix. Scoring expects the trained model to preserve `feature_names_in_` and requires `Metadata_JCP2022` for aggregation.

# External Dependencies And Gotchas

- `aws` CLI must be on `PATH` for the `download-profiles` stage.
- `imagecodecs` is a real runtime dependency for some TIFF decodes in the crop-generation stage.
- Default MLflow tracking URI in the code/justfile is `sqlite:////home/camo/projects/nuclear_speckles_analysis/mlflow.db`; override with `MLFLOW_TRACKING_URI` when that machine-local path is wrong.
- Path/runtime overrides are environment-driven in `justfile`: `OUTPUT_ROOT`, `DOWNLOADED_PROFILES_DIR`, `DOWNLOADED_CPG0016_CSV_DIR`, `CROP_WORK_DIR`, `MAX_CONCURRENT_REQUESTS`, `LOAD_DATA_WORKERS`, `CROP_DOWNLOAD_WORKERS`, `MODEL_ID`, `MLFLOW_TRACKING_URI`, `MORPHEM_BATCH_SIZE`, `MORPHEM_DEVICE`.
