"""Contract tests for the standalone Isolation Forest training and scoring scripts.

These tests load the scripts as modules so they can validate the functional
contracts without depending on the CLI wrappers. The assertions focus on the
pipeline behavior that downstream stages rely on: MorphEm feature selection,
model feature-name preservation, output column filtering, and
Metadata_JCP2022-level aggregation.
"""

import importlib.util
import json
from pathlib import Path

import joblib
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT_PATH = REPO_ROOT / "scripts" / "train_isolation_forest.py"
SCORE_SCRIPT_PATH = REPO_ROOT / "scripts" / "score_isolation_forest.py"


def _load_module(module_name: str, module_path: Path):
    """Load a Python module directly from a script path for test access.

    Args:
        module_name: Synthetic module name to assign during loading.
        module_path: Filesystem path to the script to load.

    Returns:
        The imported module object.

    Raises:
        RuntimeError: If the script cannot be loaded as a module.
    """

    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


train_isolation_forest = _load_module(
    "train_isolation_forest",
    TRAIN_SCRIPT_PATH,
)
score_isolation_forest = _load_module(
    "score_isolation_forest",
    SCORE_SCRIPT_PATH,
)


def _build_test_dataframe() -> pd.DataFrame:
    """Construct a compact feature table that exercises training and scoring.

    Returns:
        A dataframe containing metadata, positional columns, URL columns, two
        MorphEm-style feature columns, and one extra column that should be
        filtered out of the per-cell scoring output.
    """

    return pd.DataFrame(
        {
            "Metadata_JCP2022": ["JCP_A", "JCP_A", "JCP_B", "JCP_B", "JCP_B", "JCP_C"],
            "Metadata_Plate": ["plate_1"] * 6,
            "Metadata_Well": ["A01", "A01", "B01", "B01", "B01", "C01"],
            "Metadata_Site": [1, 2, 1, 2, 3, 1],
            "Metadata_Source": ["source_10"] * 6,
            "AreaShape_Center_X": [10.0, 11.0, 20.0, 21.0, 22.0, 30.0],
            "AreaShape_Center_Y": [15.0, 16.0, 25.0, 26.0, 27.0, 35.0],
            "AreaShape_BoundingBoxMinimum_X": [5, 6, 15, 16, 17, 25],
            "AreaShape_BoundingBoxMaximum_X": [15, 16, 25, 26, 27, 35],
            "AreaShape_BoundingBoxMinimum_Y": [10, 11, 20, 21, 22, 30],
            "AreaShape_BoundingBoxMaximum_Y": [20, 21, 30, 31, 32, 40],
            "URL_OrigDNA": [f"s3://bucket/dna_{index}.tif" for index in range(6)],
            "URL_IllumDNA": [f"s3://bucket/illum_{index}.npy" for index in range(6)],
            "morphem_feature_0000": [0.0, 0.1, 3.0, 3.1, 3.2, 6.0],
            "morphem_feature_0001": [0.2, 0.3, 3.3, 3.4, 3.5, 6.2],
            "OtherColumn": list("abcdef"),
        }
    )


def test_training_script_selects_morphem_features_and_writes_sidecar(tmp_path) -> None:
    """Verify training selects only MorphEm columns and preserves them in artifacts.

    Args:
        tmp_path: Temporary directory provided by pytest for artifact writes.

    Returns:
        None.

    This test protects the contract that training selects columns by the
    MorphEm prefix, fits the model with those exact feature names, and writes a
    JSON sidecar that matches the trained feature list.
    """

    dataframe = _build_test_dataframe()
    output_model = tmp_path / "model.joblib"
    output_feature_columns = tmp_path / "feature_columns.json"

    feature_columns = train_isolation_forest.select_feature_columns(dataframe)
    assert feature_columns == ["morphem_feature_0000", "morphem_feature_0001"]

    model = train_isolation_forest.train_isolation_forest_model(dataframe, feature_columns)
    joblib.dump(model, output_model)
    train_isolation_forest.write_feature_columns(feature_columns, output_feature_columns)

    assert output_model.exists()
    assert json.loads(output_feature_columns.read_text(encoding="utf-8")) == feature_columns
    assert list(model.feature_names_in_) == feature_columns


def test_scoring_script_keeps_metadata_area_urls_and_aggregates_by_jcp2022(tmp_path) -> None:
    """Verify scoring keeps required columns and writes correct aggregated output.

    Args:
        tmp_path: Temporary directory provided by pytest for parquet writes.

    Returns:
        None.

    This test protects the contract that scoring drops unrelated columns from
    the per-cell output, preserves metadata, positional, and URL columns, and
    writes a Metadata_JCP2022-level aggregation consistent with the cell-level
    anomaly scores.
    """

    dataframe = _build_test_dataframe()
    feature_columns = train_isolation_forest.select_feature_columns(dataframe)
    model = train_isolation_forest.train_isolation_forest_model(dataframe, feature_columns)

    scored_df = score_isolation_forest.score_cells(dataframe=dataframe, model=model)
    cell_output_df = score_isolation_forest.build_cell_output(scored_df)
    aggregated_output_df = score_isolation_forest.build_aggregated_output(scored_df)

    cell_output_path = tmp_path / "scored_cells.parquet"
    aggregated_output_path = tmp_path / "scored_by_jcp.parquet"
    score_isolation_forest.write_parquet(cell_output_df, cell_output_path)
    score_isolation_forest.write_parquet(aggregated_output_df, aggregated_output_path)

    reloaded_cell_output_df = pd.read_parquet(cell_output_path)
    reloaded_aggregated_output_df = pd.read_parquet(aggregated_output_path)

    assert "OtherColumn" not in reloaded_cell_output_df.columns
    assert "Metadata_JCP2022" in reloaded_cell_output_df.columns
    assert "AreaShape_Center_X" in reloaded_cell_output_df.columns
    assert "AreaShape_Center_Y" in reloaded_cell_output_df.columns
    assert "URL_OrigDNA" in reloaded_cell_output_df.columns
    assert "URL_IllumDNA" in reloaded_cell_output_df.columns
    assert "Result_inlier" in reloaded_cell_output_df.columns
    assert "Result_anomaly_score" in reloaded_cell_output_df.columns

    assert list(reloaded_aggregated_output_df.columns) == [
        "Metadata_JCP2022",
        "Result_anomaly_score_mean",
        "sample_count",
    ]
    assert set(reloaded_aggregated_output_df["Metadata_JCP2022"]) == {"JCP_A", "JCP_B", "JCP_C"}

    expected_aggregated_df = (
        reloaded_cell_output_df.groupby("Metadata_JCP2022", dropna=False)
        .agg(
            Result_anomaly_score_mean=("Result_anomaly_score", "mean"),
            sample_count=("Result_anomaly_score", "size"),
        )
        .reset_index()
        .sort_values(by="Metadata_JCP2022")
        .reset_index(drop=True)
    )
    observed_aggregated_df = reloaded_aggregated_output_df.sort_values(
        by="Metadata_JCP2022"
    ).reset_index(drop=True)

    pd.testing.assert_frame_equal(observed_aggregated_df, expected_aggregated_df)
