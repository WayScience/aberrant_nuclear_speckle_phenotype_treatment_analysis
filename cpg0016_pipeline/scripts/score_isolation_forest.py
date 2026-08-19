#!/usr/bin/env python
"""Score a feature parquet with a trained Isolation Forest."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd


METADATA_PREFIX = "Metadata_"
POSITIONAL_PREFIX = "AreaShape_"
URL_PREFIX = "URL_"
RESULT_COLUMNS = ["Result_inlier", "Result_anomaly_score"]
AGGREGATION_GROUP_COLUMN = "Metadata_JCP2022"
AGGREGATED_SCORE_COLUMN = "Result_anomaly_score_mean"
AGGREGATED_COUNT_COLUMN = "sample_count"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for model scoring.

    Returns:
        Parsed command-line arguments containing the input feature parquet, the
        trained model path, and both output parquet destinations.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Score a parquet with a trained Isolation Forest, write a per-cell parquet, "
            "and write a Metadata_JCP2022-aggregated parquet."
        )
    )
    parser.add_argument(
        "--input-parquet",
        type=Path,
        required=True,
        help="Parquet file containing metadata, positional columns, URLs, and model features.",
    )
    parser.add_argument(
        "--input-model",
        type=Path,
        required=True,
        help="Trained Isolation Forest .joblib path.",
    )
    parser.add_argument(
        "--output-cell-parquet",
        type=Path,
        required=True,
        help="Destination parquet path for per-cell scores.",
    )
    parser.add_argument(
        "--output-jcp-parquet",
        type=Path,
        required=True,
        help="Destination parquet path for Metadata_JCP2022-level mean scores.",
    )
    return parser.parse_args()


def get_model_feature_columns(model: object) -> list[str]:
    """Extract the feature-name list stored on a fitted model.

    Args:
        model: Previously trained model object loaded from disk.

    Returns:
        Ordered feature names preserved on the trained model.

    Raises:
        ValueError: If the loaded model does not expose ``feature_names_in_``.
    """

    feature_columns = getattr(model, "feature_names_in_", None)
    if feature_columns is None:
        raise ValueError(
            "Loaded model does not expose feature_names_in_. Train the model with a "
            "DataFrame so score-time feature names are preserved."
        )
    return list(feature_columns)


def validate_scoring_inputs(dataframe: pd.DataFrame, feature_columns: list[str]) -> None:
    """Validate that the dataframe contains the model features and group column.

    Args:
        dataframe: Input dataframe to score.
        feature_columns: Feature columns expected by the trained model.

    Returns:
        None.

    Raises:
        ValueError: If any feature column is missing or if
            ``Metadata_JCP2022`` is absent.
    """

    missing_feature_columns = [
        column for column in feature_columns if column not in dataframe.columns
    ]
    if missing_feature_columns:
        raise ValueError(
            "Input parquet is missing model feature columns: "
            f"{missing_feature_columns}"
        )

    if AGGREGATION_GROUP_COLUMN not in dataframe.columns:
        raise ValueError(
            f"Input parquet is missing required aggregation column: {AGGREGATION_GROUP_COLUMN}"
        )


def score_cells(dataframe: pd.DataFrame, model: object) -> pd.DataFrame:
    """Apply the model to each row and append inlier/anomaly columns.

    Args:
        dataframe: Feature dataframe containing metadata and model inputs.
        model: Trained Isolation Forest model.

    Returns:
        Copy of the input dataframe with ``Result_inlier`` and
        ``Result_anomaly_score`` appended.
    """

    feature_columns = get_model_feature_columns(model)
    validate_scoring_inputs(dataframe=dataframe, feature_columns=feature_columns)

    scored_df = dataframe.copy()
    features_df = scored_df.loc[:, feature_columns]
    scored_df["Result_inlier"] = model.predict(features_df)
    scored_df["Result_anomaly_score"] = model.decision_function(features_df)
    return scored_df


def select_cell_output_columns(dataframe: pd.DataFrame) -> list[str]:
    """Keep metadata, positional, URL, and result columns for per-cell output.

    Args:
        dataframe: Scored dataframe containing original inputs plus results.

    Returns:
        Ordered subset of column names retained in the per-cell parquet.
    """

    keep_columns: list[str] = []
    for column in dataframe.columns:
        if column in RESULT_COLUMNS:
            keep_columns.append(column)
            continue
        if column.startswith(METADATA_PREFIX):
            keep_columns.append(column)
            continue
        if column.startswith(POSITIONAL_PREFIX):
            keep_columns.append(column)
            continue
        if column.startswith(URL_PREFIX):
            keep_columns.append(column)
    return keep_columns


def build_cell_output(scored_df: pd.DataFrame) -> pd.DataFrame:
    """Build the sorted per-cell scored parquet dataframe.

    Args:
        scored_df: Dataframe after model predictions and anomaly scores.

    Returns:
        Per-cell output dataframe restricted to metadata, position, URL, and
        result columns, sorted from most anomalous to least anomalous.
    """

    cell_output_df = scored_df.loc[:, select_cell_output_columns(scored_df)].copy()
    cell_output_df.sort_values(by="Result_anomaly_score", ascending=True, inplace=True)
    cell_output_df.reset_index(drop=True, inplace=True)
    return cell_output_df


def build_aggregated_output(scored_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate scored rows by ``Metadata_JCP2022``.

    Args:
        scored_df: Dataframe after model predictions and anomaly scores.

    Returns:
        Aggregated dataframe containing one row per ``Metadata_JCP2022`` value
        with mean anomaly score and sample count.
    """

    aggregated_df = (
        scored_df.groupby(AGGREGATION_GROUP_COLUMN, dropna=False)
        .agg(
            Result_anomaly_score_mean=("Result_anomaly_score", "mean"),
            sample_count=("Result_anomaly_score", "size"),
        )
        .reset_index()
    )
    aggregated_df.sort_values(by=AGGREGATED_SCORE_COLUMN, ascending=True, inplace=True)
    aggregated_df.reset_index(drop=True, inplace=True)
    return aggregated_df


def write_parquet(dataframe: pd.DataFrame, output_path: Path) -> None:
    """Write a dataframe to parquet, creating parent directories as needed.

    Args:
        dataframe: Dataframe to serialize.
        output_path: Destination parquet path.

    Returns:
        None.
    """

    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_parquet(output_path, index=False)


def main() -> None:
    """Score one feature parquet and write both detailed and aggregated outputs.

    Returns:
        None.

    This entrypoint reads the same MorphEm feature parquet used by training,
    applies the trained Isolation Forest, and writes both per-cell and
    Metadata_JCP2022-aggregated parquet outputs.
    """

    args = parse_args()
    dataframe = pd.read_parquet(args.input_parquet)
    model = joblib.load(args.input_model)
    if hasattr(model, "n_jobs"):
        model.n_jobs = -1

    scored_df = score_cells(dataframe=dataframe, model=model)
    cell_output_df = build_cell_output(scored_df)
    aggregated_output_df = build_aggregated_output(scored_df)

    write_parquet(cell_output_df, args.output_cell_parquet)
    write_parquet(aggregated_output_df, args.output_jcp_parquet)

    print(f"Saved {len(cell_output_df)} scored cells to {args.output_cell_parquet}")
    print(
        f"Saved {len(aggregated_output_df)} Metadata_JCP2022 aggregates to "
        f"{args.output_jcp_parquet}"
    )


if __name__ == "__main__":
    main()
