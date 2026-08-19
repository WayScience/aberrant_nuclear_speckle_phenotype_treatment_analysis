#!/usr/bin/env python
"""Train an Isolation Forest model from MorphEm feature columns."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import IsolationForest


DEFAULT_FEATURE_PREFIX = "morphem_feature_"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for Isolation Forest training.

    Returns:
        Parsed command-line arguments containing the input parquet path, model
        output path, optional feature-column sidecar path, and feature prefix.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Train one dataset-level Isolation Forest from a parquet containing "
            "MorphEm feature columns."
        )
    )
    parser.add_argument(
        "--input-parquet",
        type=Path,
        required=True,
        help="Parquet file containing MorphEm feature columns and metadata.",
    )
    parser.add_argument(
        "--output-model",
        type=Path,
        required=True,
        help="Destination .joblib path for the trained Isolation Forest.",
    )
    parser.add_argument(
        "--output-feature-columns",
        type=Path,
        help=(
            "Optional destination JSON path for the trained feature column list. "
            "Defaults to <output-model stem>_feature_columns.json."
        ),
    )
    parser.add_argument(
        "--feature-prefix",
        type=str,
        default=DEFAULT_FEATURE_PREFIX,
        help="Column prefix used to select model feature columns.",
    )
    return parser.parse_args()


def get_default_feature_columns_path(output_model: Path) -> Path:
    """Build the default feature-column sidecar path for a model file.

    Args:
        output_model: Destination path for the trained model artifact.

    Returns:
        JSON path located next to the model file and named with the model stem
        plus ``_feature_columns.json``.
    """

    return output_model.with_name(f"{output_model.stem}_feature_columns.json")


def select_feature_columns(
    dataframe: pd.DataFrame,
    feature_prefix: str = DEFAULT_FEATURE_PREFIX,
) -> list[str]:
    """Select feature columns by prefix from the input dataframe.

    Args:
        dataframe: Feature dataframe containing metadata plus MorphEm columns.
        feature_prefix: Prefix used to identify model input columns.

    Returns:
        Ordered list of dataframe columns whose names start with
        ``feature_prefix``.

    Raises:
        ValueError: If no matching feature columns are found.
    """

    feature_columns = [
        column for column in dataframe.columns if column.startswith(feature_prefix)
    ]
    if not feature_columns:
        raise ValueError(
            "No feature columns were found with prefix "
            f"{feature_prefix!r} in input parquet."
        )
    return feature_columns


def train_isolation_forest_model(
    dataframe: pd.DataFrame,
    feature_columns: list[str],
) -> IsolationForest:
    """Fit an Isolation Forest using the requested feature columns.

    Args:
        dataframe: Input dataframe containing the model features.
        feature_columns: Column names to use as the model design matrix.

    Returns:
        A fitted ``IsolationForest`` instance whose ``feature_names_in_``
        attribute preserves the training column order.

    Raises:
        ValueError: If any requested feature column is missing from the
            dataframe.
    """

    missing_columns = [column for column in feature_columns if column not in dataframe.columns]
    if missing_columns:
        raise ValueError(f"Input parquet is missing feature columns: {missing_columns}")

    model = IsolationForest(n_estimators=800, random_state=0, n_jobs=-1)
    model.fit(dataframe.loc[:, feature_columns])
    return model


def write_feature_columns(feature_columns: list[str], output_path: Path) -> None:
    """Write the trained feature-column list to JSON.

    Args:
        feature_columns: Ordered feature names used to train the model.
        output_path: Destination JSON path for the sidecar file.

    Returns:
        None.
    """

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file_handle:
        json.dump(feature_columns, file_handle, indent=2)


def main() -> None:
    """Train a model from one input parquet and write model artifacts.

    Returns:
        None.

    This entrypoint reads one MorphEm feature parquet, selects the model input
    columns by prefix, writes the trained ``.joblib`` model, and emits a JSON
    sidecar listing the feature columns required for later scoring.
    """

    args = parse_args()
    dataframe = pd.read_parquet(args.input_parquet)
    feature_columns = select_feature_columns(
        dataframe=dataframe,
        feature_prefix=args.feature_prefix,
    )
    model = train_isolation_forest_model(
        dataframe=dataframe,
        feature_columns=feature_columns,
    )

    args.output_model.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.output_model)

    feature_columns_path = (
        args.output_feature_columns
        if args.output_feature_columns is not None
        else get_default_feature_columns_path(args.output_model)
    )
    write_feature_columns(feature_columns, feature_columns_path)

    print(
        f"Saved Isolation Forest trained on {len(feature_columns)} features to "
        f"{args.output_model}"
    )
    print(f"Saved feature columns to {feature_columns_path}")


if __name__ == "__main__":
    main()
