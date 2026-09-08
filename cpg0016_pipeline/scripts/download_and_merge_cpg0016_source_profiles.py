#!/usr/bin/env python
"""Download Image/Nuclei CSVs for one source and merge them into parquets."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cpg0016_morphem_pipeline.cpg0016 import CPG0016AnalysisCSVDownloader
from cpg0016_morphem_pipeline.cpg0016.analysis_csv_downloader import (
    extract_metadata_from_dataset_path,
)


IMAGE_KEEP_COLUMNS = [
    "ImageNumber",
    "Metadata_Plate",
    "Metadata_Well",
    "Metadata_Site",
    "Metadata_Source",
    "Metadata_Batch",
]
NUCLEI_KEEP_COLUMNS = [
    "ImageNumber",
    "AreaShape_BoundingBoxMinimum_X",
    "AreaShape_BoundingBoxMaximum_X",
    "AreaShape_BoundingBoxMinimum_Y",
    "AreaShape_BoundingBoxMaximum_Y",
    "AreaShape_Center_X",
    "AreaShape_Center_Y",
]
MERGED_PARQUET_FILENAME = "Nuclei_Image_merged.parquet"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for one-source profile download and merge.

    Returns:
        Parsed command-line arguments containing the source identifier, output
        directory, AWS CLI parallel mode, and maximum S3 concurrency.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Download Image.csv and Nuclei.csv for one CPG0016 source, trim them in place, "
            "merge on ImageNumber, write parquet, and remove the input CSVs after a successful merge."
        )
    )
    parser.add_argument("--source", required=True, help="One source name such as source_10.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("downloaded_profiles_csvs"),
        help="Local root directory for downloaded analysis CSVs.",
    )
    parser.add_argument(
        "--parallel",
        dest="parallel",
        action="store_true",
        help="Enable parallel AWS CLI transfers.",
    )
    parser.add_argument(
        "--no-parallel",
        dest="parallel",
        action="store_false",
        help="Disable parallel AWS CLI transfers.",
    )
    parser.set_defaults(parallel=True)
    parser.add_argument(
        "--max-concurrent-requests",
        type=int,
        default=50,
        help="Maximum AWS CLI concurrent requests when parallel downloads are enabled.",
    )
    return parser.parse_args()


def validate_source(source: str) -> str:
    """Validate and normalize a requested CPG0016 source name.

    Args:
        source: Raw source string supplied by the caller.

    Returns:
        Normalized source name after surrounding whitespace is removed.

    Raises:
        ValueError: If the source is ``source_all`` or does not look like an
            explicit ``source_<id>`` dataset segment.
    """

    normalized_source = source.strip()
    if normalized_source == "source_all":
        raise ValueError("source_all is not supported. Use one explicit source such as 'source_10'.")
    if not normalized_source.startswith("source_"):
        raise ValueError("Invalid source. Expected one explicit source such as 'source_10'.")
    return normalized_source


def trim_image_csv(image_path: Path) -> bool:
    """Trim Image.csv to just the columns needed by downstream pipeline stages.

    Args:
        image_path: Path to one downloaded ``Image.csv`` file.

    Returns:
        ``True`` when at least one requested keep column was present and the CSV
        was rewritten in place, otherwise ``False``.
    """

    image_df = pd.read_csv(image_path)
    metadata = extract_metadata_from_dataset_path(image_path)
    image_df["Metadata_Source"] = metadata["Metadata_Source"]
    image_df["Metadata_Batch"] = metadata["Metadata_Batch"]
    image_columns = [column for column in IMAGE_KEEP_COLUMNS if column in image_df.columns]
    if not image_columns:
        return False
    image_df.loc[:, image_columns].to_csv(image_path, index=False)
    return True


def trim_nuclei_csv(nuclei_path: Path) -> bool:
    """Trim Nuclei.csv to bbox and center columns needed downstream.

    Args:
        nuclei_path: Path to one downloaded ``Nuclei.csv`` file.

    Returns:
        ``True`` when at least one requested keep column was present and the CSV
        was rewritten in place, otherwise ``False``.
    """

    nuclei_df = pd.read_csv(nuclei_path)
    nuclei_columns = [column for column in NUCLEI_KEEP_COLUMNS if column in nuclei_df.columns]
    if not nuclei_columns:
        return False
    nuclei_df.loc[:, nuclei_columns].to_csv(nuclei_path, index=False)
    return True


def merge_csvs_to_parquet(image_path: Path, nuclei_path: Path) -> bool:
    """Merge one Image/Nuclei CSV pair on ``ImageNumber`` and write parquet.

    Args:
        image_path: Path to the trimmed ``Image.csv`` file.
        nuclei_path: Path to the trimmed ``Nuclei.csv`` file.

    Returns:
        ``True`` when the merge succeeds and the parquet is written. Returns
        ``False`` when either input is missing ``ImageNumber``.

    Side Effects:
        Writes ``Nuclei_Image_merged.parquet`` next to the input CSVs and
        deletes the two source CSV files after a successful merge.
    """

    image_df = pd.read_csv(image_path)
    nuclei_df = pd.read_csv(nuclei_path)
    if "ImageNumber" not in image_df.columns or "ImageNumber" not in nuclei_df.columns:
        return False
    merged_df = pd.merge(left=nuclei_df, right=image_df, how="inner", on="ImageNumber")
    merged_path = image_path.parent / MERGED_PARQUET_FILENAME
    merged_df.to_parquet(merged_path, index=False)
    image_path.unlink()
    nuclei_path.unlink()
    return True


def main() -> None:
    """Download one source's profile CSVs, merge them, and print a summary.

    Returns:
        None.

    This entrypoint downloads ``Image.csv`` and ``Nuclei.csv`` files for one
    explicit source, trims them to the schema needed downstream, merges them on
    ``ImageNumber``, and reports how many files were rewritten, skipped, and
    converted to parquet.
    """

    args = parse_args()
    source = validate_source(args.source)

    downloader = CPG0016AnalysisCSVDownloader(
        output_dir=args.output_dir,
        parallel=args.parallel,
        max_concurrent_requests=args.max_concurrent_requests,
    )
    download_summary = downloader.download_all_csv_profiles(
        csv_names=["image", "nuclei"],
        sources=[source],
    )

    image_rewritten = 0
    nuclei_rewritten = 0
    nuclei_skipped = 0
    merge_skipped = 0
    parquet_written = 0
    csv_pairs_deleted = 0

    for csv_set in downloader.iter_analysis_csv_sets():
        image_path = csv_set.image_local_path
        nuclei_path = csv_set.nuclei_local_path

        if image_path is None or nuclei_path is None:
            merge_skipped += 1
            continue

        if not trim_image_csv(image_path):
            merge_skipped += 1
            continue
        image_rewritten += 1

        if not trim_nuclei_csv(nuclei_path):
            nuclei_skipped += 1
            merge_skipped += 1
            continue
        nuclei_rewritten += 1

        if not merge_csvs_to_parquet(image_path, nuclei_path):
            merge_skipped += 1
            continue

        parquet_written += 1
        csv_pairs_deleted += 1

    print(f"Source: {source}")
    print(f"Download summary: {download_summary}")
    print(f"Image.csv files rewritten: {image_rewritten}")
    print(f"Nuclei.csv files rewritten: {nuclei_rewritten}")
    print(f"Nuclei.csv files skipped: {nuclei_skipped}")
    print(f"Merged parquet files written: {parquet_written}")
    print(f"Merge operations skipped: {merge_skipped}")
    print(f"CSV pairs deleted after merge: {csv_pairs_deleted}")


if __name__ == "__main__":
    main()
