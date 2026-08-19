#!/usr/bin/env python
"""Download CPG0016 ``load_data_with_illum.csv`` files for the pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cpg0016_morphem_pipeline.cpg0016 import CPG0016LoadDataWithIllumDownloader


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for load-data CSV download.

    Returns:
        Parsed command-line arguments containing the destination directory,
        overwrite behavior, worker count, and parallel execution mode.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Download all CPG0016 load_data_with_illum.csv files needed by the "
            "single-cell sampling stage."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("downloaded_cpg0016_csvs"),
        help="Local root directory for downloaded load_data_with_illum.csv files.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-download existing local CSV files.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Worker count to use for metadata CSV downloads.",
    )
    parser.add_argument(
        "--parallel",
        dest="parallel",
        action="store_true",
        help="Download metadata CSVs in parallel.",
    )
    parser.add_argument(
        "--no-parallel",
        dest="parallel",
        action="store_false",
        help="Download metadata CSVs serially.",
    )
    parser.set_defaults(parallel=True)
    return parser.parse_args()


def main() -> None:
    """Download load-data metadata files and print a summary.

    Returns:
        None.

    This entrypoint writes the mirrored ``load_data_with_illum.csv`` tree under
    the requested output directory and prints the download summary used to
    confirm whether all metadata prerequisites were materialized successfully.
    """

    args = parse_args()
    downloader = CPG0016LoadDataWithIllumDownloader(
        csv_download_dir=args.output_dir,
        overwrite=args.overwrite,
        workers=args.workers,
        parallel=args.parallel,
        verbose=True,
    )
    summary = downloader.csv_download_summary
    print(f"Downloaded CSV root: {args.output_dir}")
    print(f"Total jobs: {summary.total_jobs}")
    print(f"Downloaded: {summary.downloaded}")
    print(f"Skipped: {summary.skipped}")
    print(f"Failed: {summary.failed}")


if __name__ == "__main__":
    main()
