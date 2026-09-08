#!/usr/bin/env python
"""Download DNA images, crop them, normalize them, and write an ordered zarr."""

from __future__ import annotations

import argparse
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
import sys
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import tifffile
import zarr
from numcodecs import Blosc


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cpg0016_morphem_pipeline.cpg0016 import DownloadJob
from cpg0016_morphem_pipeline.cpg0016.load_data_with_illum_downloader import run_download_jobs


CROP_SIZE = 128
LOWER_PERCENTILE = 1.0
UPPER_PERCENTILE = 99.0
NORMALIZATION_HISTOGRAM_BINS = 8192
REQUIRED_COLUMNS = [
    "URL_OrigDNA",
    "URL_IllumDNA",
    "AreaShape_BoundingBoxMinimum_X",
    "AreaShape_BoundingBoxMaximum_X",
    "AreaShape_BoundingBoxMinimum_Y",
    "AreaShape_BoundingBoxMaximum_Y",
]


@dataclass(frozen=True)
class NormalizationConfig:
    """Persisted normalization parameters for robust percentile min-max scaling.

    Attributes:
        method: Human-readable normalization method identifier.
        lower_percentile: Lower percentile used to estimate the clipping bound.
        upper_percentile: Upper percentile used to estimate the clipping bound.
        lower_value: Estimated lower clipping value.
        upper_value: Estimated upper clipping value.
    """

    method: str
    lower_percentile: float
    upper_percentile: float
    lower_value: float
    upper_value: float


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for zarr crop generation.

    Returns:
        Parsed command-line arguments containing the sampled parquet path, zarr
        output path, scratch work directory, worker count, and overwrite mode.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Download DNA images and illumination files from a sampled parquet, "
            "apply illumination correction, compute 128x128 bbox-midpoint-centered crops, "
            "normalize them with dataset-level robust percentiles, and write a strict-order zarr store."
        )
    )
    parser.add_argument(
        "--input-parquet",
        type=Path,
        required=True,
        help="Parquet file containing URL_OrigDNA, URL_IllumDNA, and bbox columns.",
    )
    parser.add_argument(
        "--output-zarr",
        type=Path,
        required=True,
        help="Destination zarr store path for normalized crops.",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=Path("temp_single_cell_crop_work"),
        help="Scratch directory used for downloads and temporary corrected crops.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Worker count for S3 downloads.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing output zarr store and re-download files.",
    )
    return parser.parse_args()


def validate_input_columns(dataframe: pd.DataFrame) -> None:
    """Validate that all crop-generation columns are present.

    Args:
        dataframe: Sampled single-cell dataframe used to drive crop downloads.

    Returns:
        None.

    Raises:
        ValueError: If any required URL or bounding-box column is missing.
    """

    missing_columns = [column for column in REQUIRED_COLUMNS if column not in dataframe.columns]
    if missing_columns:
        raise ValueError(f"Input parquet is missing required columns: {missing_columns}")


def _filename_from_s3_url(s3_url: str) -> str:
    """Extract a filename from an S3 URL.

    Args:
        s3_url: Fully qualified S3 URL in ``s3://bucket/key`` form.

    Returns:
        Final filename segment from the URL path.

    Raises:
        ValueError: If the input is not a valid S3 URL or has no filename.
    """

    parsed = urlparse(s3_url)
    if parsed.scheme != "s3" or not parsed.netloc or not parsed.path:
        raise ValueError(f"Invalid S3 URL: {s3_url}")
    filename = Path(parsed.path).name
    if not filename:
        raise ValueError(f"S3 URL does not contain a filename: {s3_url}")
    return filename


def _compute_shifted_window(start: int, end: int, target_size: int, axis_limit: int) -> tuple[int, int]:
    """Compute an in-bounds crop window centered on the bbox midpoint.

    Args:
        start: Bounding-box start coordinate along one axis.
        end: Bounding-box end coordinate along one axis.
        target_size: Desired crop size along the same axis.
        axis_limit: Image size along the same axis.

    Returns:
        Tuple of ``(window_start, window_end)`` coordinates.

    Raises:
        ValueError: If the axis or target size is invalid, or if strict-mode
            cropping would require padding.
    """

    if axis_limit <= 0:
        raise ValueError(f"axis_limit must be positive, got {axis_limit}")
    if target_size <= 0:
        raise ValueError(f"target_size must be positive, got {target_size}")
    if target_size > axis_limit:
        raise ValueError(
            f"target_size {target_size} exceeds image axis limit {axis_limit}; strict mode does not pad"
        )

    center = (start + end) / 2.0
    window_start = int(round(center - target_size / 2.0))
    window_start = min(max(window_start, 0), axis_limit - target_size)
    window_end = window_start + target_size
    return window_start, window_end


def _apply_illumination_correction(orig_path: Path, illum_path: Path) -> np.ndarray:
    """Divide an original DNA image by its illumination correction array.

    Args:
        orig_path: Path to the downloaded DNA TIFF image.
        illum_path: Path to the downloaded illumination ``.npy`` array.

    Returns:
        Illumination-corrected image as a ``float32`` array.

    Raises:
        RuntimeError: If TIFF decompression requires ``imagecodecs``.
        ValueError: If dimensionality, shapes, or illumination values are
            invalid.
    """

    try:
        orig = tifffile.imread(orig_path).astype(np.float32)
    except ValueError as exc:
        if "requires the 'imagecodecs' package" in str(exc):
            raise RuntimeError(
                "This TIFF uses a compression codec that tifffile cannot decode without imagecodecs. "
                "Install it in the active environment and rerun the script."
            ) from exc
        raise

    illum = np.load(illum_path).astype(np.float32)
    if orig.ndim != 2:
        raise ValueError(f"Expected a 2D image at {orig_path}, got shape {orig.shape}")
    if illum.ndim != 2:
        raise ValueError(f"Expected a 2D illumination array at {illum_path}, got shape {illum.shape}")
    if orig.shape != illum.shape:
        raise ValueError(
            f"Shape mismatch for {orig_path.name}: image {orig.shape} vs illumination {illum.shape}"
        )
    if np.any(illum <= 0):
        raise ValueError(f"Illumination array contains non-positive values: {illum_path}")

    return orig / illum


def _sample_id(row: pd.Series, bbox_center_x: float, bbox_center_y: float) -> str:
    """Build a stable row identifier shared between parquet and zarr outputs.

    Args:
        row: Sampled parquet row containing plate, well, and site metadata.
        bbox_center_x: Bounding-box midpoint x-coordinate.
        bbox_center_y: Bounding-box midpoint y-coordinate.

    Returns:
        Stable sample identifier used to verify row alignment across artifacts.
    """

    sample_parts: list[str] = []
    for column in ["Metadata_Plate", "Metadata_Well", "Metadata_Site"]:
        if column in row.index:
            sample_parts.append(f"{column.removeprefix('Metadata_').lower()}={row[column]}")
    sample_parts.append(f"center_x={bbox_center_x:.6f}")
    sample_parts.append(f"center_y={bbox_center_y:.6f}")
    return "|".join(sample_parts)


def _build_download_dataframe(dataframe: pd.DataFrame, work_dir: Path) -> pd.DataFrame:
    """Attach per-row output directories used by the download helpers.

    Args:
        dataframe: Sampled single-cell dataframe.
        work_dir: Scratch directory for staged downloads.

    Returns:
        Copy of the dataframe with an ``OutputDir`` column pointing to the
        unique per-row download directory.
    """

    download_df = dataframe.copy()
    download_df["OutputDir"] = download_df.index.map(lambda row_idx: str(work_dir / "downloads" / f"row_{row_idx:06d}"))
    return download_df


def _build_jobs_from_dataframe(
    *,
    dataframe: pd.DataFrame,
    columns: list[str],
    output_dir_column: str,
) -> list[DownloadJob]:
    """Build validated download jobs from explicit metadata rows.

    Args:
        dataframe: Dataframe containing S3 URLs and output directories.
        columns: URL columns to convert into download jobs.
        output_dir_column: Column containing per-row output directories.

    Returns:
        Ordered list of validated download jobs.

    Raises:
        ValueError: If a URL is missing or invalid, an output directory is
            blank, or two different URLs map to the same local path.
    """

    jobs: list[DownloadJob] = []
    planned_paths: dict[Path, tuple[str, object, str]] = {}

    for row_index, row in dataframe.iterrows():
        raw_output_dir = row[output_dir_column]
        if pd.isna(raw_output_dir) or str(raw_output_dir).strip() == "":
            raise ValueError(
                f"Invalid output directory value at row index {row_index} "
                f"for column {output_dir_column}: {raw_output_dir!r}"
            )
        output_dir = Path(str(raw_output_dir).strip())

        for column in columns:
            raw_s3_url = row[column]
            if pd.isna(raw_s3_url):
                raise ValueError(f"Missing required S3 URL at row index {row_index} for column {column}")

            s3_url = str(raw_s3_url).strip()
            if not s3_url.startswith("s3://"):
                raise ValueError(
                    f"Invalid S3 URL at row index {row_index} for column {column}: {raw_s3_url!r}"
                )

            filename = _filename_from_s3_url(s3_url)
            local_path = output_dir / filename
            existing = planned_paths.get(local_path)
            if existing is None:
                planned_paths[local_path] = (s3_url, row_index, column)
                jobs.append(DownloadJob(s3_url=s3_url, local_path=local_path))
                continue

            existing_s3_url, existing_row_index, existing_column = existing
            if existing_s3_url == s3_url:
                continue

            raise ValueError(
                f"Conflicting S3 URLs for local path {local_path}: row index {existing_row_index} "
                f"column {existing_column} uses {existing_s3_url!r}, but row index {row_index} "
                f"column {column} uses {s3_url!r}"
            )

    return jobs


def _materialize_corrected_crops(
    dataframe: pd.DataFrame,
    work_dir: Path,
    workers: int,
    overwrite: bool,
) -> tuple[list[Path], list[dict[str, object]]]:
    """Download source files, correct them, and save temporary crop arrays.

    Args:
        dataframe: Sampled single-cell dataframe.
        work_dir: Scratch directory for downloads and temporary crops.
        workers: Number of workers used for S3 downloads.
        overwrite: Whether to redownload existing local files.

    Returns:
        Tuple containing the temporary crop ``.npy`` paths and the metadata rows
        later stored in zarr attributes.

    Raises:
        RuntimeError: If any strict-mode download fails.
        ValueError: If a crop cannot be produced at the requested size.
    """

    download_df = _build_download_dataframe(dataframe=dataframe, work_dir=work_dir)
    image_jobs = _build_jobs_from_dataframe(
        dataframe=download_df,
        columns=["URL_OrigDNA"],
        output_dir_column="OutputDir",
    )
    illum_jobs = _build_jobs_from_dataframe(
        dataframe=download_df,
        columns=["URL_IllumDNA"],
        output_dir_column="OutputDir",
    )

    image_summary = run_download_jobs(
        jobs=image_jobs,
        overwrite=overwrite,
        workers=workers,
        parallel=True,
        verbose=True,
    )
    illum_summary = run_download_jobs(
        jobs=illum_jobs,
        overwrite=overwrite,
        workers=workers,
        parallel=True,
        verbose=True,
    )

    if image_summary.failed or illum_summary.failed:
        raise RuntimeError(
            "Strict mode aborting because one or more downloads failed. "
            f"image failures={image_summary.failed}, illumination failures={illum_summary.failed}"
        )

    temp_crop_dir = work_dir / "corrected_crops"
    temp_crop_dir.mkdir(parents=True, exist_ok=True)
    corrected_crop_paths: list[Path] = []
    zarr_metadata_rows: list[dict[str, object]] = []

    for row_index, row in download_df.iterrows():
        output_dir = Path(row["OutputDir"])
        orig_path = output_dir / _filename_from_s3_url(str(row["URL_OrigDNA"]))
        illum_path = output_dir / _filename_from_s3_url(str(row["URL_IllumDNA"]))

        corrected_image = _apply_illumination_correction(orig_path=orig_path, illum_path=illum_path)

        x0 = int(row["AreaShape_BoundingBoxMinimum_X"])
        x1 = int(row["AreaShape_BoundingBoxMaximum_X"])
        y0 = int(row["AreaShape_BoundingBoxMinimum_Y"])
        y1 = int(row["AreaShape_BoundingBoxMaximum_Y"])

        crop_x0, crop_x1 = _compute_shifted_window(
            start=x0,
            end=x1,
            target_size=CROP_SIZE,
            axis_limit=corrected_image.shape[1],
        )
        crop_y0, crop_y1 = _compute_shifted_window(
            start=y0,
            end=y1,
            target_size=CROP_SIZE,
            axis_limit=corrected_image.shape[0],
        )

        crop = corrected_image[crop_y0:crop_y1, crop_x0:crop_x1].astype(np.float32, copy=False)
        if crop.shape != (CROP_SIZE, CROP_SIZE):
            raise ValueError(
                f"Expected crop shape {(CROP_SIZE, CROP_SIZE)} for row {row_index}, got {crop.shape}"
            )

        bbox_center_x = (x0 + x1) / 2.0
        bbox_center_y = (y0 + y1) / 2.0
        sample_id = _sample_id(row=row, bbox_center_x=bbox_center_x, bbox_center_y=bbox_center_y)

        temp_crop_path = temp_crop_dir / f"crop_{row_index:06d}.npy"
        np.save(temp_crop_path, crop)
        corrected_crop_paths.append(temp_crop_path)

        zarr_metadata_rows.append({"sample_id": sample_id})

        shutil.rmtree(output_dir)

    return corrected_crop_paths, zarr_metadata_rows


def fit_robust_percentile_normalization(corrected_crop_paths: list[Path]) -> NormalizationConfig:
    """Estimate robust percentile normalization bounds across all crops.

    Args:
        corrected_crop_paths: Temporary crop files used to estimate dataset-wide
            clipping bounds.

    Returns:
        Normalization configuration capturing the robust percentile bounds.

    Raises:
        ValueError: If no crops exist, no finite values are present, or the
            estimated bounds are not strictly increasing.
    """

    if not corrected_crop_paths:
        raise ValueError("No corrected crops were generated.")

    finite_min = np.inf
    finite_max = -np.inf
    finite_count = 0

    for path in corrected_crop_paths:
        crop = np.load(path)
        finite_mask = np.isfinite(crop)
        finite_values = crop[finite_mask]
        if finite_values.size == 0:
            continue
        finite_min = min(finite_min, float(finite_values.min()))
        finite_max = max(finite_max, float(finite_values.max()))
        finite_count += int(finite_values.size)

    if finite_count == 0:
        raise ValueError("Corrected crops do not contain any finite values.")

    if finite_min == finite_max:
        raise ValueError(
            "Robust percentile normalization bounds must be increasing, got constant value "
            f"{finite_min}"
        )

    bin_edges = np.linspace(
        finite_min,
        finite_max,
        NORMALIZATION_HISTOGRAM_BINS + 1,
        dtype=np.float64,
    )
    histogram = np.zeros(NORMALIZATION_HISTOGRAM_BINS, dtype=np.int64)

    for path in corrected_crop_paths:
        crop = np.load(path)
        finite_values = crop[np.isfinite(crop)]
        if finite_values.size == 0:
            continue
        histogram += np.histogram(finite_values, bins=bin_edges)[0]

    cumulative_counts = np.cumsum(histogram)
    if cumulative_counts[-1] == 0:
        raise ValueError("Corrected crops do not contain any finite values.")

    def percentile_from_histogram(percentile: float) -> float:
        rank = percentile / 100.0 * (cumulative_counts[-1] - 1)
        target_count = int(np.floor(rank)) + 1
        bin_index = int(np.searchsorted(cumulative_counts, target_count, side="left"))
        previous_count = 0 if bin_index == 0 else int(cumulative_counts[bin_index - 1])
        bin_count = int(histogram[bin_index])
        left_edge = float(bin_edges[bin_index])
        right_edge = float(bin_edges[bin_index + 1])
        if bin_count <= 0 or right_edge <= left_edge:
            return left_edge

        within_bin_rank = rank - previous_count
        fraction = min(max(within_bin_rank / bin_count, 0.0), 1.0)
        return left_edge + fraction * (right_edge - left_edge)

    lower_value = percentile_from_histogram(LOWER_PERCENTILE)
    upper_value = percentile_from_histogram(UPPER_PERCENTILE)
    if lower_value >= upper_value:
        raise ValueError(
            "Robust percentile normalization bounds must be increasing, got "
            f"lower_value={lower_value}, upper_value={upper_value}"
        )

    return NormalizationConfig(
        method="robust_percentile_minmax",
        lower_percentile=LOWER_PERCENTILE,
        upper_percentile=UPPER_PERCENTILE,
        lower_value=lower_value,
        upper_value=upper_value,
    )


def apply_robust_percentile_normalization(
    crop: np.ndarray,
    normalization_config: NormalizationConfig,
) -> np.ndarray:
    """Normalize a crop into the ``[0, 1]`` range using stored bounds.

    Args:
        crop: Illumination-corrected crop array.
        normalization_config: Dataset-level clipping bounds and method metadata.

    Returns:
        Normalized crop clipped into the closed interval ``[0, 1]``.
    """

    normalized_crop = np.clip(
        crop,
        normalization_config.lower_value,
        normalization_config.upper_value,
    )
    normalized_crop = (normalized_crop - normalization_config.lower_value) / (
        normalization_config.upper_value - normalization_config.lower_value
    )
    return np.clip(normalized_crop, 0.0, 1.0)


def _write_normalized_zarr(
    corrected_crop_paths: list[Path],
    output_zarr: Path,
    input_parquet: Path,
    normalization_config: NormalizationConfig,
    metadata_rows: list[dict[str, object]],
) -> None:
    """Write normalized crops and alignment metadata to a zarr store.

    Args:
        corrected_crop_paths: Temporary crop files to normalize and write.
        output_zarr: Destination zarr store path.
        input_parquet: Sampled parquet used to create the crops.
        normalization_config: Stored normalization parameters.
        metadata_rows: Per-row metadata used to populate zarr attributes.

    Returns:
        None.
    """

    if output_zarr.exists():
        if output_zarr.is_dir():
            shutil.rmtree(output_zarr)
        else:
            output_zarr.unlink()

    root = zarr.open_group(str(output_zarr), mode="w", zarr_format=2)
    crops = root.create_array(
        name="crops",
        shape=(len(corrected_crop_paths), CROP_SIZE, CROP_SIZE),
        chunks=(min(256, len(corrected_crop_paths)), CROP_SIZE, CROP_SIZE),
        dtype="float32",
        compressor=Blosc(cname="zstd", clevel=7, shuffle=Blosc.BITSHUFFLE),
    )

    for crop_index, crop_path in enumerate(corrected_crop_paths):
        crop = np.load(crop_path).astype(np.float32, copy=False)
        crop = apply_robust_percentile_normalization(
            crop=crop,
            normalization_config=normalization_config,
        )
        crops[crop_index] = crop
        crop_path.unlink()

    normalization_attrs = asdict(normalization_config)
    root.attrs.update(
        {
            "source_parquet_path": str(input_parquet.resolve()),
            "crop_size": CROP_SIZE,
            "normalization_method": normalization_config.method,
            **{f"normalization_{key}": value for key, value in normalization_attrs.items() if key != "method"},
            "crop_center_method": "bbox_midpoint",
            "crop_window_method": "shifted_in_bounds",
            "illumination_correction": "orig_div_illum",
            "normalized_dtype": "float32",
            "strict_row_order": True,
            "row_count": len(corrected_crop_paths),
            "sample_ids": [row["sample_id"] for row in metadata_rows],
        }
    )


def main() -> None:
    """Run the crop-download, correction, normalization, and zarr-write pipeline.

    Returns:
        None.

    This entrypoint converts the sampled single-cell parquet into a strict-order
    zarr store whose rows stay aligned with the sampled parquet through stored
    sample identifiers and row-count metadata.
    """

    args = parse_args()
    if args.output_zarr.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output zarr store already exists: {args.output_zarr}. Pass --overwrite to replace it."
        )

    dataframe = pd.read_parquet(args.input_parquet).reset_index(drop=True)
    validate_input_columns(dataframe)

    args.work_dir.mkdir(parents=True, exist_ok=True)

    corrected_crop_paths, metadata_rows = _materialize_corrected_crops(
        dataframe=dataframe,
        work_dir=args.work_dir,
        workers=args.workers,
        overwrite=args.overwrite,
    )
    normalization_config = fit_robust_percentile_normalization(
        corrected_crop_paths=corrected_crop_paths,
    )
    _write_normalized_zarr(
        corrected_crop_paths=corrected_crop_paths,
        output_zarr=args.output_zarr,
        input_parquet=args.input_parquet,
        normalization_config=normalization_config,
        metadata_rows=metadata_rows,
    )

    corrected_crop_dir = args.work_dir / "corrected_crops"
    if corrected_crop_dir.exists():
        shutil.rmtree(corrected_crop_dir)

    print(
        f"Saved {len(corrected_crop_paths)} normalized crops to {args.output_zarr} "
        "with normalization "
        f"{normalization_config.method} using bounds "
        f"[{normalization_config.lower_value}, {normalization_config.upper_value}]"
    )


if __name__ == "__main__":
    main()
