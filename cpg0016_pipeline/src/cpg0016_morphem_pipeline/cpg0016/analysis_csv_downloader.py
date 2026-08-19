"""Discover and download CPG0016 single-cell analysis CSV files.

This module indexes the public ``workspace/analysis`` tree for
``cellpainting-gallery/cpg0016-jump`` and groups discovered CSVs by their
containing analysis folder. Each grouped record exposes the common CellProfiler
profile files such as ``Image.csv``, ``Nuclei.csv``, ``Cells.csv``, and
``Cytoplasm.csv`` while preserving the S3-relative local download layout.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional, Sequence
from urllib.parse import urlparse

import pandas as pd
import s3fs

from cpg0016_morphem_pipeline.cpg0016.load_data_with_illum_downloader import (
    CPG0016_BUCKET,
    CPG0016_PREFIX,
    DownloadSummary,
    is_source_all_path,
    remote_path_to_s3_url,
    s3_url_to_relative_local_path,
)


ANALYSIS_CSV_GLOB_PATTERN = f"{CPG0016_BUCKET}/{CPG0016_PREFIX}/*/workspace/analysis/**/*.csv"
ANALYSIS_PROFILE_FILENAMES = (
    "Image.csv",
    "Nuclei.csv",
    "Cells.csv",
    "Cytoplasm.csv",
)
AWS_DEFAULT_MAX_CONCURRENT_REQUESTS = 50
ANALYSIS_PROFILE_NAME_ALIASES = {
    "image": "Image.csv",
    "image.csv": "Image.csv",
    "nuclei": "Nuclei.csv",
    "nuclei.csv": "Nuclei.csv",
    "cells": "Cells.csv",
    "cells.csv": "Cells.csv",
    "cytoplasm": "Cytoplasm.csv",
    "cytoplasm.csv": "Cytoplasm.csv",
}
SOURCE_NAME_PATTERN = re.compile(r"^source_[^/]+$")


@dataclass(frozen=True)
class AnalysisCSVSet:
    """Grouped analysis CSV paths for one analysis folder.

    Attributes:
        folder_relative_path: Dataset-relative folder containing one analysis
            CSV group.
        folder_local_path: Local folder under the configured output root.
        folder_s3_url: Trailing-slash S3 URL for the folder when known.
        image_s3_url: Remote ``Image.csv`` URL when known.
        nuclei_s3_url: Remote ``Nuclei.csv`` URL when known.
        cells_s3_url: Remote ``Cells.csv`` URL when known.
        cytoplasm_s3_url: Remote ``Cytoplasm.csv`` URL when known.
        image_local_path: Expected or discovered local ``Image.csv`` path.
        nuclei_local_path: Expected or discovered local ``Nuclei.csv`` path.
        cells_local_path: Expected or discovered local ``Cells.csv`` path.
        cytoplasm_local_path: Expected or discovered local ``Cytoplasm.csv`` path.
        other_s3_urls: Additional remote CSVs keyed by filename.
        other_local_paths: Additional local CSVs keyed by filename.
    """

    folder_relative_path: Path
    folder_local_path: Path
    folder_s3_url: Optional[str] = None
    image_s3_url: Optional[str] = None
    nuclei_s3_url: Optional[str] = None
    cells_s3_url: Optional[str] = None
    cytoplasm_s3_url: Optional[str] = None
    image_local_path: Optional[Path] = None
    nuclei_local_path: Optional[Path] = None
    cells_local_path: Optional[Path] = None
    cytoplasm_local_path: Optional[Path] = None
    other_s3_urls: dict[str, str] = field(default_factory=dict)
    other_local_paths: dict[str, Path] = field(default_factory=dict)

    def read_csv(self, filename: str) -> pd.DataFrame:
        """Read one analysis CSV and set path-derived metadata when available.

        Args:
            filename: Requested CSV filename within this grouped analysis set.

        Returns:
            Loaded dataframe with ``Metadata_Source`` and ``Metadata_Batch``
            overwritten from the dataset path when those values can be inferred.
        """

        local_path = self._get_local_path(filename)
        source_path = self._get_source_path(filename)
        dataframe = pd.read_csv(local_path)
        try:
            metadata_values = extract_metadata_from_dataset_path(source_path)
        except ValueError:
            return dataframe

        dataframe["Metadata_Source"] = metadata_values["Metadata_Source"]
        dataframe["Metadata_Batch"] = metadata_values["Metadata_Batch"]
        return dataframe

    def _get_local_path(self, filename: str) -> Path:
        """Return the local path for one CSV filename or raise if unavailable."""

        local_path = self._get_local_path_or_none(filename)
        if local_path is None:
            raise ValueError(f"CSV not available in this analysis set: {filename}")
        return local_path

    def _get_source_path(self, filename: str) -> str:
        """Return the best provenance path for one CSV, preferring the S3 URL."""

        s3_url = self._get_s3_url_or_none(filename)
        if s3_url is not None:
            return s3_url
        return str(self._get_local_path(filename))

    def _get_local_path_or_none(self, filename: str) -> Optional[Path]:
        """Return the local path for one CSV filename when present in this set."""

        if filename == "Image.csv":
            return self.image_local_path
        if filename == "Nuclei.csv":
            return self.nuclei_local_path
        if filename == "Cells.csv":
            return self.cells_local_path
        if filename == "Cytoplasm.csv":
            return self.cytoplasm_local_path
        return self.other_local_paths.get(filename)

    def _get_s3_url_or_none(self, filename: str) -> Optional[str]:
        """Return the S3 URL for one CSV filename when present in this set."""

        if filename == "Image.csv":
            return self.image_s3_url
        if filename == "Nuclei.csv":
            return self.nuclei_s3_url
        if filename == "Cells.csv":
            return self.cells_s3_url
        if filename == "Cytoplasm.csv":
            return self.cytoplasm_s3_url
        return self.other_s3_urls.get(filename)


def _split_dataset_path_parts(path_str: str | Path) -> list[str]:
    """Return normalized path segments for an S3 URL or mirrored local path.

    Args:
        path_str: S3 URL or mirrored local path.

    Returns:
        Normalized path segments.

    Raises:
        ValueError: If an S3 URL does not belong to the configured CPG0016
            bucket.
    """

    raw_path = str(path_str).strip()
    parsed = urlparse(raw_path)
    if parsed.scheme == "s3":
        if parsed.netloc != CPG0016_BUCKET:
            raise ValueError(f"Invalid CPG0016 dataset path: {path_str}")
        return [part for part in parsed.path.split("/") if part]
    return [part for part in raw_path.replace("\\", "/").split("/") if part]


def extract_metadata_from_dataset_path(path_str: str | Path) -> dict[str, str]:
    """Extract CPG0016 source and batch segments from a dataset path.

    Args:
        path_str: S3 URL or mirrored local dataset path.

    Returns:
        Mapping containing ``Metadata_Source`` and ``Metadata_Batch``.

    Raises:
        ValueError: If the path does not match the expected CPG0016 analysis
            directory structure.
    """

    path_parts = _split_dataset_path_parts(path_str)

    try:
        prefix_index = path_parts.index(CPG0016_PREFIX)
        source = path_parts[prefix_index + 1]
        if path_parts[prefix_index + 2 : prefix_index + 4] != ["workspace", "analysis"]:
            raise ValueError
        batch = path_parts[prefix_index + 4]
    except (ValueError, IndexError) as exc:
        raise ValueError(f"Invalid CPG0016 dataset path: {path_str}") from exc

    return {
        "Metadata_Source": source,
        "Metadata_Batch": batch,
    }


def extract_metadata_source_from_dataset_path(path_str: str | Path) -> str:
    """Extract the CPG0016 source segment from an S3 URL or mirrored local path."""

    return extract_metadata_from_dataset_path(path_str)["Metadata_Source"]


def extract_metadata_batch_from_dataset_path(path_str: str | Path) -> str:
    """Extract the CPG0016 batch segment from an S3 URL or mirrored local path."""

    return extract_metadata_from_dataset_path(path_str)["Metadata_Batch"]


def _copy_analysis_csv_set(csv_set: AnalysisCSVSet) -> AnalysisCSVSet:
    """Return a defensive copy of one grouped CSV-set record.

    Args:
        csv_set: Grouped analysis CSV record.

    Returns:
        New ``AnalysisCSVSet`` instance with copied mappings.
    """

    return AnalysisCSVSet(
        folder_relative_path=csv_set.folder_relative_path,
        folder_local_path=csv_set.folder_local_path,
        folder_s3_url=csv_set.folder_s3_url,
        image_s3_url=csv_set.image_s3_url,
        nuclei_s3_url=csv_set.nuclei_s3_url,
        cells_s3_url=csv_set.cells_s3_url,
        cytoplasm_s3_url=csv_set.cytoplasm_s3_url,
        image_local_path=csv_set.image_local_path,
        nuclei_local_path=csv_set.nuclei_local_path,
        cells_local_path=csv_set.cells_local_path,
        cytoplasm_local_path=csv_set.cytoplasm_local_path,
        other_s3_urls=dict(csv_set.other_s3_urls),
        other_local_paths=dict(csv_set.other_local_paths),
    )


def _folder_remote_path_to_s3_url(folder_remote_path: str) -> str:
    """Convert a remote folder path into a normalized trailing-slash S3 URL.

    Args:
        folder_remote_path: Remote folder path in ``bucket/key`` form.

    Returns:
        Normalized trailing-slash S3 URL.
    """

    cleaned_folder_remote_path = str(folder_remote_path).rstrip("/")
    return remote_path_to_s3_url(cleaned_folder_remote_path) + "/"


def _build_analysis_csv_set(
    *,
    folder_relative_path: Path,
    output_dir: Path,
    remote_urls_by_name: Optional[dict[str, str]] = None,
    local_paths_by_name: Optional[dict[str, Path]] = None,
) -> AnalysisCSVSet:
    """Build one ``AnalysisCSVSet`` from grouped remote and/or local files.

    Args:
        folder_relative_path: Dataset-relative folder for the group.
        output_dir: Local output root.
        remote_urls_by_name: Optional remote URLs keyed by filename.
        local_paths_by_name: Optional local paths keyed by filename.

    Returns:
        Fully populated grouped analysis CSV record.
    """

    remote_urls_by_name = remote_urls_by_name or {}
    local_paths_by_name = local_paths_by_name or {}

    folder_local_path = output_dir / folder_relative_path
    folder_s3_url: Optional[str] = None
    if remote_urls_by_name:
        folder_s3_url = _folder_remote_path_to_s3_url(
            f"{CPG0016_BUCKET}/{folder_relative_path.as_posix()}"
        )

    known_remote_urls = {
        filename: remote_urls_by_name[filename]
        for filename in ANALYSIS_PROFILE_FILENAMES
        if filename in remote_urls_by_name
    }
    known_local_paths = {
        filename: local_paths_by_name.get(filename, folder_local_path / filename)
        for filename in ANALYSIS_PROFILE_FILENAMES
        if filename in remote_urls_by_name or filename in local_paths_by_name
    }

    return AnalysisCSVSet(
        folder_relative_path=folder_relative_path,
        folder_local_path=folder_local_path,
        folder_s3_url=folder_s3_url,
        image_s3_url=known_remote_urls.get("Image.csv"),
        nuclei_s3_url=known_remote_urls.get("Nuclei.csv"),
        cells_s3_url=known_remote_urls.get("Cells.csv"),
        cytoplasm_s3_url=known_remote_urls.get("Cytoplasm.csv"),
        image_local_path=known_local_paths.get("Image.csv"),
        nuclei_local_path=known_local_paths.get("Nuclei.csv"),
        cells_local_path=known_local_paths.get("Cells.csv"),
        cytoplasm_local_path=known_local_paths.get("Cytoplasm.csv"),
        other_s3_urls={
            filename: url
            for filename, url in remote_urls_by_name.items()
            if filename not in ANALYSIS_PROFILE_FILENAMES
        },
        other_local_paths={
            filename: path
            for filename, path in local_paths_by_name.items()
            if filename not in ANALYSIS_PROFILE_FILENAMES
        },
    )


def normalize_analysis_csv_filenames(csv_names: Sequence[str] | None) -> tuple[str, ...]:
    """Normalize requested CSV names to canonical analysis profile filenames.

    Args:
        csv_names: Optional requested filenames or shorthand names.

    Returns:
        Canonical ordered tuple of requested analysis CSV filenames.

    Raises:
        ValueError: If any requested name is not recognized.
    """

    if csv_names is None:
        return ANALYSIS_PROFILE_FILENAMES

    normalized_names: list[str] = []
    invalid_names: list[str] = []
    for csv_name in csv_names:
        normalized_name = ANALYSIS_PROFILE_NAME_ALIASES.get(str(csv_name).strip().lower())
        if normalized_name is None:
            invalid_names.append(str(csv_name))
            continue
        if normalized_name not in normalized_names:
            normalized_names.append(normalized_name)

    if invalid_names:
        valid_names = ", ".join(ANALYSIS_PROFILE_FILENAMES)
        invalid_display = ", ".join(repr(name) for name in invalid_names)
        raise ValueError(
            f"Invalid analysis CSV names: {invalid_display}. Valid options are: {valid_names}"
        )

    return tuple(normalized_names)


def normalize_analysis_sources(sources: Sequence[str] | None) -> tuple[str, ...] | None:
    """Normalize requested source names to canonical dataset source segments.

    Args:
        sources: Optional iterable of requested source names.

    Returns:
        Canonical tuple of requested source names, or ``None`` when all sources
        are allowed.

    Raises:
        ValueError: If any source does not match the expected dataset segment
            format.
    """

    if sources is None:
        return None

    normalized_sources: list[str] = []
    invalid_sources: list[str] = []
    for source in sources:
        normalized_source = str(source).strip()
        if not SOURCE_NAME_PATTERN.fullmatch(normalized_source):
            invalid_sources.append(str(source))
            continue
        if normalized_source not in normalized_sources:
            normalized_sources.append(normalized_source)

    if invalid_sources:
        invalid_display = ", ".join(repr(source) for source in invalid_sources)
        raise ValueError(
            "Invalid analysis sources: "
            f"{invalid_display}. Sources must match the dataset path segment format, for example 'source_10'."
        )

    return tuple(normalized_sources)


def build_analysis_csv_sets_from_s3_urls(
    s3_urls: list[str],
    *,
    output_dir: Path,
) -> list[AnalysisCSVSet]:
    """Group discovered analysis CSV URLs by their containing folder.

    Args:
        s3_urls: Discovered analysis CSV URLs.
        output_dir: Local output root for mirrored paths.

    Returns:
        Grouped analysis CSV records keyed by containing folder.
    """

    grouped_urls: dict[Path, dict[str, str]] = {}
    for s3_url in s3_urls:
        relative_path = s3_url_to_relative_local_path(s3_url)
        folder_relative_path = relative_path.parent
        grouped_urls.setdefault(folder_relative_path, {})[relative_path.name] = s3_url

    csv_sets = [
        _build_analysis_csv_set(
            folder_relative_path=folder_relative_path,
            output_dir=output_dir,
            remote_urls_by_name=urls_by_name,
        )
        for folder_relative_path, urls_by_name in sorted(grouped_urls.items())
    ]
    return csv_sets


def build_analysis_csv_sets_from_local_paths(
    local_paths: list[Path],
    *,
    output_dir: Path,
) -> list[AnalysisCSVSet]:
    """Group local analysis CSV files by their containing folder.

    Args:
        local_paths: Local analysis CSV files.
        output_dir: Local output root.

    Returns:
        Grouped analysis CSV records keyed by containing folder.
    """

    grouped_paths: dict[Path, dict[str, Path]] = {}
    for local_path in local_paths:
        folder_relative_path = local_path.relative_to(output_dir).parent
        grouped_paths.setdefault(folder_relative_path, {})[local_path.name] = local_path

    csv_sets = [
        _build_analysis_csv_set(
            folder_relative_path=folder_relative_path,
            output_dir=output_dir,
            local_paths_by_name=paths_by_name,
        )
        for folder_relative_path, paths_by_name in sorted(grouped_paths.items())
    ]
    return csv_sets


def _build_analysis_csv_include_patterns(
    csv_filenames: Sequence[str],
    sources: Sequence[str] | None = None,
) -> list[str]:
    """Build AWS CLI include patterns for the requested analysis CSV filenames.

    Args:
        csv_filenames: Canonical filenames to include.
        sources: Optional subset of source names.

    Returns:
        AWS CLI include patterns covering the requested filenames and sources.
    """

    source_patterns = tuple(sources) if sources is not None else ("source_*",)
    return [
        f"{source_pattern}/workspace/analysis/**/{filename}"
        for source_pattern in source_patterns
        for filename in csv_filenames
    ]


def _create_aws_cli_config(max_concurrent_requests: int) -> str:
    """Create a temporary AWS CLI config file that sets S3 concurrency.

    Args:
        max_concurrent_requests: Desired AWS CLI S3 concurrency.

    Returns:
        Path to the generated temporary AWS config file.
    """

    temp_dir = tempfile.mkdtemp(prefix="jump-analysis-aws-")
    config_path = Path(temp_dir) / "config"
    config_path.write_text(
        "\n".join(
            [
                "[default]",
                "s3 =",
                f"    max_concurrent_requests = {max_concurrent_requests}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return str(config_path)


class CPG0016AnalysisCSVDownloader:
    """Discover, iterate, and download CPG0016 analysis CSV folders.

    The downloader can operate entirely from a previously downloaded local tree
    or invoke the AWS CLI to mirror a selected subset of the public
    ``workspace/analysis`` hierarchy.
    """

    def __init__(
        self,
        output_dir: Path | str,
        *,
        workers: int = 4,
        parallel: bool = True,
        verbose: bool = True,
        use_existing_csvs_without_s3_check: bool = False,
        max_concurrent_requests: int = AWS_DEFAULT_MAX_CONCURRENT_REQUESTS,
    ) -> None:
        """Initialize the analysis CSV downloader in S3 or local-only mode.

        Args:
            output_dir: Local root directory for mirrored analysis CSVs.
            workers: Reserved worker count field kept for API consistency.
            parallel: Whether AWS CLI downloads should use high concurrency.
            verbose: Whether to surface AWS CLI output and shared downloader logs.
            use_existing_csvs_without_s3_check: Whether to skip S3 and inspect
                the local output tree only.
            max_concurrent_requests: AWS CLI S3 concurrency limit.

        Returns:
            None.

        Raises:
            ValueError: If ``output_dir`` is missing or the concurrency is
                invalid.
        """

        if output_dir is None:
            raise ValueError("output_dir must be provided")
        if max_concurrent_requests < 1:
            raise ValueError("max_concurrent_requests must be >= 1")

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.workers = workers
        self.parallel = parallel
        self.verbose = verbose
        self.use_existing_csvs_without_s3_check = use_existing_csvs_without_s3_check
        self.max_concurrent_requests = max_concurrent_requests

        if self.use_existing_csvs_without_s3_check:
            self.analysis_csv_urls: list[str] = []
        else:
            self.analysis_csv_urls = []

        self.analysis_csv_sets = self.discover_local_analysis_csv_sets()

    def discover_analysis_csv_urls(self) -> list[str]:
        """Discover public CPG0016 analysis CSV URLs while excluding ``source_all``.

        Returns:
            Sorted list of analysis CSV S3 URLs.
        """

        fs = s3fs.S3FileSystem(anon=True)
        remote_paths = sorted(fs.glob(ANALYSIS_CSV_GLOB_PATTERN))
        analysis_csv_urls: list[str] = []
        for remote_path in remote_paths:
            if is_source_all_path(remote_path):
                continue
            analysis_csv_urls.append(remote_path_to_s3_url(remote_path))
        return analysis_csv_urls

    def discover_local_analysis_csv_sets(self) -> list[AnalysisCSVSet]:
        """Discover previously downloaded local analysis CSV groups.

        Returns:
            Grouped analysis CSV records reconstructed from the local mirror.
        """

        local_paths = self._discover_local_analysis_csv_paths()
        return build_analysis_csv_sets_from_local_paths(local_paths, output_dir=self.output_dir)

    def get_analysis_csv_sets(self) -> list[AnalysisCSVSet]:
        """Return copies of the grouped analysis CSV records.

        Returns:
            Defensive copies of the known grouped analysis CSV records.
        """

        return [_copy_analysis_csv_set(csv_set) for csv_set in self.analysis_csv_sets]

    def iter_analysis_csv_sets(self) -> Iterator[AnalysisCSVSet]:
        """Yield grouped analysis CSV records one folder at a time.

        Yields:
            Defensive copies of grouped analysis CSV records.
        """

        for csv_set in self.analysis_csv_sets:
            yield _copy_analysis_csv_set(csv_set)

    def download_all_csv_profiles(
        self,
        csv_names: Sequence[str] | None = None,
        sources: Sequence[str] | None = None,
    ) -> DownloadSummary:
        """Download analysis CSV files into ``output_dir`` with the AWS CLI.

        Args:
            csv_names: Optional subset of filenames or shorthand names.
            sources: Optional subset of source names.

        Returns:
            Download summary derived from the mirrored local output tree.
        """

        if self.use_existing_csvs_without_s3_check:
            return DownloadSummary(total_jobs=0, downloaded=0, skipped=0, failed=0, failures=[])

        requested_filenames = tuple(normalize_analysis_csv_filenames(csv_names))
        requested_sources = normalize_analysis_sources(sources)
        requested_filename_set = set(requested_filenames)
        requested_source_set = set(requested_sources) if requested_sources is not None else None
        preexisting_paths = self._discover_local_analysis_csv_paths(
            requested_filenames=requested_filename_set,
            requested_sources=requested_source_set,
        )
        self._run_aws_analysis_csv_download(requested_filenames, requested_sources)
        post_download_paths = self._discover_local_analysis_csv_paths(
            requested_filenames=requested_filename_set,
            requested_sources=requested_source_set,
        )

        self.analysis_csv_urls = [self._local_analysis_csv_path_to_s3_url(path) for path in post_download_paths]
        self.analysis_csv_sets = build_analysis_csv_sets_from_s3_urls(
            self.analysis_csv_urls,
            output_dir=self.output_dir,
        )

        preexisting_path_set = set(preexisting_paths)
        downloaded = sum(1 for path in post_download_paths if path not in preexisting_path_set)
        skipped = len(post_download_paths) - downloaded
        return DownloadSummary(
            total_jobs=len(post_download_paths),
            downloaded=downloaded,
            skipped=skipped,
            failed=0,
            failures=[],
        )

    def _discover_local_analysis_csv_paths(
        self,
        requested_filenames: Optional[set[str]] = None,
        requested_sources: Optional[set[str]] = None,
    ) -> list[Path]:
        """Discover local analysis CSV files under ``output_dir``.

        Args:
            requested_filenames: Optional filename filter.
            requested_sources: Optional source filter.

        Returns:
            Sorted list of matching local analysis CSV paths.
        """

        analysis_root = self.output_dir / CPG0016_PREFIX
        if not analysis_root.exists():
            return []

        local_paths = [
            local_path
            for local_path in sorted(analysis_root.rglob("workspace/analysis/**/*.csv"))
            if not is_source_all_path(local_path.relative_to(self.output_dir).as_posix())
            and (requested_filenames is None or local_path.name in requested_filenames)
            and (
                requested_sources is None
                or extract_metadata_source_from_dataset_path(local_path.relative_to(self.output_dir))
                in requested_sources
            )
        ]
        return local_paths

    def _local_analysis_csv_path_to_s3_url(self, local_path: Path) -> str:
        """Convert a local analysis CSV path under ``output_dir`` back into its S3 URL.

        Args:
            local_path: Mirrored local analysis CSV path.

        Returns:
            Synthetic S3 URL corresponding to ``local_path``.
        """

        relative_path = local_path.relative_to(self.output_dir).as_posix().lstrip("/")
        return f"s3://{CPG0016_BUCKET}/{relative_path}"

    def _run_aws_analysis_csv_download(
        self,
        requested_filenames: Sequence[str],
        requested_sources: Sequence[str] | None,
    ) -> None:
        """Download the requested analysis CSVs with the AWS CLI transfer manager.

        Args:
            requested_filenames: Canonical analysis filenames to download.
            requested_sources: Optional subset of source names.

        Returns:
            None.

        Raises:
            RuntimeError: If the AWS CLI is unavailable or the transfer fails.
        """

        aws_path = shutil.which("aws")
        if aws_path is None:
            raise RuntimeError(
                "AWS CLI is required to download CPG0016 analysis CSVs. Install `aws` and retry."
            )

        destination_root = self.output_dir / CPG0016_PREFIX
        destination_root.mkdir(parents=True, exist_ok=True)

        command = [
            aws_path,
            "s3",
            "cp",
            f"s3://{CPG0016_BUCKET}/{CPG0016_PREFIX}/",
            str(destination_root),
            "--recursive",
            "--exclude",
            "*",
            "--exclude",
            "source_all/*",
        ]
        for include_pattern in _build_analysis_csv_include_patterns(
            requested_filenames,
            requested_sources,
        ):
            command.extend(["--include", include_pattern])
        command.extend(["--no-sign-request", "--only-show-errors"])

        effective_max_concurrent_requests = 1 if not self.parallel else self.max_concurrent_requests
        config_path = _create_aws_cli_config(effective_max_concurrent_requests)
        config_dir = Path(config_path).parent
        env = os.environ.copy()
        env["AWS_CONFIG_FILE"] = config_path
        env.setdefault("AWS_PAGER", "")

        try:
            result = subprocess.run(
                command,
                check=False,
                capture_output=not self.verbose,
                text=True,
                env=env,
            )
        finally:
            shutil.rmtree(config_dir, ignore_errors=True)

        if result.returncode != 0:
            stderr = (result.stderr or "").strip()
            stdout = (result.stdout or "").strip()
            details = stderr or stdout or f"aws exited with status {result.returncode}"
            raise RuntimeError(f"AWS CLI analysis CSV download failed: {details}")
