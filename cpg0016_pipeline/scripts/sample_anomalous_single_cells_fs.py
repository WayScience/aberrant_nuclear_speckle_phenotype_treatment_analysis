#!/usr/bin/env python
"""Sample single-cell profiles from merged CPG0016 parquets.

This script first collects only per-parquet well keys from discovered
``Nuclei_Image_merged.parquet`` files, merges those keys with perturbation
metadata, then performs per-parquet pre-sampling with a seeded RNG before a
final global quota pass and DNA URL merge.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cpg0016_morphem_pipeline.cpg0016.perturbation_metadata import (
    load_experiment_perturbation_metadata,
)

try:
    from cpg0016_morphem_pipeline.cpg0016 import CPG0016LoadDataWithIllumDownloader
except ModuleNotFoundError:
    CPG0016LoadDataWithIllumDownloader = None


DEFAULT_DOWNLOADED_PROFILES_DIR = Path("downloaded_profiles_csvs")
DEFAULT_LOAD_DATA_CSV_DIR = Path("downloaded_cpg0016_csvs")
DEFAULT_OUTPUT_PARQUET = Path("sampled_anomalous_single_cells.parquet")
MERGED_PROFILE_FILENAME = "Nuclei_Image_merged.parquet"
PROFILE_JOIN_COLUMNS = [
    "Metadata_Source",
    "Metadata_Batch",
    "Metadata_Plate",
    "Metadata_Well",
    "Metadata_Site",
]
PERTURBATION_METADATA_JOIN_COLUMNS = [
    "Metadata_Source",
    "Metadata_Batch",
    "Metadata_Plate",
    "Metadata_Well",
]
PROFILE_GROUP_COLUMNS = [
    "Metadata_Source",
    "Metadata_Batch",
    "Metadata_Plate",
    "Metadata_Well",
]
PARQUET_SAMPLE_COLUMNS = [
    "Metadata_Source",
    "Metadata_Batch",
    "Metadata_Plate",
    "Metadata_Well",
    "Metadata_Site",
    "AreaShape_Center_X",
    "AreaShape_Center_Y",
    "AreaShape_BoundingBoxMinimum_X",
    "AreaShape_BoundingBoxMaximum_X",
    "AreaShape_BoundingBoxMinimum_Y",
    "AreaShape_BoundingBoxMaximum_Y",
]
SAMPLED_OUTPUT_COLUMNS = [
    "Metadata_Source",
    "Metadata_Batch",
    "Metadata_Plate",
    "Metadata_Well",
    "Metadata_Site",
    "AreaShape_Center_X",
    "AreaShape_Center_Y",
    "AreaShape_BoundingBoxMinimum_X",
    "AreaShape_BoundingBoxMaximum_X",
    "AreaShape_BoundingBoxMinimum_Y",
    "AreaShape_BoundingBoxMaximum_Y",
    "Metadata_JCP2022",
    "Metadata_pert_type",
    "Metadata_Name",
]
GROUP_SAMPLING_COLUMNS = [
    "Metadata_Source",
    "Metadata_Batch",
    "Metadata_Plate",
    "Metadata_JCP2022",
]
LOAD_DATA_COLUMNS = [
    "URL_OrigDNA",
    "URL_IllumDNA",
    "Metadata_Source",
    "Metadata_Batch",
    "Metadata_Plate",
    "Metadata_Well",
    "Metadata_Site",
]
FINAL_OUTPUT_COLUMNS = [
    "AreaShape_Center_X",
    "AreaShape_Center_Y",
    "AreaShape_BoundingBoxMinimum_X",
    "AreaShape_BoundingBoxMaximum_X",
    "AreaShape_BoundingBoxMinimum_Y",
    "AreaShape_BoundingBoxMaximum_Y",
    "URL_OrigDNA",
    "URL_IllumDNA",
    "Metadata_JCP2022",
    "Metadata_pert_type",
    "Metadata_Name",
    "Metadata_Source",
    "Metadata_Batch",
    "Metadata_Plate",
    "Metadata_Well",
    "Metadata_Site",
]
POSITIVE_SAMPLES_PER_PLATE_TREATMENT = 5
NEGCON_SAMPLES_PER_PLATE = 1
PRE_POSITIVE_SAMPLES_PER_GROUP = 10
PRE_NEGCON_SAMPLES_PER_GROUP = 2
RANDOM_SEED = 0


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the sampling script.

    Returns:
        Parsed namespace containing the merged-profile input directory and
        sampled parquet output path.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Sample fixed-count nuclei from downloaded merged CPG0016 single-cell "
            "parquets and merge them with DNA image URLs."
        )
    )
    parser.add_argument(
        "--downloaded-profiles-dir",
        type=Path,
        default=DEFAULT_DOWNLOADED_PROFILES_DIR,
        help="Directory containing downloaded Nuclei_Image_merged.parquet files.",
    )
    parser.add_argument(
        "--output-parquet",
        type=Path,
        default=DEFAULT_OUTPUT_PARQUET,
        help="Path for the sampled single-cell parquet output.",
    )
    parser.add_argument(
        "--load-data-csv-dir",
        type=Path,
        default=DEFAULT_LOAD_DATA_CSV_DIR,
        help="Directory containing downloaded load_data_with_illum.csv files.",
    )
    return parser.parse_args()


def discover_merged_profile_paths(downloaded_profiles_dir: Path) -> list[Path]:
    """Discover merged single-cell profile parquet paths.

    Args:
        downloaded_profiles_dir: Root directory containing recursively nested
            ``Nuclei_Image_merged.parquet`` files.

    Returns:
        Sorted list of discovered merged parquet paths.

    Raises:
        FileNotFoundError: If no merged parquet files are found.
    """
    merged_profile_paths = sorted(downloaded_profiles_dir.rglob(MERGED_PROFILE_FILENAME))
    if not merged_profile_paths:
        raise FileNotFoundError(
            f"No {MERGED_PROFILE_FILENAME} files found under {downloaded_profiles_dir}."
        )
    return merged_profile_paths


def load_downloaded_profile_groups(merged_profile_paths: list[Path]) -> pd.DataFrame:
    """Load and concatenate only well-group keys from merged profile parquets.

    Args:
        merged_profile_paths: Discovered merged profile parquet paths.

    Returns:
        Concatenated dataframe restricted to
        ``[Metadata_Source, Metadata_Batch, Metadata_Plate, Metadata_Well]``.

    Raises:
        ValueError: If required grouping columns are missing.
    """
    group_dfs: list[pd.DataFrame] = []
    for profile_path in merged_profile_paths:
        profile_df = pd.read_parquet(
            profile_path,
            columns=PROFILE_GROUP_COLUMNS,
        )
        missing_group_columns = [
            column
            for column in PROFILE_GROUP_COLUMNS
            if column not in profile_df.columns
        ]
        if missing_group_columns:
            raise ValueError(
                "Downloaded profiles are missing required grouping columns: "
                f"{missing_group_columns}"
            )
        if profile_df.empty:
            continue
        group_dfs.append(profile_df.loc[:, PROFILE_GROUP_COLUMNS])

    if not group_dfs:
        raise ValueError(
            "No non-empty merged profile parquets were found to build profile groups."
        )

    return pd.concat(group_dfs, axis=0, ignore_index=True)


def merge_profiles_with_metadata(profiles_df: pd.DataFrame) -> pd.DataFrame:
    """Attach perturbation metadata and remove empty perturbations.

    Args:
        profiles_df: Concatenated profile group dataframe.

    Returns:
        Dataframe restricted to rows with matching perturbation metadata and
        enriched with treatment metadata columns.

    Raises:
        ValueError: If required metadata columns are missing from the metadata
            source.
    """
    perturbation_metadata_df = load_experiment_perturbation_metadata()
    perturbation_metadata_df = perturbation_metadata_df.loc[
        perturbation_metadata_df["Metadata_pert_type"] != "empty"
    ].copy()

    required_metadata_columns = PERTURBATION_METADATA_JOIN_COLUMNS + [
        "Metadata_JCP2022",
        "Metadata_pert_type",
        "Metadata_Name",
    ]
    missing_metadata_columns = [
        column
        for column in required_metadata_columns
        if column not in perturbation_metadata_df.columns
    ]
    if missing_metadata_columns:
        raise ValueError(
            "Perturbation metadata is missing required columns: "
            f"{missing_metadata_columns}"
        )

    metadata_columns = [
        column
        for column in perturbation_metadata_df.columns
        if column not in profiles_df.columns
        or column in PERTURBATION_METADATA_JOIN_COLUMNS
    ]
    metadata_df = perturbation_metadata_df.loc[:, metadata_columns].drop_duplicates()

    merged_df = pd.merge(
        left=profiles_df,
        right=metadata_df,
        how="inner",
        on=PERTURBATION_METADATA_JOIN_COLUMNS,
    )
    required_columns = PROFILE_GROUP_COLUMNS + [
        "Metadata_JCP2022",
        "Metadata_pert_type",
        "Metadata_Name",
    ]
    missing_required_columns = [
        column for column in required_columns if column not in merged_df.columns
    ]
    if missing_required_columns:
        raise ValueError(
            "Merged profile metadata is missing required columns: "
            f"{missing_required_columns}"
        )
    return merged_df.loc[:, required_columns].drop_duplicates()


def load_load_data_metadata(load_data_csv_dir: Path) -> pd.DataFrame:
    """Load and trim ``load_data_with_illum.csv`` metadata for DNA URLs.

    Args:
        load_data_csv_dir: Root directory containing downloaded
            ``load_data_with_illum.csv`` files.

    Returns:
        Deduplicated dataframe containing only the requested DNA URL columns and
        merge keys.

    Raises:
        ValueError: If required load-data metadata columns are missing.
    """
    if CPG0016LoadDataWithIllumDownloader is not None:
        downloader = CPG0016LoadDataWithIllumDownloader(
            csv_download_dir=load_data_csv_dir,
            parallel=True,
            workers=8,
            use_existing_csvs_without_s3_check=True,
        )
        load_data_df = downloader.get_dataframe()
    else:
        csv_paths = sorted(load_data_csv_dir.rglob("load_data_with_illum.csv"))
        if not csv_paths:
            raise FileNotFoundError(
                f"No load_data_with_illum.csv files found under {load_data_csv_dir}."
            )
        load_data_df = pd.concat(
            [pd.read_csv(csv_path) for csv_path in csv_paths],
            axis=0,
            ignore_index=True,
        )

    missing_load_data_columns = [
        column for column in LOAD_DATA_COLUMNS if column not in load_data_df.columns
    ]
    if missing_load_data_columns:
        raise ValueError(
            "Load-data metadata is missing required columns: "
            f"{missing_load_data_columns}"
        )

    return load_data_df.loc[:, LOAD_DATA_COLUMNS].drop_duplicates(
        subset=PROFILE_JOIN_COLUMNS
    )


def random_group_sample(
    dataframe: pd.DataFrame,
    group_columns: list[str],
    n_rows: int,
    random_seed: int,
) -> pd.DataFrame:
    """Select an exact seeded random sample per group.

    Args:
        dataframe: Input dataframe.
        group_columns: Columns defining the grouping for fixed-count sampling.
        n_rows: Number of rows to retain from each group.
        random_seed: Seed used to drive reproducible per-group sampling.

    Returns:
        Dataframe containing up to ``n_rows`` randomly sampled rows from each
        group.
    """
    if dataframe.empty:
        return dataframe.copy()

    sampled_groups: list[pd.DataFrame] = []
    for _, group_df in dataframe.groupby(group_columns, sort=True, group_keys=False):
        sampled_groups.append(
            group_df.sample(n=min(n_rows, len(group_df)), random_state=random_seed)
        )
    return pd.concat(sampled_groups, axis=0, ignore_index=True)


def sample_profiles(
    profiles_df: pd.DataFrame,
    positive_n_rows: int,
    negcon_n_rows: int,
) -> pd.DataFrame:
    """Sample positive treatments and negative controls with a seeded RNG.

    Args:
        profiles_df: Metadata-enriched single-cell dataframe.

    Returns:
        Concatenated dataframe containing up to the requested counts per
        ``(Metadata_Source, Metadata_Batch, Metadata_Plate, Metadata_JCP2022)``
        group.
    """
    positive_df = profiles_df.loc[profiles_df["Metadata_pert_type"] != "negcon"].copy()
    sampled_positive_df = random_group_sample(
        dataframe=positive_df,
        group_columns=GROUP_SAMPLING_COLUMNS,
        n_rows=positive_n_rows,
        random_seed=RANDOM_SEED,
    )

    negcon_df = profiles_df.loc[profiles_df["Metadata_pert_type"] == "negcon"].copy()
    sampled_negcon_df = random_group_sample(
        dataframe=negcon_df,
        group_columns=GROUP_SAMPLING_COLUMNS,
        n_rows=negcon_n_rows,
        random_seed=RANDOM_SEED,
    )

    return pd.concat(
        [sampled_positive_df, sampled_negcon_df],
        axis=0,
        ignore_index=True,
    )


def load_presampled_profiles_for_parquet(
    profile_path: Path,
    metadata_df: pd.DataFrame,
) -> pd.DataFrame:
    """Load one parquet, merge with metadata, and apply relaxed pre-sampling.

    Args:
        profile_path: Parquet file to process.
        metadata_df: Perturbation metadata restricted to available profile well
            groups.

    Returns:
        Pre-sampled dataframe for one parquet file.
    """
    profile_df = pd.read_parquet(profile_path, columns=PARQUET_SAMPLE_COLUMNS)
    if profile_df.empty:
        return pd.DataFrame(columns=SAMPLED_OUTPUT_COLUMNS)

    missing_sample_columns = [
        column for column in PARQUET_SAMPLE_COLUMNS if column not in profile_df.columns
    ]
    if missing_sample_columns:
        raise ValueError(
            f"Downloaded profile parquet {profile_path} is missing required columns: "
            f"{missing_sample_columns}"
        )

    merged_df = pd.merge(
        left=profile_df,
        right=metadata_df,
        how="inner",
        on=PERTURBATION_METADATA_JOIN_COLUMNS,
    )
    if merged_df.empty:
        return pd.DataFrame(columns=SAMPLED_OUTPUT_COLUMNS)

    sampled_df = sample_profiles(
        merged_df,
        positive_n_rows=PRE_POSITIVE_SAMPLES_PER_GROUP,
        negcon_n_rows=PRE_NEGCON_SAMPLES_PER_GROUP,
    )
    return sampled_df.loc[:, SAMPLED_OUTPUT_COLUMNS]


def load_presampled_profiles(
    merged_profile_paths: list[Path],
    metadata_df: pd.DataFrame,
) -> pd.DataFrame:
    """Process merged profile parquets one at a time and concatenate pre-samples.

    Args:
        merged_profile_paths: Discovered merged profile parquet paths.
        metadata_df: Perturbation metadata restricted to available profile well
            groups.

    Returns:
        Concatenated pre-sampled dataframe across all merged profile parquets.
    """
    presampled_dfs: list[pd.DataFrame] = []
    for profile_path in merged_profile_paths:
        presampled_df = load_presampled_profiles_for_parquet(profile_path, metadata_df)
        if not presampled_df.empty:
            presampled_dfs.append(presampled_df)
    if not presampled_dfs:
        return pd.DataFrame(columns=SAMPLED_OUTPUT_COLUMNS)
    return pd.concat(presampled_dfs, axis=0, ignore_index=True)


def finalize_global_quotas(sampled_profiles_df: pd.DataFrame) -> pd.DataFrame:
    """Apply final exact global quotas after concatenating pre-sampled rows."""
    return sample_profiles(
        sampled_profiles_df,
        positive_n_rows=POSITIVE_SAMPLES_PER_PLATE_TREATMENT,
        negcon_n_rows=NEGCON_SAMPLES_PER_PLATE,
    ).loc[:, SAMPLED_OUTPUT_COLUMNS]


def merge_sampled_with_load_data(
    sampled_profiles_df: pd.DataFrame,
    load_data_df: pd.DataFrame,
) -> pd.DataFrame:
    """Inner join sampled profiles with trimmed load-data metadata.

    Args:
        sampled_profiles_df: Trimmed sampled profile dataframe.
        load_data_df: Trimmed load-data dataframe containing DNA URL columns.

    Returns:
        Inner-joined dataframe restricted to the final requested schema.
    """
    merged_df = pd.merge(
        left=sampled_profiles_df,
        right=load_data_df,
        how="inner",
        on=PROFILE_JOIN_COLUMNS,
    )
    return merged_df.loc[:, FINAL_OUTPUT_COLUMNS]


def main() -> None:
    """Run the merged-profile sampling pipeline and write parquet output.

    Returns:
        None.
    """
    args = parse_args()
    merged_profile_paths = discover_merged_profile_paths(args.downloaded_profiles_dir)
    profile_groups_df = load_downloaded_profile_groups(merged_profile_paths)
    metadata_df = merge_profiles_with_metadata(profile_groups_df)
    profilesdf = load_presampled_profiles(merged_profile_paths, metadata_df)
    profilesdf = finalize_global_quotas(profilesdf)
    load_data_df = load_load_data_metadata(args.load_data_csv_dir)
    profilesdf = merge_sampled_with_load_data(profilesdf, load_data_df)
    args.output_parquet.parent.mkdir(parents=True, exist_ok=True)
    profilesdf.to_parquet(args.output_parquet, index=False)
    print(f"Saved {len(profilesdf)} sampled rows to {args.output_parquet}")


if __name__ == "__main__":
    main()
