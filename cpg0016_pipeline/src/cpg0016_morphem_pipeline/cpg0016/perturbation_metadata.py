"""Utilities for loading merged JUMP experiment and perturbation metadata."""

from __future__ import annotations

import pandas as pd

WELL_URL = "https://github.com/jump-cellpainting/datasets/raw/refs/heads/main/metadata/well.csv.gz"
PLATE_URL = "https://github.com/jump-cellpainting/datasets/raw/refs/heads/main/metadata/plate.csv.gz"
PERTURBATION_CONTROL_URL = "https://raw.githubusercontent.com/jump-cellpainting/datasets/refs/heads/main/metadata/perturbation_control.csv"
PERTURBATION_URLS = {
    "compound": "https://github.com/jump-cellpainting/datasets/raw/refs/heads/main/metadata/compound.csv.gz",
    "crispr": "https://github.com/jump-cellpainting/datasets/raw/refs/heads/main/metadata/crispr.csv.gz",
    "orf": "https://github.com/jump-cellpainting/datasets/raw/refs/heads/main/metadata/orf.csv.gz",
}


def load_experiment_perturbation_metadata() -> pd.DataFrame:
    """Load plate, well, and perturbation metadata into one merged table.

    Returns:
        Dataframe containing merged experiment metadata and perturbation
        annotations keyed by ``Metadata_JCP2022``.

    Notes:
        This helper reads the current upstream metadata files from the public
        JUMP Cell Painting metadata repository each time it is called.
    """

    well_df = pd.read_csv(WELL_URL)
    plate_df = pd.read_csv(PLATE_URL)
    experiment_metadata_df = pd.merge(
        left=well_df,
        right=plate_df,
        how="inner",
        on=["Metadata_Source", "Metadata_Plate"],
    )

    perturbation_df = pd.concat(
        [pd.read_csv(url) for url in PERTURBATION_URLS.values()],
        axis=0,
        ignore_index=True,
    ).drop(columns="Metadata_pert_type")
    perturbation_control_df = pd.read_csv(PERTURBATION_CONTROL_URL)
    perturbation_metadata_df = pd.merge(
        left=perturbation_df,
        right=perturbation_control_df,
        how="left",
        on=["Metadata_JCP2022"],
    )
    perturbation_metadata_df["Metadata_Name"] = perturbation_metadata_df[
        "Metadata_Name_x"
    ].fillna(perturbation_metadata_df["Metadata_Name_y"])
    perturbation_metadata_df = perturbation_metadata_df.drop(
        columns=["Metadata_Name_x", "Metadata_Name_y"]
    )

    return pd.merge(
        left=experiment_metadata_df,
        right=perturbation_metadata_df,
        how="left",
        on=["Metadata_JCP2022"],
    )
