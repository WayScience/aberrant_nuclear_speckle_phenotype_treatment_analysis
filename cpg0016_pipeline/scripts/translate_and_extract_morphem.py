#!/usr/bin/env python
"""Translate normalized crops and extract MorphEm embeddings."""

from __future__ import annotations

import argparse
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import zarr
from huggingface_hub import snapshot_download
from mlflow.entities import LoggedModel
from mlflow.exceptions import MlflowException
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModel


MORPHEM_MODEL_NAME = "CaicedoLab/MorphEm"
DEFAULT_MLFLOW_TRACKING_URI = "sqlite:////home/camo/projects/nuclear_speckles_analysis/mlflow.db"
SAMPLE_ID_METADATA_COLUMNS = ["Metadata_Plate", "Metadata_Well", "Metadata_Site"]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for translation and embedding extraction.

    Returns:
        Parsed command-line arguments containing zarr and parquet inputs,
        feature parquet output path, MLflow model settings, batch size, and
        inference device.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Load normalized single-cell DNA crops from a zarr store, translate them "
            "with an MLflow image-to-image model, extract MorphEm embeddings from the "
            "translated crops, and save the features with source metadata to parquet."
        )
    )
    parser.add_argument(
        "--input-zarr",
        type=Path,
        required=True,
        help="Path to the normalized crop zarr store.",
    )
    parser.add_argument(
        "--input-parquet",
        type=Path,
        required=True,
        help="Path to the sampled single-cell parquet used to generate the zarr store.",
    )
    parser.add_argument(
        "--output-parquet",
        type=Path,
        required=True,
        help="Destination parquet path for metadata plus MorphEm features.",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        required=True,
        help="MLflow logged model identifier for the image-to-image translation model.",
    )
    parser.add_argument(
        "--mlflow-tracking-uri",
        type=str,
        default=DEFAULT_MLFLOW_TRACKING_URI,
        help=(
            "MLflow tracking URI used to resolve --model-id. "
            f"Default: {DEFAULT_MLFLOW_TRACKING_URI}"
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Number of crops to process per translation and MorphEm batch.",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Device to use for translation and MorphEm inference.",
    )
    return parser.parse_args()


def load_morphem_model() -> torch.nn.Module:
    """Load the local cached MorphEm backbone from Hugging Face files.

    Returns:
        MorphEm backbone ready for feature extraction.

    Raises:
        RuntimeError: If expected MorphEm weights are missing after loading.
    """

    model_dir = Path(
        snapshot_download(
            repo_id=MORPHEM_MODEL_NAME,
            local_files_only=True,
        )
    )
    config = AutoConfig.from_pretrained(
        model_dir,
        trust_remote_code=True,
    )
    model = AutoModel.from_config(
        config,
        trust_remote_code=True,
    )

    if not hasattr(model, "all_tied_weights_keys"):
        model.post_init()

    incompatible_keys = model.load_state_dict(
        load_file(model_dir / "model.safetensors"),
        strict=False,
    )

    if incompatible_keys.missing_keys:
        raise RuntimeError(
            "Missing MorphEm weights after loading: "
            f"{incompatible_keys.missing_keys}"
        )

    return model


def load_translation_model(
    model_id: str,
    tracking_uri: str,
    device: torch.device,
) -> torch.nn.Module:
    """Load the MLflow translation model onto the requested device.

    Args:
        model_id: MLflow logged model identifier.
        tracking_uri: MLflow tracking backend URI.
        device: Torch device used for inference.

    Returns:
        Translation model moved to the requested device and set to eval mode.

    Raises:
        MlflowException: If the logged model cannot be resolved from MLflow.
    """

    mlflow.set_tracking_uri(tracking_uri)
    try:
        logged_model: LoggedModel = mlflow.get_logged_model(model_id)
    except MlflowException as exc:
        raise MlflowException(
            f"Could not load MLflow model_id='{model_id}' from tracking URI "
            f"'{mlflow.get_tracking_uri()}'."
        ) from exc

    model = mlflow.pytorch.load_model(logged_model.model_uri)
    model = model.to(device)
    model.eval()
    return model


def preprocess_morphem_single_channel(
    crops: torch.Tensor,
    output_size: tuple[int, int] = (224, 224),
    eps: float = 1e-7,
) -> torch.Tensor:
    """Standardize single-channel crops and resize them for MorphEm input.

    Args:
        crops: Tensor shaped ``(N, H, W)`` or ``(N, 1, H, W)``.
        output_size: Spatial size expected by MorphEm.
        eps: Numerical stability term for variance normalization.

    Returns:
        Normalized and resized tensor shaped ``(N, 1, output_h, output_w)``.

    Raises:
        ValueError: If the input does not represent single-channel crops.
    """

    if crops.ndim == 3:
        crops = crops.unsqueeze(1)

    if crops.ndim != 4:
        raise ValueError(
            "Expected crops shaped (N, H, W) or (N, 1, H, W), "
            f"but received {tuple(crops.shape)}."
        )

    if crops.shape[1] != 1:
        raise ValueError(
            f"Expected exactly one channel, but received {crops.shape[1]}."
        )

    crops = crops.to(torch.float32)
    mean = crops.mean(dim=(-2, -1), keepdim=True)
    variance = crops.var(
        dim=(-2, -1),
        correction=0,
        keepdim=True,
    )
    crops = (crops - mean) / torch.sqrt(variance + eps)

    return F.interpolate(
        crops,
        size=output_size,
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )


def resolve_device(device_name: str) -> torch.device:
    """Resolve the requested execution device, handling ``auto`` and CUDA checks.

    Args:
        device_name: Requested device string from the CLI.

    Returns:
        Concrete torch device used for translation and MorphEm inference.

    Raises:
        RuntimeError: If CUDA is explicitly requested but unavailable.
    """

    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return torch.device(device_name)


def build_sample_id(row: pd.Series) -> str:
    """Rebuild the stable sample ID stored in the zarr metadata.

    Args:
        row: Sampled single-cell parquet row.

    Returns:
        Stable sample identifier reconstructed from metadata and bbox midpoint.
    """

    bbox_center_x = (
        float(row["AreaShape_BoundingBoxMinimum_X"])
        + float(row["AreaShape_BoundingBoxMaximum_X"])
    ) / 2.0
    bbox_center_y = (
        float(row["AreaShape_BoundingBoxMinimum_Y"])
        + float(row["AreaShape_BoundingBoxMaximum_Y"])
    ) / 2.0

    sample_parts: list[str] = []
    for column in SAMPLE_ID_METADATA_COLUMNS:
        if column in row.index:
            sample_parts.append(f"{column.removeprefix('Metadata_').lower()}={row[column]}")
    sample_parts.append(f"center_x={bbox_center_x:.6f}")
    sample_parts.append(f"center_y={bbox_center_y:.6f}")
    return "|".join(sample_parts)


def load_and_validate_inputs(
    input_zarr: Path,
    input_parquet: Path,
) -> tuple[pd.DataFrame, zarr.Array, list[str], str]:
    """Load inputs and validate strict row alignment between parquet and zarr.

    Args:
        input_zarr: Normalized crop zarr path.
        input_parquet: Sampled single-cell parquet path.

    Returns:
        Tuple of sampled dataframe, zarr crop array, stored sample IDs, and the
        source parquet path recorded in zarr attributes.

    Raises:
        ValueError: If the zarr structure is incomplete or the sampled parquet
            does not align exactly with the zarr metadata.
    """

    sampled_df = pd.read_parquet(input_parquet).reset_index(drop=True)
    root = zarr.open(str(input_zarr), mode="r")
    if "crops" not in root:
        raise ValueError(f"Zarr store does not contain a 'crops' dataset: {input_zarr}")

    crops = root["crops"]
    row_count = int(root.attrs.get("row_count", -1))
    sample_ids = list(root.attrs.get("sample_ids", []))
    source_parquet_path = str(root.attrs.get("source_parquet_path", ""))

    if row_count < 0:
        raise ValueError(f"Zarr store is missing a valid 'row_count' attribute: {input_zarr}")

    if len(sampled_df) != row_count:
        raise ValueError(
            "Input parquet row count does not match zarr row_count: "
            f"parquet_rows={len(sampled_df)}, zarr_row_count={row_count}"
        )

    if crops.shape[0] != row_count:
        raise ValueError(
            "Zarr crop dataset length does not match zarr row_count: "
            f"crops_rows={crops.shape[0]}, zarr_row_count={row_count}"
        )

    if len(sample_ids) != row_count:
        raise ValueError(
            "Zarr sample_ids length does not match zarr row_count: "
            f"sample_ids={len(sample_ids)}, zarr_row_count={row_count}"
        )

    reconstructed_sample_ids = sampled_df.apply(build_sample_id, axis=1).tolist()
    if reconstructed_sample_ids != sample_ids:
        mismatch_index = next(
            index
            for index, (left, right) in enumerate(zip(reconstructed_sample_ids, sample_ids))
            if left != right
        )
        raise ValueError(
            "Input parquet rows do not align with zarr sample_ids. "
            f"First mismatch at row {mismatch_index}: "
            f"parquet_sample_id={reconstructed_sample_ids[mismatch_index]!r}, "
            f"zarr_sample_id={sample_ids[mismatch_index]!r}"
        )

    return sampled_df, crops, sample_ids, source_parquet_path


def _normalize_translated_batch(
    translated_batch: torch.Tensor,
    expected_batch_size: int,
) -> torch.Tensor:
    """Validate translation model output shape and normalize it to ``(N, 1, H, W)``.

    Args:
        translated_batch: Raw translation model output.
        expected_batch_size: Batch size expected from the current input slice.

    Returns:
        Translation output as a ``float32`` tensor shaped ``(N, 1, 128, 128)``.

    Raises:
        TypeError: If the model output is not a tensor.
        ValueError: If the batch size, channel count, or spatial size is wrong.
    """

    if not isinstance(translated_batch, torch.Tensor):
        raise TypeError(
            "Translation model output must be a torch.Tensor, "
            f"but received {type(translated_batch)!r}."
        )

    if translated_batch.ndim == 3:
        if translated_batch.shape[0] != expected_batch_size:
            raise ValueError(
                "Translated batch size does not match input batch size: "
                f"translated_batch_size={translated_batch.shape[0]}, "
                f"expected_batch_size={expected_batch_size}"
            )
        translated_batch = translated_batch.unsqueeze(1)
    elif translated_batch.ndim == 4:
        if translated_batch.shape[0] != expected_batch_size:
            raise ValueError(
                "Translated batch size does not match input batch size: "
                f"translated_batch_size={translated_batch.shape[0]}, "
                f"expected_batch_size={expected_batch_size}"
            )
    else:
        raise ValueError(
            "Expected translated crops shaped (N, 128, 128) or (N, 1, 128, 128), "
            f"but received {tuple(translated_batch.shape)}."
        )

    if translated_batch.shape[1] != 1:
        raise ValueError(
            "Expected translated crops to have exactly one channel, "
            f"but received shape {tuple(translated_batch.shape)}."
        )

    if translated_batch.shape[-2:] != (128, 128):
        raise ValueError(
            "Expected translated crops to have spatial size (128, 128), "
            f"but received shape {tuple(translated_batch.shape)}."
        )

    return translated_batch.to(torch.float32)


def _run_translation_model(
    model: torch.nn.Module,
    crop_batch: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Run one crop batch through the translation model.

    Args:
        model: Translation model loaded from MLflow.
        crop_batch: One batch of normalized DNA crops.
        device: Device used for inference.

    Returns:
        Translated crop batch normalized to ``(N, 1, 128, 128)``.
    """

    model_input = crop_batch.to(torch.float32).unsqueeze(1)
    model_input = model_input.to(device, non_blocking=device.type == "cuda")
    translated_batch = model(model_input)
    translated_batch = _normalize_translated_batch(
        translated_batch=translated_batch,
        expected_batch_size=crop_batch.shape[0],
    )
    return translated_batch


def extract_embeddings_from_translated_zarr(
    crops: zarr.Array,
    translation_model: torch.nn.Module,
    morphem_model: torch.nn.Module,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    """Translate zarr crops in batches and extract MorphEm embeddings.

    Args:
        crops: Zarr-backed crop array.
        translation_model: MLflow translation model.
        morphem_model: MorphEm backbone used for feature extraction.
        device: Torch device used for inference.
        batch_size: Number of crops to process per iteration.

    Returns:
        Two-dimensional embedding array with one row per crop.

    Raises:
        ValueError: If ``batch_size`` is invalid or no embeddings are produced.
    """

    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")

    embedding_batches: list[np.ndarray] = []
    for start in range(0, crops.shape[0], batch_size):
        stop = min(start + batch_size, crops.shape[0])
        crop_batch = np.asarray(crops[start:stop], dtype=np.float32)
        crop_tensor = torch.from_numpy(crop_batch)
        translated_batch = _run_translation_model(
            model=translation_model,
            crop_batch=crop_tensor,
            device=device,
        )
        morphem_input = preprocess_morphem_single_channel(translated_batch)
        morphem_input = morphem_input.to(device, non_blocking=device.type == "cuda")
        embeddings = morphem_model.forward_features(morphem_input)["x_norm_clstoken"]
        embedding_batches.append(embeddings.cpu().numpy())

    if not embedding_batches:
        raise ValueError("No crops were available for translation and MorphEm feature extraction.")

    return np.concatenate(embedding_batches, axis=0)


def build_feature_dataframe(
    sampled_df: pd.DataFrame,
    sample_ids: list[str],
    source_parquet_path: str,
    embeddings: np.ndarray,
) -> pd.DataFrame:
    """Combine sampled metadata, traceability columns, and MorphEm features.

    Args:
        sampled_df: Input sampled parquet rows.
        sample_ids: Stable sample identifiers verified against the zarr store.
        source_parquet_path: Source parquet path stored in zarr attributes.
        embeddings: MorphEm embedding matrix.

    Returns:
        Output dataframe containing metadata, traceability columns, and
        ``morphem_feature_####`` columns.

    Raises:
        ValueError: If the embedding matrix is not two-dimensional or is not
            row-aligned with the sampled dataframe.
    """

    if embeddings.ndim != 2:
        raise ValueError(
            f"Expected embeddings to be 2D, but received shape {embeddings.shape}."
        )

    if len(sampled_df) != embeddings.shape[0]:
        raise ValueError(
            "Sampled parquet rows do not match extracted embedding rows: "
            f"parquet_rows={len(sampled_df)}, embedding_rows={embeddings.shape[0]}"
        )

    feature_columns = [
        f"morphem_feature_{feature_index:04d}"
        for feature_index in range(embeddings.shape[1])
    ]
    features_df = pd.DataFrame(embeddings, columns=feature_columns)
    traceability_df = pd.DataFrame(
        {
            "crop_row_index": np.arange(len(sampled_df), dtype=np.int64),
            "sample_id": sample_ids,
            "source_parquet_path": source_parquet_path,
        }
    )

    return pd.concat(
        [sampled_df.reset_index(drop=True), traceability_df, features_df],
        axis=1,
    )


def main() -> None:
    """Run translation and MorphEm feature extraction for one zarr/parquet pair.

    Returns:
        None.

    This entrypoint validates alignment between the sampled parquet and the
    normalized zarr store, loads the translation and MorphEm models, extracts a
    feature vector for every crop, and writes the combined output parquet used
    by both training and scoring.
    """

    args = parse_args()
    device = resolve_device(args.device)
    sampled_df, crops, sample_ids, source_parquet_path = load_and_validate_inputs(
        input_zarr=args.input_zarr,
        input_parquet=args.input_parquet,
    )

    translation_model = load_translation_model(
        model_id=args.model_id,
        tracking_uri=args.mlflow_tracking_uri,
        device=device,
    )
    morphem_model = load_morphem_model().to(device).eval()

    with torch.inference_mode():
        embeddings = extract_embeddings_from_translated_zarr(
            crops=crops,
            translation_model=translation_model,
            morphem_model=morphem_model,
            device=device,
            batch_size=args.batch_size,
        )

    output_df = build_feature_dataframe(
        sampled_df=sampled_df,
        sample_ids=sample_ids,
        source_parquet_path=source_parquet_path,
        embeddings=embeddings,
    )

    args.output_parquet.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_parquet(args.output_parquet, index=False)
    print(
        f"Saved {len(output_df)} rows with {embeddings.shape[1]} MorphEm features to "
        f"{args.output_parquet}"
    )


if __name__ == "__main__":
    main()
