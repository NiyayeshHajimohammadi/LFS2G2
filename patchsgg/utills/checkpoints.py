"""Checkpoint resolution and loading utilities."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional, Union

import torch
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import EntryNotFoundError, RepositoryNotFoundError

logger = logging.getLogger(__name__)


def resolve_checkpoint_path(
    local_path: Optional[Union[str, Path]],
    *,
    hf_repo_id: Optional[str] = None,
    hf_filename: Optional[str] = None,
    revision: str = "main",
    cache_dir: Optional[Union[str, Path]] = None,
    local_files_only: bool = False,
    token: Optional[Union[str, bool]] = None,
) -> Path:
    """Return a local checkpoint path, downloading one Hub file if necessary.

    Resolution order:

    1. Existing local file.
    2. Cached or downloadable Hugging Face Hub file.
    3. Raise FileNotFoundError.
    """
    resolved_local: Optional[Path] = None

    if local_path:
        resolved_local = Path(local_path).expanduser()

        if resolved_local.is_file():
            logger.info("Using local checkpoint: %s", resolved_local)
            return resolved_local

    if hf_repo_id is None:
        missing = str(resolved_local) if resolved_local else "<not configured>"
        raise FileNotFoundError(
            f"Checkpoint not found locally at {missing}, and no Hugging Face "
            "repository was configured."
        )

    if hf_filename is None:
        if resolved_local is None:
            raise ValueError(
                "hf_filename is required when local_path is not configured."
            )
        hf_filename = resolved_local.name

    try:
        downloaded = hf_hub_download(
            repo_id=hf_repo_id,
            filename=hf_filename,
            revision=revision,
            cache_dir=str(cache_dir) if cache_dir else None,
            local_files_only=local_files_only,
            token=token,
        )
    except RepositoryNotFoundError as exc:
        raise FileNotFoundError(
            f"Hugging Face repository {hf_repo_id!r} does not exist or "
            "is not accessible."
        ) from exc
    except EntryNotFoundError as exc:
        raise FileNotFoundError(
            f"File {hf_filename!r} was not found in repository "
            f"{hf_repo_id!r} at revision {revision!r}."
        ) from exc

    path = Path(downloaded)
    logger.info("Resolved Hub checkpoint to: %s", path)
    return path


def load_torch_checkpoint(
    local_path: Optional[Union[str, Path]],
    *,
    hf_repo_id: Optional[str] = None,
    hf_filename: Optional[str] = None,
    revision: str = "main",
    map_location: Union[str, torch.device] = "cpu",
    cache_dir: Optional[Union[str, Path]] = None,
    local_files_only: bool = False,
    token: Optional[Union[str, bool]] = None,
    weights_only: bool = True,
) -> Any:
    """Resolve and load a tensor, state dict, or safe PyTorch checkpoint."""
    checkpoint_path = resolve_checkpoint_path(
        local_path,
        hf_repo_id=hf_repo_id,
        hf_filename=hf_filename,
        revision=revision,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
        token=token,
    )

    try:
        return torch.load(
            checkpoint_path,
            map_location=map_location,
            weights_only=weights_only,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Unable to load checkpoint from {checkpoint_path}"
        ) from exc