"""Local-first checkpoint resolution with an optional Hugging Face Hub fallback."""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def get_model_path_with_hf_fallback(
    local_path: str,
    hf_repo_id: Optional[str] = None,
    filename: Optional[str] = None,
    cache_dir: Optional[str] = None,
) -> str:
    """Return a local checkpoint path, downloading it from HF only when needed.

    ``local_path`` always wins. The Hub dependency is imported lazily so toy/offline workflows do
    not require network packages. A missing local file without ``hf_repo_id`` raises a precise
    ``FileNotFoundError`` instead of failing later inside ``torch.load``.
    """
    resolved = Path(os.path.expandvars(os.path.expanduser(str(local_path))))
    if resolved.is_file():
        return str(resolved)
    if not hf_repo_id:
        raise FileNotFoundError(
            f"checkpoint not found at {resolved}. Set the corresponding *_hf_repo_id config field "
            "to enable a Hugging Face fallback."
        )
    try:
        from huggingface_hub import hf_hub_download
        from huggingface_hub.utils import HfHubHTTPError
    except ImportError as exc:
        raise ImportError(
            "huggingface_hub is required because the local checkpoint is missing; install project dependencies"
        ) from exc

    remote_filename = filename or resolved.name
    try:
        downloaded = hf_hub_download(
            repo_id=str(hf_repo_id),
            filename=str(remote_filename),
            cache_dir=None if cache_dir is None else str(cache_dir),
        )
    except (HfHubHTTPError, OSError) as exc:
        raise FileNotFoundError(
            f"checkpoint was absent locally ({resolved}) and could not be downloaded from "
            f"{hf_repo_id}/{remote_filename}: {exc}"
        ) from exc
    logger.info("downloaded checkpoint %s/%s to %s", hf_repo_id, remote_filename, downloaded)
    return downloaded
