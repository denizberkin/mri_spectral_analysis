from __future__ import annotations

import os

from mrixfields import env as _base


def _get(name: str, required: bool = True) -> str | None:
    _base.load_env()
    value = os.environ.get(name)
    if not value:
        if required:
            raise RuntimeError(
                f"{name} is not set. Configure it in .env (repo root) "
                "or set the environment variable. See .env.example."
            )
        return None
    return _base._resolve_path(value)


def get_ct4h_dir() -> str:
    """DICOM root the index's series_dir paths are relative to.

    This is the folder that directly contains the downloaded patient folder, e.g.
    .../CT4Harmonization-Multicentric (which in turn holds CT4Harmonization-Multicentric/<study>/...).
    """
    return _get("CT4H_DIR")


def get_ct4h_preprocessed_dir() -> str | None:
    """Where the preprocessing script writes cached slices (and index.csv). None if not set."""
    return _get("CT4H_PREPROCESSED_DIR", required=False)
