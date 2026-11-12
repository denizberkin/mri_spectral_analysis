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


def get_ldct_dir() -> str:
    """Folder that directly contains the patient folders (C001 ... L100)."""
    return _get("LDCT_DIR")


def get_ldct_preprocessed_dir() -> str | None:
    """Where scripts/preprocess_ldct.py writes the cached slices. None if not set."""
    return _get("LDCT_PREPROCESSED_DIR", required=False)
