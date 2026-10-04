"""Locate repository resources for source-only experiment launchers."""
import os
from pathlib import Path

def repository_root() -> Path:
    override = os.environ.get("RTP_CODEC_ROOT")
    if override:
        root = Path(override).resolve()
        if not (root / "pyproject.toml").is_file():
            raise FileNotFoundError(f"RTP_CODEC_ROOT is not a repository checkout: {root}")
        return root
    for root in Path(__file__).resolve().parents:
        if (root / "pyproject.toml").is_file() and (root / "research" / "scripts").is_dir():
            return root
    raise FileNotFoundError("This launcher requires a source checkout; set RTP_CODEC_ROOT.")
