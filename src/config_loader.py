"""
Utility to load the central YAML configuration file.

Usage:
    from src.config_loader import load_config
    cfg = load_config()          # loads config.yaml next to project root
    cfg = load_config("path/to/config.yaml")  # explicit path
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


# Default: config.yaml sits at the workspace root (one level above src/)
_DEFAULT_CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"


def load_config(config_path: str | os.PathLike | None = None) -> dict[str, Any]:
    """Load and return the YAML configuration as a nested dict.

    Args:
        config_path: Path to the YAML file.  Defaults to
                     ``<project_root>/config.yaml``.

    Returns:
        Parsed configuration dictionary.

    Raises:
        FileNotFoundError: If the config file does not exist.
    """
    path = Path(config_path) if config_path is not None else _DEFAULT_CONFIG_PATH
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path.resolve()}")

    with path.open("r", encoding="utf-8") as f:
        print(f"Loading configuration from: {path.resolve()}")
        cfg = yaml.safe_load(f)

    return cfg
