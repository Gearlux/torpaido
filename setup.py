"""Build step: ship the project's YAML configs (``config/``, ``examples/``, ``graphs/``) in the wheel.

They go as wheel data under ``share/<distribution>/``, keeping their paths, where ``pkg://<distribution>/<path>``
(confluid) finds them when the project is installed from its wheel; an editable install reads the checkout's files.

Everything else is declared in ``pyproject.toml``.
"""

import tomllib
from pathlib import Path
from typing import List, Tuple

from setuptools import setup

HERE = Path(__file__).resolve().parent


def shipped_configs(distribution: str) -> List[Tuple[str, List[str]]]:
    """Every ``config/``, ``examples/`` and ``graphs/`` YAML file as wheel data at ``share/<distribution>/<its path>``.

    ``pkg://<distribution>/<path>`` (confluid) reads a config there when this distribution is installed from its
    wheel; an editable install reads the checkout's file instead. Hidden files and folders stay out. Paths are
    relative to this file, as setuptools takes them.
    """
    files = sorted(
        path.relative_to(HERE)
        for top in ("config", "examples", "graphs")
        if (HERE / top).is_dir()
        for path in (HERE / top).rglob("*.yaml")
        if not any(part.startswith(".") for part in path.relative_to(HERE).parts)
    )
    folders = sorted({file.parent for file in files})
    return [
        (f"share/{distribution}/{folder.as_posix()}", [file.as_posix() for file in files if file.parent == folder])
        for folder in folders
    ]


#: The normalized name (PEP 503) — the folder under ``share/`` that ``pkg://`` reads.
DISTRIBUTION = tomllib.loads((HERE / "pyproject.toml").read_text())["project"]["name"].lower().replace("_", "-")

setup(data_files=shipped_configs(DISTRIBUTION))
