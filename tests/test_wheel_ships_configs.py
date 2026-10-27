"""The wheel ships every YAML config of ``config/``, ``examples/`` and ``graphs/`` where ``pkg://`` finds it.

``pkg://<distribution>/<path>`` (confluid) reads a wheel-installed config from ``share/<distribution>/<path>``
(``setup.py``). The wheel is built from a COPY of the files git sees here (tracked, plus untracked ones not ignored),
never from the checkout itself: pip builds in place, and setuptools copies whatever a leftover ``build/lib`` holds into
the next wheel, so a stale tree could hide a missing file.
"""

import importlib.util
import shutil
import subprocess
import sys
import tomllib
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DISTRIBUTION = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["name"].lower().replace("_", "-")


def _source_copy(dest: Path) -> Path:
    """The files git sees in this project, copied to ``dest`` — no ``build/``, no ``*.egg-info``."""
    listing = ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"]
    names = subprocess.run(listing, cwd=ROOT, check=True, capture_output=True, text=True).stdout.split("\0")
    for name in names:
        source = ROOT / name
        if name and source.is_file():  # a tracked file deleted in the working tree is listed, but gone
            (dest / name).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, dest / name)
    return dest


def test_the_wheel_ships_every_yaml_config_where_a_pkg_reference_finds_it(tmp_path: Path) -> None:
    source = _source_copy(tmp_path / "source")
    expected = sorted(
        path.relative_to(source).as_posix()
        for top in ("config", "examples", "graphs")
        if (source / top).is_dir()
        for path in (source / top).rglob("*.yaml")
        if not any(part.startswith(".") for part in path.relative_to(source).parts)
    )
    isolation = [] if importlib.util.find_spec("setuptools") is None else ["--no-build-isolation"]
    out = tmp_path / "wheel"
    command = [sys.executable, "-m", "pip", "wheel", str(source), "--no-deps", "-q", "-w", str(out), *isolation]
    subprocess.run(command, check=True, capture_output=True)
    marker = f".data/data/share/{DISTRIBUTION}/"
    names = zipfile.ZipFile(next(out.glob("*.whl"))).namelist()
    shipped = sorted(name.split(marker, 1)[1] for name in names if marker in name)
    assert expected, "the project has no YAML config to ship — the test reads the wrong folders"
    assert shipped == expected
