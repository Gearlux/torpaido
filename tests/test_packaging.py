"""The distribution metadata a PyPI upload actually requires.

Every rule here guards a failure the build, the test suite, the linters and the
examples all pass straight through — measured on the sibling packages' wheels before
their first releases, when ``twine check --strict`` refused them:

* a dependency written as a direct URL makes PyPI *reject the upload* — the file
  is uploaded, parsed, and refused, so the failure arrives after everything green;
* a ``readme``/``license-files`` entry naming a file that is not there fails at
  build time on a clean checkout but not in an editable install;
* missing readme/classifiers/URLs cost nothing at build time and produce a landing
  page with no description, no license, no home, and no issue tracker.

The rules read ``pyproject.toml``; nothing is built. Whether every extra RESOLVES
against PyPI needs the network, so the release workflow checks that instead
(``.github/workflows/release.yml``). The README is the landing page, so its links
into the repository must be absolute (the last rule).
"""

import re
import tomllib
from pathlib import Path
from typing import Any, Dict, List, cast

_REPO = Path(__file__).resolve().parent.parent
_PYPROJECT = _REPO / "pyproject.toml"


def _project() -> Dict[str, Any]:
    with _PYPROJECT.open("rb") as fh:
        return cast(Dict[str, Any], tomllib.load(fh)["project"])


def _every_requirement() -> List[str]:
    """Core dependencies plus every extra's — PyPI validates all of them alike."""
    project = _project()
    requirements = list(project.get("dependencies", []))
    for extra in project.get("optional-dependencies", {}).values():
        requirements.extend(extra)
    return requirements


def test_no_requirement_is_a_direct_url() -> None:
    """A ``name @ git+https://...`` requirement is refused by PyPI at upload time.

    Warehouse validates every ``Requires-Dist`` and rejects any whose requirement
    carries a URL. It applies to extras exactly as to core dependencies, which is the
    trap: an extra nobody installs still blocks the whole upload.
    """
    direct = [req for req in _every_requirement() if "@" in req and "://" in req.split("@", 1)[1]]

    assert not direct, f"PyPI rejects direct-URL requirements: {direct}"


def test_the_declared_readme_and_license_files_exist() -> None:
    """``readme`` / ``license-files`` name files that are actually in the tree.

    Both are read at BUILD time, so an editable install never notices a missing one;
    the failure surfaces on the clean checkout that builds the release artifact.
    """
    project = _project()
    declared = [project["readme"], *project.get("license-files", [])]
    missing = [name for name in declared if not (_REPO / name).exists()]

    assert not missing, f"pyproject names files that do not exist: {missing}"


def test_the_landing_page_metadata_is_present() -> None:
    """Description, readme, license, classifiers and URLs — the PyPI page's whole frame.

    Each is optional to the build, which is why all of them were missing at once:
    the wheel built identically without them and ``twine check --strict`` was the
    first thing to notice (``long_description missing``).
    """
    project = _project()
    missing = [key for key in ("description", "readme", "license", "classifiers", "urls") if not project.get(key)]

    assert not missing, f"release metadata missing from pyproject: {missing}"


def test_the_development_status_classifier_matches_the_version() -> None:
    """An ``aN``/``bN`` version says pre-release; the classifier must say so too.

    These drift in opposite directions — the version moves every release, the
    classifier is written once and forgotten — leaving a "Production/Stable" badge
    on an alpha.
    """
    project = _project()
    status = [c for c in project["classifiers"] if c.startswith("Development Status ::")]
    assert len(status) == 1, f"expected exactly one Development Status classifier, got {status}"

    is_prerelease = "a" in project["version"] or "b" in project["version"] or "rc" in project["version"]
    expected = "3 - Alpha" if is_prerelease else "4 - Beta"

    assert expected in status[0], f"version {project['version']} does not match classifier {status[0]!r}"


def test_the_changelog_is_shipped_in_the_sdist() -> None:
    """`MANIFEST.in` must carry `CHANGELOG.md` — nothing else pulls it in.

    setuptools builds an sdist from what is *referenced*: `readme` brings README.md,
    `license-files` brings LICENSE, `packages.find` + `package-data` bring the package.
    A CHANGELOG is referenced by nothing, so without `MANIFEST.in` it is silently absent.

    Asserted on the declaration rather than by building an sdist: building one takes
    seconds and needs a clean tree, and the declaration is what actually goes stale.
    """
    manifest = _REPO / "MANIFEST.in"
    assert manifest.exists(), "MANIFEST.in is missing; the sdist would ship no CHANGELOG"

    included = {
        line.split(maxsplit=1)[1].strip()
        for line in manifest.read_text().splitlines()
        if line.strip().startswith("include ")
    }
    missing = [name for name in ("CHANGELOG.md",) if name not in included]

    assert not missing, f"MANIFEST.in does not include: {missing}"


def _released_versions(changelog: str) -> List[str]:
    """The `## [<version>]` headings, newest first, leaving out `## [Unreleased]`.

    Keep a Changelog collects changes under `## [Unreleased]` between releases. That
    section names no version, so it is not the entry the declared version must match —
    the first check of this rule failed on exactly that, the day after the first release,
    when a follow-up change was written up under `Unreleased` as the format asks.
    """
    headings = re.findall(r"^## \[([^\]]+)\]", changelog, flags=re.MULTILINE)
    return [heading for heading in headings if heading.lower() != "unreleased"]


def test_an_unreleased_section_is_not_a_version() -> None:
    """Guard the helper: `Unreleased` is skipped, a real version is not."""
    assert _released_versions("## [Unreleased]\n\n- a change\n\n## [0.1.0a1] — 2026-09-28\n") == ["0.1.0a1"]
    assert _released_versions("## [0.2.0] — 2026-10-01\n\n## [0.1.0] — 2026-09-01\n") == ["0.2.0", "0.1.0"]
    assert _released_versions("## [Unreleased]\n\n- a change\n") == []


def test_the_newest_changelog_entry_is_the_declared_version() -> None:
    """The version being released has its CHANGELOG entry, and it is the newest one.

    The version in `pyproject.toml` is bumped in one edit and the CHANGELOG in
    another; the release workflow checks the tag against the first and nothing
    checks the second, so a release whose notes describe the previous version
    would publish without complaint. An `## [Unreleased]` section above it is
    allowed: bumping the version still fails here until the new version's own
    heading is written.
    """
    versions = _released_versions((_REPO / "CHANGELOG.md").read_text())

    assert versions, "CHANGELOG.md has no `## [<version>]` entry"
    assert (
        versions[0] == _project()["version"]
    ), f"newest CHANGELOG entry is {versions[0]!r}, pyproject.toml declares {_project()['version']!r}"


def test_the_readme_links_into_the_repository_are_absolute() -> None:
    """PyPI shows the README as the landing page and resolves no relative link: a link to
    ``docs/compiling.md`` there leads nowhere. Links within the repository's own pages may
    stay relative; the README's must name ``https://github.com/Gearlux/torpaido/blob/main/``.
    """
    targets = re.findall(r"\]\(([^)\s]+)\)", (_REPO / "README.md").read_text())
    relative = [target for target in targets if "://" not in target and not target.startswith("#")]

    assert not relative, f"README links PyPI cannot resolve: {relative}"
