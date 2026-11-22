# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — release version surface audit

"""Audit source and installed package version surfaces."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib as _tomllib
else:
    import tomli as _tomllib


ROOT = Path(__file__).resolve().parents[1]


#: The label of the package's own version line, also its path below the checkout.
VERSION_LABEL = "src/sc_neurocore/_version.py"

#: The package root, which re-exports that version as ``__version__``.
INIT_LABEL = "src/sc_neurocore/__init__.py"

#: The line by which the package root imports the version module.
INIT_IMPORT = "from sc_neurocore import _version"

#: The one line by which the package root re-exports the version as ``__version__``.
INIT_REEXPORT = "__version__: str = _version.VERSION"


@dataclass(frozen=True)
class VersionSurface:
    """One place that states a release version.

    Attributes
    ----------
    label : str
        A checked-in file's path below the checkout, or ``installed:<name>[:<dist-info>]``
        for an installed distribution.
    version : str
        The version it states.
    """

    label: str
    version: str


def _normalise_distribution_name(name: str) -> str:
    """Return a distribution name in its PEP 503 normal form.

    Parameters
    ----------
    name : str
        The name as the metadata states it.

    Returns
    -------
    str
        Lower case, with each run of ``-``, ``_`` and ``.`` replaced by one ``-``.
    """
    return re.sub(r"[-_.]+", "-", name).lower()


def _toml_version(path: Path, table: str) -> str:
    """Return the ``version`` key of one table of a TOML manifest.

    Parameters
    ----------
    path : Path
        The manifest.
    table : str
        The table that holds the version, such as ``project`` or ``package``.

    Returns
    -------
    str
        The version.
    """
    payload = _tomllib.loads(path.read_text(encoding="utf-8"))
    return str(payload[table]["version"])


def _init_version(root: Path) -> str:
    """Return the package's ``VERSION``, refusing a root that states another one.

    The version is stated once, in ``_version.py``. The package root must
    re-export it and must not state a version of its own.

    Parameters
    ----------
    root : Path
        The checkout.

    Returns
    -------
    str
        The version string.

    Raises
    ------
    RuntimeError
        If ``_version.py`` defines no ``VERSION``, or if the package root does not
        re-export it or states a literal ``__version__``; named by its label in
        any checkout.
    """
    text = (root / VERSION_LABEL).read_text(encoding="utf-8")
    match = re.search(
        r'^VERSION\s*(?::\s*Final\[str\])?\s*=\s*["\']([^"\']+)["\']', text, re.MULTILINE
    )
    if match is None:
        raise RuntimeError(f"{VERSION_LABEL} does not define VERSION")
    init_lines = (root / INIT_LABEL).read_text(encoding="utf-8").splitlines()
    bindings = [line for line in init_lines if re.match(r"__version__\s*[:=]", line)]
    if any(line != INIT_REEXPORT for line in bindings):
        raise RuntimeError(f"{INIT_LABEL} states its own __version__; re-export VERSION")
    if not bindings or INIT_IMPORT not in init_lines:
        raise RuntimeError(f"{INIT_LABEL} does not re-export VERSION as __version__")
    return match.group(1)


def _metadata_version(root: Path, label: str, pattern: str) -> str:
    """Return the version a release metadata file states, or refuse a file that states none.

    Parameters
    ----------
    root : Path
        The checkout.
    label : str
        The file's path below ``root``, also its label in the audit.
    pattern : str
        A multiline regular expression whose first group is the version.

    Returns
    -------
    str
        The stated version.

    Raises
    ------
    RuntimeError
        If the file states no version in the expected form.
    """
    match = re.search(pattern, (root / label).read_text(encoding="utf-8"), re.MULTILINE)
    if match is None:
        raise RuntimeError(f"{label} does not state a version")
    return match.group(1)


def _zenodo_version(root: Path) -> str:
    """Return the version ``.zenodo.json`` gives the archive, or refuse a file without one.

    Parameters
    ----------
    root : Path
        The checkout.

    Returns
    -------
    str
        The ``version`` field.

    Raises
    ------
    RuntimeError
        If the file has no string ``version`` field.
    """
    payload = json.loads((root / ".zenodo.json").read_text(encoding="utf-8"))
    version = payload.get("version") if isinstance(payload, dict) else None
    if not isinstance(version, str):
        raise RuntimeError(".zenodo.json does not state a version")
    return version


def source_surfaces(root: Path = ROOT) -> list[VersionSurface]:
    """Return every checked-in surface that states the release version.

    The package and engine manifests come first, the project version leading.
    The release metadata follows: ``.zenodo.json`` is what an archived release
    records, ``CITATION.cff`` is how the work is cited, and ``conda/meta.yaml``
    builds the conda package. Each states the version by hand, so each is held
    to the project version here.

    Parameters
    ----------
    root : Path
        The checkout.

    Returns
    -------
    list[VersionSurface]
        One surface per file, in a fixed order.
    """
    return [
        VersionSurface("pyproject.toml", _toml_version(root / "pyproject.toml", "project")),
        VersionSurface(VERSION_LABEL, _init_version(root)),
        VersionSurface(
            "engine/Cargo.toml", _toml_version(root / "engine" / "Cargo.toml", "package")
        ),
        VersionSurface(
            "bridge/pyproject.toml",
            _toml_version(root / "bridge" / "pyproject.toml", "project"),
        ),
        VersionSurface(".zenodo.json", _zenodo_version(root)),
        VersionSurface(
            "CITATION.cff",
            _metadata_version(root, "CITATION.cff", r"""^version:\s*["']?([^"'\s]+)["']?\s*$"""),
        ),
        VersionSurface(
            "conda/meta.yaml",
            _metadata_version(
                root, "conda/meta.yaml", r"""\{%\s*set\s+version\s*=\s*"([^"]+)"\s*%\}"""
            ),
        ),
    ]


def installed_surfaces() -> list[VersionSurface]:
    """Return the version of each installed SC-NeuroCore distribution.

    A stale ``.dist-info`` left beside the current one is returned too, labelled
    with its folder, so that the audit names it.

    Returns
    -------
    list[VersionSurface]
        One surface per installed ``sc-neurocore`` or ``sc-neurocore-engine``
        distribution.
    """
    surfaces: list[VersionSurface] = []
    expected_names = {"sc-neurocore", "sc-neurocore-engine"}
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata.get("Name", "")
        normalised = _normalise_distribution_name(name)
        if normalised not in expected_names:
            continue
        dist_path = getattr(distribution, "_path", None)
        label = f"installed:{normalised}"
        if dist_path is not None:
            label = f"{label}:{Path(dist_path).name}"
        surfaces.append(VersionSurface(label, distribution.version))
    return surfaces


def audit_versions(*, include_installed: bool = True, root: Path = ROOT) -> list[str]:
    """Return one line per surface that does not state the project version.

    Parameters
    ----------
    include_installed : bool
        Whether the installed distributions are held to it too.
    root : Path
        The checkout.

    Returns
    -------
    list[str]
        ``"<label> has version <v>, expected <project version>"`` per mismatch;
        empty when every surface agrees.
    """
    surfaces = source_surfaces(root)
    if include_installed:
        surfaces.extend(installed_surfaces())

    expected = surfaces[0].version
    return [
        f"{surface.label} has version {surface.version}, expected {expected}"
        for surface in surfaces
        if surface.version != expected
    ]


def remedies(failures: list[str]) -> list[str]:
    """Return what to do about the audit's failures, by the kind of surface that failed.

    A checked-in file is corrected by editing it; an installed distribution by
    reinstalling the checkout. Each remedy is given once, and only for a kind
    of surface that failed.

    Parameters
    ----------
    failures : list[str]
        The lines :func:`audit_versions` returned.

    Returns
    -------
    list[str]
        The remedies, checked-in files first.
    """
    found: list[str] = []
    if any(not failure.startswith("installed:") for failure in failures):
        found.append(
            "Edit each named file to state the project version (pyproject.toml); "
            "these files are not generated."
        )
    if any(failure.startswith("installed:") for failure in failures):
        found.append("Refresh the environment with: python -m pip install -e .")
    return found


def main(argv: list[str] | None = None) -> int:
    """Run the audit and report each mismatch with its remedy.

    Parameters
    ----------
    argv : list[str] or None
        Command-line arguments; the process's own when None.

    Returns
    -------
    int
        0 when every surface states the project version, 1 otherwise.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-only",
        action="store_true",
        help="Only compare checked-in source metadata; skip installed distributions.",
    )
    args = parser.parse_args(argv)

    failures = audit_versions(include_installed=not args.source_only)
    if not failures:
        return 0

    print("Version surface audit failed:")
    for failure in failures:
        print(f"- {failure}")
    for remedy in remedies(failures):
        print(remedy)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
