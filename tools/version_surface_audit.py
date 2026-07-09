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
import re
import sys
from dataclasses import dataclass
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib as _tomllib
else:
    import tomli as _tomllib


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class VersionSurface:
    label: str
    version: str


def _normalise_distribution_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _toml_version(path: Path, table: str) -> str:
    payload = _tomllib.loads(path.read_text(encoding="utf-8"))
    return str(payload[table]["version"])


def _init_version(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    match = re.search(r'^__version__\s*=\s*["\']([^"\']+)["\']', text, re.MULTILINE)
    if match is None:
        raise RuntimeError(f"{path.relative_to(ROOT)} does not define __version__")
    return match.group(1)


def source_surfaces(root: Path = ROOT) -> list[VersionSurface]:
    return [
        VersionSurface("pyproject.toml", _toml_version(root / "pyproject.toml", "project")),
        VersionSurface(
            "src/sc_neurocore/__init__.py",
            _init_version(root / "src" / "sc_neurocore" / "__init__.py"),
        ),
        VersionSurface(
            "engine/Cargo.toml", _toml_version(root / "engine" / "Cargo.toml", "package")
        ),
        VersionSurface(
            "bridge/pyproject.toml",
            _toml_version(root / "bridge" / "pyproject.toml", "project"),
        ),
    ]


def installed_surfaces() -> list[VersionSurface]:
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
    surfaces = source_surfaces(root)
    if include_installed:
        surfaces.extend(installed_surfaces())

    expected = surfaces[0].version
    return [
        f"{surface.label} has version {surface.version}, expected {expected}"
        for surface in surfaces
        if surface.version != expected
    ]


def main(argv: list[str] | None = None) -> int:
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
    print("Refresh the environment with: python -m pip install -e .")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
