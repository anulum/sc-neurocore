# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li

from __future__ import annotations

from pathlib import Path

if hasattr(__import__("sys"), "version_info") and __import__("sys").version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _project_version() -> str:
    pyproject = tomllib.loads((_repo_root() / "pyproject.toml").read_text(encoding="utf-8"))
    return str(pyproject["project"]["version"])


def test_public_release_surfaces_use_project_version() -> None:
    version = _project_version()
    surfaces = {
        "README.md": _repo_root() / "README.md",
        "docs/index.md": _repo_root() / "docs" / "index.md",
        "docs/CHANGELOG.md": _repo_root() / "docs" / "CHANGELOG.md",
    }

    for label, path in surfaces.items():
        text = path.read_text(encoding="utf-8")
        assert version in text, f"{label} does not mention package version {version}"


def test_engine_release_metadata_uses_project_version() -> None:
    version = _project_version()
    surfaces = {
        "engine/Cargo.toml": _repo_root() / "engine" / "Cargo.toml",
        "bridge/pyproject.toml": _repo_root() / "bridge" / "pyproject.toml",
    }

    for label, path in surfaces.items():
        payload = tomllib.loads(path.read_text(encoding="utf-8"))
        assert payload["package" if label.endswith("Cargo.toml") else "project"][
            "version"
        ] == version


def test_docs_index_does_not_advertise_previous_release_version() -> None:
    version = _project_version()
    major, minor, patch = (int(part) for part in version.split("."))
    if minor == 0:
        return
    previous_minor = f"{major}.{minor - 1}.{patch}"

    docs_index = (_repo_root() / "docs" / "index.md").read_text(encoding="utf-8")

    assert f"Version {previous_minor}" not in docs_index
