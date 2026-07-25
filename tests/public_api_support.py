# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Shared public-distribution contract helpers

"""Repository and package metadata helpers for public API contract tests."""

from pathlib import Path
import sys
from typing import Any

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib  # pragma: no cover - Python 3.10 compatibility path.


def _repo_root() -> Path:
    """Return the repository root used by distribution contract checks."""
    return Path(__file__).resolve().parents[1]


def _project_metadata() -> dict[str, Any]:
    """Load project metadata from the repository pyproject file."""
    pyproject = _repo_root() / "pyproject.toml"
    return tomllib.loads(pyproject.read_text(encoding="utf-8"))


def _package_root() -> Path:
    """Return the source-tree package root used by package-data tests."""
    return _repo_root() / "src" / "sc_neurocore"
