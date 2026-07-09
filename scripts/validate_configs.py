# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Repository configuration validator

"""Validate the repository configuration files expected by local maintenance.

The script is intentionally small: it checks that the repository root contains
the core project directories and verifies that ``pyproject.toml`` parses as
TOML. It can be executed from the repository root or from the ``scripts``
directory, matching the historical command-line behavior.
"""

from __future__ import annotations

import sys
from collections.abc import Mapping
from pathlib import Path
from typing import BinaryIO, Protocol, cast


REQUIRED_PATHS: tuple[str, ...] = (
    "pyproject.toml",
    "src/sc_neurocore",
    "hdl",
    "tests",
    "docs/API_REFERENCE.md",
    "docs/guides/USER_MANUAL.md",
)


class _TomlLoader(Protocol):
    """Minimal TOML loader protocol shared by ``tomllib`` and ``tomli``."""

    def load(self, file_obj: BinaryIO, /) -> Mapping[str, object]:
        """Parse a binary TOML file object into a mapping."""


def validate_project_structure(root: Path) -> bool:
    """Return whether all required repository paths exist under ``root``.

    Parameters
    ----------
    root:
        Repository root to validate.

    Returns
    -------
    bool
        ``True`` when every path in :data:`REQUIRED_PATHS` exists.
    """
    print("Validating Project Structure...")

    all_ok = True
    for relative_path in REQUIRED_PATHS:
        path = root / relative_path
        if path.exists():
            print(f"  [OK] Found {relative_path}")
        else:
            print(f"  [MISSING] {relative_path}")
            all_ok = False

    print(f"Structure Validation: {'PASSED' if all_ok else 'FAILED'}")
    return all_ok


def validate_pyproject(root: Path) -> bool:
    """Return whether ``pyproject.toml`` exists and parses as TOML.

    Parameters
    ----------
    root:
        Repository root containing ``pyproject.toml``.

    Returns
    -------
    bool
        ``True`` when the TOML parser is available and the file parses.
    """
    print("\nValidating pyproject.toml...")
    pyproject_path = root / "pyproject.toml"
    if not pyproject_path.exists():
        print("  [ERROR] pyproject.toml missing")
        return False

    toml = _load_toml_module()
    if toml is None:
        print("  [WARNING] tomli not installed, skipping strict parse check.")
        return True

    try:
        with pyproject_path.open("rb") as file_obj:
            data = toml.load(file_obj)
        project = _mapping_value(data.get("project", {}), "project")
    except Exception as exc:
        print(f"  [ERROR] Parsing failed: {exc}")
        return False

    print(f"  Name: {project.get('name')}")
    print(f"  Version: {project.get('version')}")
    print("pyproject.toml Validation: PASSED")
    return True


def main() -> int:
    """Run repository configuration validation and return a process exit code."""
    root = _resolve_project_root(Path.cwd())
    structure_ok = validate_project_structure(root)
    pyproject_ok = validate_pyproject(root)
    return 0 if structure_ok and pyproject_ok else 1


def _load_toml_module() -> _TomlLoader | None:
    """Return the available TOML parser for the running Python version."""
    if sys.version_info >= (3, 11):
        import tomllib

        return cast(_TomlLoader, tomllib)

    return _load_legacy_toml_module()  # pragma: no cover - Python <3.11 fallback.


def _load_legacy_toml_module() -> _TomlLoader | None:  # pragma: no cover - Python <3.11 fallback.
    """Return ``tomli`` for Python versions without ``tomllib``."""
    try:
        import tomli
    except ImportError:
        return None
    return cast(_TomlLoader, tomli)


def _mapping_value(value: object, field_name: str) -> Mapping[str, object]:
    """Return ``value`` as a mapping or raise for malformed TOML structure."""
    if isinstance(value, Mapping):
        return value
    raise TypeError(f"{field_name} must be a table")


def _resolve_project_root(cwd: Path) -> Path:
    """Return the repository root for commands run from root or ``scripts``."""
    if (cwd / "src").exists():
        return cwd
    if cwd.name == "scripts" and (cwd.parent / "src").exists():
        return cwd.parent
    return cwd


if __name__ == "__main__":  # pragma: no cover - subprocess entry point.
    raise SystemExit(main())
