# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Compiler public surface policy tests

"""Regression tests for the documented compiler package surface."""

from __future__ import annotations

from pathlib import Path

import sc_neurocore.compiler as compiler


REPO_ROOT = Path(__file__).resolve().parents[1]
COMPILER_DIR = REPO_ROOT / "src" / "sc_neurocore" / "compiler"
SURFACE_DOC = REPO_ROOT / "docs" / "api" / "compiler_surface.md"

PUBLIC_FACADE_MODULES = {
    "adaptive_precision",
    "equation_compiler",
    "ir_type_checker",
    "live_control",
    "mlir_emitter",
    "pipeline",
    "quantizer",
}

ALLOWED_STATUSES = {
    "public facade",
    "direct public module",
    "compatibility facade",
    "internal build tool",
}


def _root_compiler_modules() -> set[str]:
    """Return root-level compiler modules that must have an explicit surface decision."""
    return {path.stem for path in COMPILER_DIR.glob("*.py") if path.name != "__init__.py"}


def _documented_surface_rows() -> dict[str, str]:
    """Parse the compiler surface table from the public documentation page."""
    rows: dict[str, str] = {}
    for line in SURFACE_DOC.read_text(encoding="utf-8").splitlines():
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) < 3 or cells[0] in {"Module", "---"}:
            continue
        module = cells[0].strip("`")
        status = cells[1]
        if module:
            rows[module] = status
    return rows


def test_compiler_root_modules_are_classified_in_public_docs() -> None:
    """Every root compiler module must be public, compatibility, or explicitly internal."""
    documented = _documented_surface_rows()
    modules = _root_compiler_modules()

    assert documented.keys() == modules
    assert set(documented.values()).issubset(ALLOWED_STATUSES)


def test_compiler_facade_module_decisions_match_package_exports() -> None:
    """The documented facade modules must match the package-level import contract."""
    documented = _documented_surface_rows()
    facade_modules = {module for module, status in documented.items() if status == "public facade"}

    assert facade_modules == PUBLIC_FACADE_MODULES
    assert compiler.__all__
    for name in compiler.__all__:
        assert hasattr(compiler, name), f"compiler.__all__ exports missing symbol {name}"
