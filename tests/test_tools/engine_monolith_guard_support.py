# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_engine_monolith_guard.py

from __future__ import annotations

"""Support extracted from test_engine_monolith_guard.py."""

import importlib.util


import runpy


import sys


from pathlib import Path


from typing import Any


import pytest


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_tool() -> Any:
    tool_path = _repo_root() / "tools" / "engine_monolith_guard.py"
    spec = importlib.util.spec_from_file_location("engine_monolith_guard", tool_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_fake_target(
    root: Path,
    *,
    lines: int,
    pyfunctions: int,
    rel_path: str = "engine/src/lib.rs",
) -> None:
    """Write one fake Rust target with exact line and pyfunction counts."""

    assert lines >= 2 * pyfunctions, "need two lines per pyfunction (attr + fn)"
    body: list[str] = []
    for index in range(pyfunctions):
        body.append("#[pyfunction]")
        body.append(f"fn f{index}() {{}}")
    while len(body) < lines:
        body.append("// pad")
    target = root / rel_path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(body[:lines]) + "\n", encoding="utf-8")


def _write_ceiling(path: Path, rel: str, *, max_lines: int, max_pyfunctions: int) -> None:
    path.write_text(
        "schema_version = 1\n"
        f'[targets."{rel}"]\n'
        f"max_lines = {max_lines}\n"
        f"max_pyfunctions = {max_pyfunctions}\n",
        encoding="utf-8",
    )



__all__ = ['importlib', 'runpy', 'sys', 'Path', 'Any', 'pytest', '_repo_root', '_load_tool', '_write_fake_target', '_write_ceiling']
