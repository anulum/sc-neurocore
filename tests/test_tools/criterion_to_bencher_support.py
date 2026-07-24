# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_criterion_to_bencher.py

from __future__ import annotations

"""Contract for the Criterion→bencher converter used by the Performance Benchmarks gate.

The converter feeds ``benchmark-action/github-action-benchmark`` (``fail-on-alert`` at a
500% threshold). A single misread value poisons the gh-pages baseline and red-gates every
later push, so the parse — especially the unit-boundary straddle that previously under-read
``[999.50 µs 1.0001 ms 1.0050 ms]`` as ``1000 ns`` instead of ``1000100 ns`` — is pinned here.
"""
import importlib.util
from pathlib import Path
from types import ModuleType
import pytest


def _load_converter() -> ModuleType:
    """Load the converter script from ``.github`` (it is not an importable package)."""
    path = Path(__file__).resolve().parents[2] / ".github" / "criterion_to_bencher.py"
    spec = importlib.util.spec_from_file_location("criterion_to_bencher", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_CONVERTER = _load_converter()


def _one(line: str) -> int | None:
    """Convert a single Criterion line and return the parsed ns value (or None if skipped)."""
    out = list(_CONVERTER.convert(line))
    if not out:
        return None
    # "test <name> ... bench: <ns> ns/iter (+/- 0)"
    return int(out[0].split("bench:")[1].split("ns/iter")[0].strip())


__all__ = ["importlib", "Path", "ModuleType", "pytest", "_load_converter", "_CONVERTER", "_one"]
