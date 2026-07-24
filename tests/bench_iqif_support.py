# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_bench_iqif.py

from __future__ import annotations

"""Production-path tests for the controlled Wu et al. IQIF benchmark."""

import hashlib


import json


import os


from pathlib import Path


import platform


import shutil


import subprocess


import numpy as np


import numpy.typing as npt


import pytest


from benchmarks import bench_model_iqif as benchmark


from sc_neurocore.accel import iqif as backends


def _passing_safety() -> dict[str, object]:
    """Return one successful focused Rust-safety result."""
    return {"command": "focused", "passed": True, "returncode": 0, "output_tail": []}


def _measured_python(
    _backend: str,
) -> tuple[float, float, float, list[float], npt.NDArray[np.int64], int, int]:
    """Return one deterministic benchmark row for CLI gate tests."""
    return 1.0, 1.0, 1.0, [1.0], np.array([128], dtype=np.int64), 0, 128


__all__ = [
    "hashlib",
    "json",
    "os",
    "Path",
    "platform",
    "shutil",
    "subprocess",
    "np",
    "npt",
    "pytest",
    "benchmark",
    "backends",
    "_passing_safety",
    "_measured_python",
]
