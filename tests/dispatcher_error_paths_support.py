# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_dispatcher_error_paths.py

from __future__ import annotations

"""Covers the non-happy-path branches of the multi-backend dispatchers:

* missing shared library (``OSError`` → ``_HAS_*_* = False``)
* dispatcher called when library unavailable (``ImportError``)
* non-zero return code from the C shim (``RuntimeError``)
* Julia missing `.jl` kernel (``FileNotFoundError``)

These branches are never exercised by the parity/dynamics suites
because those run only when the shared library is present, so we
inject the failure paths with monkey-patching.
"""
import ctypes
from pathlib import Path
from types import ModuleType
import numpy as np
import pytest
from sc_neurocore.accel.go import jansen_rit as go_jansen
from sc_neurocore.accel.go import wilson_cowan as go_wilson
from sc_neurocore.accel.go import wong_wang as go_wong
from sc_neurocore.accel.mojo import jansen_rit as mojo_jansen
from sc_neurocore.accel.mojo import wilson_cowan as mojo_wilson
from sc_neurocore.accel.mojo import wong_wang as mojo_wong

CTYPES_DISPATCHERS = [
    (go_jansen, "simulate_jansen_rit", "Jansen–Rit"),
    (go_wilson, "simulate_wilson_cowan", "Wilson-Cowan"),
    (go_wong, "simulate_wong_wang", "Wong-Wang"),
    (mojo_jansen, "simulate_jansen_rit", "Jansen–Rit"),
    (mojo_wilson, "simulate_wilson_cowan", "Wilson-Cowan"),
    (mojo_wong, "simulate_wong_wang", "Wong-Wang"),
]

__all__ = [
    "ctypes",
    "Path",
    "ModuleType",
    "np",
    "pytest",
    "go_jansen",
    "go_wilson",
    "go_wong",
    "mojo_jansen",
    "mojo_wilson",
    "mojo_wong",
    "CTYPES_DISPATCHERS",
]
