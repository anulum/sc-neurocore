# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_hls_export.py

from __future__ import annotations

"""Tests for the ap_fixed HLS C++ exporter, including a g++ compile check."""
import shutil
import subprocess
from pathlib import Path
import pytest
from sc_neurocore.compiler.intelligence.hls_export import generate_hls_cpp

_AP_FIXED_STUB = """#pragma once
#include <cmath>
template <int W, int I> using ap_fixed = double;
"""
_HLS_MATH_STUB = """#pragma once
#include <cmath>
namespace hls {
inline double exp(double x) { return std::exp(x); }
inline double log(double x) { return std::log(x); }
inline double sqrt(double x) { return std::sqrt(x); }
inline double cbrt(double x) { return std::cbrt(x); }
inline double tanh(double x) { return std::tanh(x); }
inline double cosh(double x) { return std::cosh(x); }
inline double sin(double x) { return std::sin(x); }
inline double cos(double x) { return std::cos(x); }
inline double abs(double x) { return std::fabs(x); }
}
"""

__all__ = [
    "shutil",
    "subprocess",
    "Path",
    "pytest",
    "generate_hls_cpp",
    "_AP_FIXED_STUB",
    "_HLS_MATH_STUB",
]
