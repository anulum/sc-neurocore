# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_gpu_backend.py

from __future__ import annotations

"""
Tests for sc_neurocore.accel.gpu_backend.

All tests run on both CuPy (GPU) and NumPy (CPU) — the backend is
selected automatically at import time.
"""
import numpy as np
import pytest
import sc_neurocore.accel.gpu_backend as gb
from sc_neurocore.accel.gpu_backend import (
    xp,
    to_device,
    to_host,
    gpu_pack_bitstream,
    gpu_vec_and,
    gpu_popcount,
    gpu_vec_mac,
)

__all__ = [
    "np",
    "pytest",
    "gb",
    "xp",
    "to_device",
    "to_host",
    "gpu_pack_bitstream",
    "gpu_vec_and",
    "gpu_popcount",
    "gpu_vec_mac",
]
