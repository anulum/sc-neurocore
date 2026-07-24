# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_qat_observers.py

from __future__ import annotations

"""Tests for per-tensor and per-channel quantisation observers."""
import pytest

torch = pytest.importorskip("torch")
from sc_neurocore.qat.observers import (
    MinMaxObserver,
    PerChannelMinMaxObserver,
    _quant_bounds,
    fake_quantize,
)

__all__ = [
    "pytest",
    "torch",
    "MinMaxObserver",
    "PerChannelMinMaxObserver",
    "_quant_bounds",
    "fake_quantize",
]
