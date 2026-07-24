# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_qat_lsq.py

from __future__ import annotations

"""Tests for the LSQ weight quantiser (Esser et al. 2020)."""
import math
import pytest

torch = pytest.importorskip("torch")
from sc_neurocore.qat.lsq import LSQLinear, LSQQuantizer, _LSQQuantize, _sum_to

__all__ = ["math", "pytest", "torch", "LSQLinear", "LSQQuantizer", "_LSQQuantize", "_sum_to"]
