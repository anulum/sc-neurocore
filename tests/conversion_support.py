# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_conversion.py

from __future__ import annotations

"""Tests for the ANN-to-SNN conversion engine."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
nn = torch.nn
from sc_neurocore.conversion.ann_to_snn import convert, _extract_layers
from sc_neurocore.conversion.qcfs import QCFSActivation


def _make_ann(in_f: int = 4, hidden: int = 8, out_f: int = 3) -> object:
    torch.manual_seed(42)
    return nn.Sequential(nn.Linear(in_f, hidden), nn.ReLU(), nn.Linear(hidden, out_f))


__all__ = [
    "np",
    "pytest",
    "torch",
    "nn",
    "convert",
    "_extract_layers",
    "QCFSActivation",
    "_make_ann",
]
