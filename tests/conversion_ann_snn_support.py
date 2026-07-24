# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_conversion_ann_snn.py

from __future__ import annotations

import builtins
from collections.abc import Callable
import numpy as np
import numpy.typing as npt
import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn  # noqa: E402
from sc_neurocore.conversion.ann_to_snn import (  # noqa: E402
    ConvertedSNN,
    _extract_qcfs_layers,
    convert,
    replace_relu_with_qcfs,
)
from sc_neurocore.conversion.qcfs import QCFSActivation  # noqa: E402

__all__ = [
    "builtins",
    "Callable",
    "np",
    "npt",
    "pytest",
    "torch",
    "nn",
    "ConvertedSNN",
    "_extract_qcfs_layers",
    "convert",
    "replace_relu_with_qcfs",
    "QCFSActivation",
]
