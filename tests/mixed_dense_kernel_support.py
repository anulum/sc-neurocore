# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_mixed_dense_kernel.py

from __future__ import annotations

"""Algorithm, validation, saturation and dispatch tests for the mixed-dense kernel.

Cross-language bit-exact parity is exercised separately in
``tests/test_mixed_dense_kernel_parity.py``.
"""
import numpy as np
import numpy.testing as npt
import pytest
from sc_neurocore.compiler import mixed_dense_kernel as kernel
from sc_neurocore.compiler.mixed_dense_kernel import (
    MixedDenseBatchResult,
    available_backends,
    mixed_dense_forward_batch,
    mixed_dense_forward_batch_q88_q1616,
)

__all__ = [
    "np",
    "npt",
    "pytest",
    "kernel",
    "MixedDenseBatchResult",
    "available_backends",
    "mixed_dense_forward_batch",
    "mixed_dense_forward_batch_q88_q1616",
]
