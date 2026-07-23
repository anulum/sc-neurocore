# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_dcls_tent_kernel.py

from __future__ import annotations

"""Algorithm, validation, saturation and dispatch tests for the tent kernel.

The cross-language bit-exact parity contract is exercised separately in
``tests/test_dcls_tent_kernel_parity.py``; this module pins the pure-Python
reference behaviour and the backend dispatch logic.
"""
import numpy as np
import numpy.testing as npt
import pytest
from sc_neurocore.scpn import dcls_tent_kernel as kernel
from sc_neurocore.scpn.dcls_tent_kernel import (
    DclsBatchResult,
    DclsForwardResult,
    available_backends,
    dcls_max_forward_batch,
    dcls_max_forward_batch_q88,
    dcls_max_forward_q88,
    tent_gate_q88,
)

__all__ = ['np', 'npt', 'pytest', 'kernel', 'DclsBatchResult', 'DclsForwardResult', 'available_backends', 'dcls_max_forward_batch', 'dcls_max_forward_batch_q88', 'dcls_max_forward_q88', 'tent_gate_q88']
