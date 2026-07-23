# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_darts_sc_nas.py

from __future__ import annotations

from typing import Protocol
import unittest
import pytest
torch = pytest.importorskip("torch", reason="torch not installed; DARTS tests require it")
from sc_neurocore.nas.darts_sc_nas import (  # noqa: E402
    BitstreamCandidate,
    SCMixedOp,
    SCNASNetwork,
)
class _Backwardable(Protocol):
    def backward(self) -> None: ...
def _backward(value: _Backwardable) -> None:
    value.backward()

__all__ = ['Protocol', 'unittest', 'pytest', 'torch', 'BitstreamCandidate', 'SCMixedOp', 'SCNASNetwork', '_Backwardable', '_backward']
