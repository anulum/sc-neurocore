# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_crosstalk.py

from __future__ import annotations

"""Multi-angle tests for ``sc_neurocore.optics.photonic_emitter.CrosstalkModel``.

Covers:

- Coupled-mode transfer-matrix unitarity and energy conservation.
- :class:`WaveguidePair` physical invariants (larger gap ⇒ better isolation,
  longer coupler ⇒ worse isolation up to the first half-period).
- :meth:`CrosstalkModel.analyze_bank` uniform-bank analysis, near+far pair
  accounting, ``crosstalk_safe`` threshold.
- :meth:`CrosstalkModel.analyze_pairs` arbitrary-geometry O(N²) path.
- Rust-vs-Python backend parity (requires ``_HAS_RUST_PH``).
"""
import math
import numpy as np
import pytest
from sc_neurocore.optics.photonic_emitter import (
    CrosstalkModel,
    WaveguidePair,
)

__all__ = ['math', 'np', 'pytest', 'CrosstalkModel', 'WaveguidePair']
