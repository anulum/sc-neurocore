# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_static_analysis.py

from __future__ import annotations

"""Tests for guard-bit computation, formal overflow proofs, and SVA generation."""
from typing import Any
import pytest
from sc_neurocore.compiler.static_analysis import (
    FixedPointEnvelopeProof,
    Interval,
    compute_guard_bits,
    compute_guard_bits_multi,
    generate_sva,
    prove_fixed_point_envelope,
    prove_no_overflow,
)

__all__ = ['Any', 'pytest', 'FixedPointEnvelopeProof', 'Interval', 'compute_guard_bits', 'compute_guard_bits_multi', 'generate_sva', 'prove_fixed_point_envelope', 'prove_no_overflow']
