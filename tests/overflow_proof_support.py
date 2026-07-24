# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_overflow_proof.py

from __future__ import annotations

"""Tests for formal overflow proofs via interval arithmetic."""
from sc_neurocore.compiler.overflow_proof import (
    FixedPointEnvelopeProof,
    Interval,
    prove_fixed_point_envelope,
    prove_no_overflow,
)

__all__ = ["FixedPointEnvelopeProof", "Interval", "prove_fixed_point_envelope", "prove_no_overflow"]
