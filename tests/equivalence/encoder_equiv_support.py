# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_encoder_equiv.py

from __future__ import annotations

"""Strict blueprint semantics tests for LFSR + bitstream encoder.

The encoder uses compare-before-advance semantics matching the Verilog RTL
(sc_bitstream_encoder.v): non-blocking assignments mean `bit_out` reads
the LFSR state *before* the advance that happens in the same clock edge.
"""
import pytest
pytest.importorskip("sc_neurocore_engine", reason="Rust engine not built", exc_type=ImportError)
from sc_neurocore_engine import BitstreamEncoder, Lfsr16
def _lfsr_step(reg: int) -> int:
    feedback = ((reg >> 15) ^ (reg >> 13) ^ (reg >> 12) ^ (reg >> 10)) & 1
    return ((reg << 1) & 0xFFFF) | feedback

__all__ = ['pytest', 'BitstreamEncoder', 'Lfsr16', '_lfsr_step']
