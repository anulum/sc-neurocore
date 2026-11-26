# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestEncoderBlueprintSemantics from former test_encoder_equiv.py

"""Focused suite: TestEncoderBlueprintSemantics from former test_encoder_equiv.py."""

from __future__ import annotations

from tests.equivalence.encoder_equiv_support import (
    _lfsr_step,
)
from tests.engine_requirement import require_engine

require_engine()
from sc_neurocore_engine import (
    BitstreamEncoder,
)


class TestEncoderBlueprintSemantics:
    """Hold the engine bitstream encoder to the blueprint's compare-then-advance order."""

    def test_compare_before_advance_order(self) -> None:
        """Encoder reads LFSR state, compares, then advances (Verilog RTL match)."""
        encoder = BitstreamEncoder(data_width=16, seed=0xACE1)
        x_value = 0xACE1
        reg = 0xACE1

        bits = []
        for _ in range(8):
            expected = 1 if reg < x_value else 0
            bits.append(encoder.step(x_value))
            assert bits[-1] == expected
            reg = _lfsr_step(reg)

        # 0xACE1 < 0xACE1 is false → first bit = 0
        assert bits[0] == 0, "Compare-before-advance: reg == x_value → bit = 0"
