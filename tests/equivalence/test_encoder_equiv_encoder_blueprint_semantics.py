# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestEncoderBlueprintSemantics from former test_encoder_equiv.py

"""Focused suite: TestEncoderBlueprintSemantics from former test_encoder_equiv.py."""

from __future__ import annotations

import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parent))
from encoder_equiv_support import *  # noqa: F403

class TestEncoderBlueprintSemantics:
    def test_compare_before_advance_order(self):
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
