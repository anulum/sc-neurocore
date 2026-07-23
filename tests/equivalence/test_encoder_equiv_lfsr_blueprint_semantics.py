# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestLFSRBlueprintSemantics from former test_encoder_equiv.py

"""Focused suite: TestLFSRBlueprintSemantics from former test_encoder_equiv.py."""

from __future__ import annotations

import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parent))
from encoder_equiv_support import *  # noqa: F403

class TestLFSRBlueprintSemantics:
    def test_full_cycle_matches_blueprint_formula(self):
        reg = 0xACE1
        v3 = Lfsr16(seed=reg)

        for i in range(65535):
            reg = _lfsr_step(reg)
            assert v3.step() == reg, f"LFSR divergence at step {i}"

    def test_multiple_seeds(self):
        for seed in [0xACE1, 0xBEEF, 0xACE1 + 7, 0xBEEF + 13]:
            reg = seed
            v3 = Lfsr16(seed=seed)
            for i in range(1000):
                reg = _lfsr_step(reg)
                assert v3.step() == reg, f"Seed {seed:#06x} diverged at {i}"
