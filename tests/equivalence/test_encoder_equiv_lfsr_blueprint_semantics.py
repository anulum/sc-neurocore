# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestLFSRBlueprintSemantics from former test_encoder_equiv.py

"""Focused suite: TestLFSRBlueprintSemantics from former test_encoder_equiv.py."""

from __future__ import annotations

from tests.equivalence.encoder_equiv_support import (
    _lfsr_step,
)
from tests.engine_requirement import require_engine

require_engine()
from sc_neurocore_engine import (
    Lfsr16,
)


class TestLFSRBlueprintSemantics:
    """Hold the engine LFSR to the blueprint's step formula."""

    def test_full_cycle_matches_blueprint_formula(self) -> None:
        """Over its full 65,535-step cycle the engine LFSR matches the blueprint step."""
        reg = 0xACE1
        v3 = Lfsr16(seed=reg)

        for i in range(65535):
            reg = _lfsr_step(reg)
            assert v3.step() == reg, f"LFSR divergence at step {i}"

    def test_multiple_seeds(self) -> None:
        """From four seeds the engine LFSR matches the blueprint step for 1,000 steps."""
        for seed in [0xACE1, 0xBEEF, 0xACE1 + 7, 0xBEEF + 13]:
            reg = seed
            v3 = Lfsr16(seed=seed)
            for i in range(1000):
                reg = _lfsr_step(reg)
                assert v3.step() == reg, f"Seed {seed:#06x} diverged at {i}"
