# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestHybridFisherPosnerLIFNeuron from former test_fisher_posner.py

"""Focused suite: TestHybridFisherPosnerLIFNeuron from former test_fisher_posner.py."""

from __future__ import annotations

import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parent))
from fisher_posner_support import *  # noqa: F403

class TestHybridFisherPosnerLIFNeuron:
    def test_wrapper_step(self) -> None:
        """Wrapper step should return int (0 or 1)."""
        HybridFisherPosnerLIFNeuron._reset_pools()
        n = HybridFisherPosnerLIFNeuron(n_sites=8)
        result = n.step(0.0)
        assert result in (0, 1)

    def test_wrapper_v_property(self) -> None:
        HybridFisherPosnerLIFNeuron._reset_pools()
        n = HybridFisherPosnerLIFNeuron(n_sites=4)
        assert isinstance(n.v, float)
