# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSTDPParams from former test_meta_plasticity.py

"""Focused suite: TestSTDPParams from former test_meta_plasticity.py."""

from __future__ import annotations

import numpy as np
from sc_neurocore.meta_plasticity.meta_plasticity import (
    STDPParams,
)


class TestSTDPParams:
    """Vector form of STDP parameters."""

    def test_to_vector(self) -> None:
        """The vector holds the five STDP parameters."""
        p = STDPParams()
        v = p.to_vector()
        assert len(v) == 5

    def test_from_vector_roundtrip(self) -> None:
        """Parameters survive a round trip through their vector."""
        p = STDPParams(tau_plus=15.0, a_plus=0.02, lr=0.005)
        v = p.to_vector()
        p2 = STDPParams.from_vector(v)
        assert abs(p2.tau_plus - 15.0) < 1e-6
        assert abs(p2.lr - 0.005) < 1e-6

    def test_from_vector_clamps(self) -> None:
        """Out-of-range vectors are clamped."""
        v = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        p = STDPParams.from_vector(v)
        assert p.tau_plus >= 1.0
        assert p.a_plus > 0
