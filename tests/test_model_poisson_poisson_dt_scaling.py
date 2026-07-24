# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestPoissonDtScaling from former test_model_poisson.py

"""Focused suite: TestPoissonDtScaling from former test_model_poisson.py."""

from __future__ import annotations

from tests.model_poisson_support import *  # noqa: F403


class TestPoissonDtScaling:
    def test_dt_scales_probability(self) -> None:
        """P(spike) = λ·dt/1000. Doubling dt doubles spike probability."""
        N = 100000
        n1 = PoissonNeuron(rate_hz=100.0, dt_ms=0.5)
        n2 = PoissonNeuron(rate_hz=100.0, dt_ms=1.0)
        s1 = sum(n1.step() for _ in range(N))
        s2 = sum(n2.step() for _ in range(N))
        ratio = s2 / s1 if s1 > 0 else 0
        assert 1.5 < ratio < 2.5, f"ratio = {ratio:.2f}, expected ≈2.0"

    def test_small_dt_rare_spikes(self) -> None:
        """Very small dt → very rare spikes."""
        n = PoissonNeuron(rate_hz=100.0, dt_ms=0.01)
        # P = 100 * 0.01 / 1000 = 0.001
        spikes = sum(n.step() for _ in range(100000))
        assert spikes < 500  # expected ~100
