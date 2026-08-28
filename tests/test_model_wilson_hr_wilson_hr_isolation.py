# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Wilson-HR isolated state behaviour

"""Verify the source Wilson-HR model in an isolated Python runtime."""

from __future__ import annotations

from tests.model_wilson_hr_support import *


class TestWilsonHRIsolation:
    def test_defaults(self) -> None:
        n = WilsonHRNeuron()
        assert n.v == -0.7
        assert n.r == 0.085
        assert n.capacitance == 0.8
        assert n.tau_r == 1.9
        assert n.v_peak == 0.0
        assert n.dt == 0.05

    def test_step_returns_binary(self) -> None:
        assert WilsonHRNeuron().step(0.0) in (0, 1)

    def test_two_variables_evolve(self) -> None:
        n = WilsonHRNeuron()
        v0, r0 = n.v, n.r
        for _ in range(100):
            n.step(0.3)
        assert n.v != v0 and n.r != r0

    def test_state_finite(self) -> None:
        n = WilsonHRNeuron()
        for _ in range(50_000):
            n.step(0.3)
        assert np.isfinite(n.v) and np.isfinite(n.r)

    def test_reset(self) -> None:
        n = WilsonHRNeuron()
        for _ in range(100):
            n.step(0.3)
        n.reset()
        assert n.v == -0.7 and n.r == 0.085

    def test_spike_preserves_continuous_voltage(self) -> None:
        n = WilsonHRNeuron()
        for _ in range(50_000):
            if n.step(0.1) == 1:
                assert n.v >= n.v_peak
                assert n.v != -0.7
                break
        else:
            pytest.fail("source Wilson-HR trajectory did not cross the observation threshold")
