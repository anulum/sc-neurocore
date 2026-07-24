# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSRM0EtaKernel from former test_model_srm0.py

"""Focused suite: TestSRM0EtaKernel from former test_model_srm0.py."""

from __future__ import annotations

from tests.model_srm0_support import *  # noqa: F403


class TestSRM0EtaKernel:
    def test_eta_set_on_spike(self) -> None:
        """After spike: eta = -eta_reset (negative = afterhyperpolarisation)."""
        n = SRM0Neuron()
        for _ in range(10000):
            if n.step(5.0) == 1:
                assert n._eta == -n.eta_reset
                break
        else:
            raise AssertionError("No spike")

    def test_eta_decays_exponentially(self) -> None:
        """eta *= exp(-dt/tau_eta) each step."""
        n = SRM0Neuron()
        n._eta = -5.0
        eta0 = n._eta
        n.step(0.0)
        expected = eta0 * math.exp(-n.dt / n.tau_eta)
        assert abs(n._eta - expected) < 1e-10

    def test_eta_provides_afterhyperpolarisation(self) -> None:
        """Negative eta shifts effective rest downward, slowing next spike."""
        n = SRM0Neuron()
        for _ in range(10000):
            if n.step(5.0) == 1:
                # eta is now -5.0 → effective_rest = v_rest + (-5) = -5
                # V was just reset to 0. Next step: V will be pulled down.
                n.step(5.0)
                # V should be below what it would be without eta
                assert n.v < 0.5  # much less than 5.0 * dt/tau_m
                break

    def test_eta_zero_long_after_spike(self) -> None:
        """eta decays to ~0 long after spike."""
        n = SRM0Neuron()
        n._eta = -5.0
        for _ in range(500):
            n.step(0.0)
        assert abs(n._eta) < 0.001
