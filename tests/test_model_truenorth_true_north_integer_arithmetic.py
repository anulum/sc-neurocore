# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestTrueNorthIntegerArithmetic from former test_model_truenorth.py

"""Focused suite: TestTrueNorthIntegerArithmetic from former test_model_truenorth.py."""

from __future__ import annotations

from tests.model_truenorth_support import *  # noqa: F403


class TestTrueNorthIntegerArithmetic:
    """Core: v += input - leak. All integer. No floating point."""

    def test_voltage_accumulation_exact(self):
        """v = sum(input) - steps*leak when no spikes occur."""
        n = TrueNorthNeuron(leak=0, threshold=1000)
        for _ in range(10):
            n.step(7)
        assert n.v == 70  # 10 * 7 = 70

    def test_leak_subtracted_each_step(self):
        """v += input - leak. With leak=5, input=10: net +5 per step."""
        n = TrueNorthNeuron(leak=5, threshold=1000)
        for _ in range(10):
            n.step(10)
        assert n.v == 50  # 10 * (10 - 5) = 50

    def test_leak_exceeds_input_no_spikes(self):
        """When leak > input, v decreases → never reaches threshold."""
        n = TrueNorthNeuron(leak=50)
        spikes = sum(n.step(20) for _ in range(1000))
        assert spikes == 0

    def test_spike_rate_exact(self):
        """With leak=0, input=I: spike every ceil(threshold/I) steps.

        I=10, θ=100: spike every 10 steps → 100 spikes/1000.
        """
        n = TrueNorthNeuron(leak=0, threshold=100)
        outputs = [n.step(10) for _ in range(1000)]
        assert outputs.count(1) == 100

    def test_spike_resets_to_v_reset(self):
        n = TrueNorthNeuron(threshold=100, v_reset=0)
        for _ in range(1000):
            s = n.step(50)
            if s == 1:
                assert n.v == 0
                break
        else:
            pytest.fail("No spike")

    def test_negative_saturation(self):
        """v < -threshold → reset to v_reset (prevents unbounded negative)."""
        n = TrueNorthNeuron(threshold=100)
        n.step(-200)  # v = -200 < -100
        assert n.v == 0  # reset

    def test_negative_saturation_boundary(self):
        """v = -100: NOT reset (condition is v < -threshold, not ≤)."""
        n = TrueNorthNeuron(threshold=100)
        n.v = -100
        n.step(0)  # v stays -100, check: -100 < -100 is False
        assert n.v == -100

    def test_custom_v_reset(self):
        n = TrueNorthNeuron(threshold=100, v_reset=10)
        for _ in range(1000):
            if n.step(50) == 1:
                assert n.v == 10
                break
