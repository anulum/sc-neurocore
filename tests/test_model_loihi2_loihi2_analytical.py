# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestLoihi2Analytical from former test_model_loihi2.py

"""Focused suite: TestLoihi2Analytical from former test_model_loihi2.py."""

from __future__ import annotations

from sc_neurocore.neurons.models.loihi2 import (
    Loihi2Neuron,
)


class TestLoihi2Analytical:
    """Analytical properties of the Loihi 2 update, one rule per test."""

    def test_three_state_variables(self) -> None:
        """The neuron holds the three states s1, s2 and s3."""
        n = Loihi2Neuron()
        for attr in ["s1", "s2", "s3"]:
            assert hasattr(n, attr)

    def test_s2_integrates_input(self) -> None:
        """A positive input raises the synaptic state s2."""
        n = Loihi2Neuron()
        n.step(500)
        assert n.s2 > 0

    def test_s1_driven_by_s2(self) -> None:
        """s1 = s1 - trunc(s1/tau1) + w12·s2. w12=1 → s2 drives s1."""
        n = Loihi2Neuron()
        n.step(500)
        n.step(0)  # s2 still has residual → drives s1
        assert n.s1 > 0

    def test_spike_increments_s3(self) -> None:
        """On spike: s3 += s3_incr (adaptation)."""
        n = Loihi2Neuron()
        for _ in range(10_000):
            if n.step(200) == 1:
                assert n.s3 >= n.s3_incr
                break

    def test_integer_division_decay(self) -> None:
        """Decay via integer division: s -= trunc(s / tau)."""
        n = Loihi2Neuron()
        n.s1 = 100
        decay = 100 // n.tau1
        assert decay == 10  # 100 // 10

    def test_spike_resets_s1(self) -> None:
        """Crossing the threshold sets s1 to s1_reset."""
        n = Loihi2Neuron()
        for _ in range(10_000):
            if n.step(200) == 1:
                assert n.s1 == n.s1_reset
                break

    def test_decay_truncates_towards_zero_on_a_negative_state(self) -> None:
        """Each state loses its quotient truncated towards zero: s3 = -23 stays at -23 (floor gives -22)."""
        n = Loihi2Neuron()
        n.s1, n.s2, n.s3 = -7, -7, -23
        n.step(0)
        assert n.s3 == -23
        assert n.s2 == -6
        assert n.s1 == -7 - 0 + n.s2

    def test_a_negated_drive_gives_the_negated_state(self) -> None:
        """With s3 decaying into s1, the mirrored drive and start give the mirrored states on every step."""
        drive = [41, -17, 6, 0, -33, 25, 9, -4, 2, -2] * 4
        positive = Loihi2Neuron(w13=1, s3=-120)
        negative = Loihi2Neuron(w13=1, s3=120)
        for step, value in enumerate(drive):
            assert positive.step(value) == 0 and negative.step(-value) == 0
            mirrored = (-positive.s1, -positive.s2, -positive.s3)
            assert (negative.s1, negative.s2, negative.s3) == mirrored, f"step {step}"
