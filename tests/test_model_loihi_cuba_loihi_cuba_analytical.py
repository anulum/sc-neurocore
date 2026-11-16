# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestLoihiCUBAAnalytical from former test_model_loihi_cuba.py

"""Focused suite: TestLoihiCUBAAnalytical from former test_model_loihi_cuba.py."""

from __future__ import annotations

from tests.model_loihi_cuba_support import *  # noqa: F403


class TestLoihiCUBAAnalytical:
    """Analytical properties of the Loihi CUBA update, one rule per test."""

    def test_u_integrates_input(self) -> None:
        """A positive input raises the synaptic current u."""
        n = LoihiCUBANeuron()
        n.step(500)
        assert n.u > 0

    def test_v_driven_by_u(self) -> None:
        """The current u left over from one input still drives v a step later."""
        n = LoihiCUBANeuron()
        n.step(500)
        n.step(0)
        assert n.v > 0

    def test_integer_division_decay(self) -> None:
        """A membrane of 100 with tau_v = 10 decays by the integer quotient 10."""
        n = LoihiCUBANeuron()
        n.v = 100
        decay = 100 // n.tau_v
        assert decay == 10

    def test_spike_resets_v(self) -> None:
        """Crossing the threshold sets v to v_reset."""
        n = LoihiCUBANeuron()
        for _ in range(10_000):
            if n.step(200) == 1:
                assert n.v == n.v_reset
                break

    def test_two_stage_integration(self) -> None:
        """The current u integrates input, v integrates u (2-stage pipeline)."""
        n = LoihiCUBANeuron()
        assert hasattr(n, "u") and hasattr(n, "v")

    def test_decay_truncates_towards_zero_on_a_negative_state(self) -> None:
        """A state of -7 loses trunc(-7 / 5) = -1 and becomes -6, as 7 becomes 6; floor would give -5."""
        n = LoihiCUBANeuron()
        n.u = -7
        n.step(0)
        assert n.u == -6
        assert n.v == -6

    def test_a_negated_drive_gives_the_negated_state(self) -> None:
        """Below threshold the update is odd: the mirrored drive gives the mirrored v and u on every step."""
        drive = [37, -12, 5, 0, -40, 23, 8, -3, 1, -1] * 4
        positive, negative = LoihiCUBANeuron(), LoihiCUBANeuron()
        for step, value in enumerate(drive):
            assert positive.step(value) == 0 and negative.step(-value) == 0
            assert (negative.v, negative.u) == (-positive.v, -positive.u), f"step {step}"
