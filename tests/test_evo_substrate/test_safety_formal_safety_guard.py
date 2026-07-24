# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestFormalSafetyGuard from former test_safety.py

"""Focused suite: TestFormalSafetyGuard from former test_safety.py."""

from __future__ import annotations

from tests.test_evo_substrate.safety_support import *  # noqa: F403


class TestFormalSafetyGuard:
    def test_passes_valid(self) -> None:
        guard = FormalSafetyGuard()
        g = Genome()
        g.compute_id()
        result = guard.check(g)
        assert result.passed
        assert result.violations == []

    def test_rejects_invalid(self) -> None:
        guard = FormalSafetyGuard(SafetyBounds(max_neurons=10))
        g = Genome()  # 16 neurons > 10
        g.compute_id()
        result = guard.check(g)
        assert not result.passed
        assert not result.neuron_count_ok
        assert guard.rejected == 1

    def test_rejection_rate(self) -> None:
        guard = FormalSafetyGuard(SafetyBounds(max_neurons=10))
        g = Genome()
        g.compute_id()
        guard.check(g)  # fails
        g2 = Genome()
        g2.topology.num_neurons = 5
        g2.compute_id()
        guard.check(g2)  # passes
        assert guard.rejection_rate == 0.5

    def test_rejects_connectivity_and_bitstream_violations(self) -> None:
        guard = FormalSafetyGuard(SafetyBounds(max_connectivity=0.2, max_bitstream=64))
        g = Genome()
        g.topology.connectivity = 0.8
        g.topology.bitstream_length = 128
        g.compute_id()

        result = guard.check(g)

        assert not result.passed
        assert not result.connectivity_ok
        assert not result.bitstream_ok
        assert "connectivity=0.8>0.2" in result.violations
        assert "bitstream=128>64" in result.violations
