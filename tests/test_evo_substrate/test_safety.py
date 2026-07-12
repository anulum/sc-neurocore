# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Evolutionary safety and resource-bound tests

"""Evolutionary safety and resource-bound tests."""

from __future__ import annotations

from sc_neurocore.evo_substrate.genome import Genome
from sc_neurocore.evo_substrate.safety import FormalSafetyGuard, ResourceBudget, SafetyBounds


class TestSafetyBounds:
    def test_clamp(self) -> None:
        sb = SafetyBounds(max_neurons=64)
        g = Genome()
        g.topology.num_neurons = 999
        sb.clamp(g)
        assert g.topology.num_neurons == 64

    def test_within_bounds(self) -> None:
        sb = SafetyBounds()
        g = Genome()
        assert sb.is_within_bounds(g)

    def test_out_of_bounds(self) -> None:
        sb = SafetyBounds(max_neurons=10)
        g = Genome()
        g.topology.num_neurons = 100
        assert not sb.is_within_bounds(g)


# ── Tile Deployment Tests (Gap 2) ────────────────────────────────────


class TestResourceBudget:
    def test_within_budget(self) -> None:
        rb = ResourceBudget(max_neurons=1024)
        g = Genome()
        ok, violations = rb.check(g)
        assert ok
        assert violations == []

    def test_exceeds_budget(self) -> None:
        rb = ResourceBudget(max_neurons=8)
        g = Genome()  # default 16 neurons
        ok, violations = rb.check(g)
        assert not ok
        assert len(violations) > 0

    def test_exceeds_area_budget(self) -> None:
        rb = ResourceBudget(max_neurons=1024, max_area_um2=100.0)
        g = Genome()
        g.topology.num_neurons = 32
        g.topology.bitstream_length = 64

        ok, violations = rb.check(g)

        assert not ok
        assert any(violation.startswith("area=") for violation in violations)


# ── Extinction Tests (Gap 8) ──────────────────────────────────────────


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


# ── Tournament Selection Tests (Gap 11) ──────────────────────────────
