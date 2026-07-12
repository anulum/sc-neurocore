# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Evolutionary software and FPGA fitness tests

"""Evolutionary software and FPGA fitness tests."""

from __future__ import annotations

from sc_neurocore.evo_substrate.fitness import (
    FitnessEvaluator,
    FitnessResult,
    HWFitnessCollector,
    HWFitnessReport,
)
from sc_neurocore.evo_substrate.genome import Genome


class TestFitnessEvaluator:
    def test_evaluate(self) -> None:
        ev = FitnessEvaluator()
        g = Genome()
        g.compute_id()
        result = ev.evaluate(g, {"accuracy": 0.9})
        assert result.accuracy == 0.9
        assert result.composite > 0

    def test_energy_penalty(self) -> None:
        ev = FitnessEvaluator()
        small = Genome()
        small.topology.num_neurons = 16
        small.topology.bitstream_length = 128
        small.compute_id()
        big = Genome()
        big.topology.num_neurons = 1024
        big.topology.bitstream_length = 1024
        big.compute_id()
        r_small = ev.evaluate(small, {"accuracy": 0.8})
        r_big = ev.evaluate(big, {"accuracy": 0.8})
        assert r_small.energy_score > r_big.energy_score

    def test_composite_weights(self) -> None:
        r = FitnessResult("test", accuracy=1.0, energy_score=0.0, latency_score=0.0)
        c = r.compute_composite(w_acc=1.0, w_energy=0.0, w_latency=0.0)
        assert c == 1.0


# ── ReplicationEngine Tests ─────────────────────────────────────────


class TestHWFitness:
    def test_report(self) -> None:
        r = HWFitnessReport("test_id", fpga_accuracy=0.9, fmax_mhz=200.0)
        assert r.hw_composite > 0

    def test_collector(self) -> None:
        col = HWFitnessCollector()
        col.submit(HWFitnessReport("g1", fpga_accuracy=0.8))
        assert col.total_reports == 1
        assert col.get("g1") is not None
        assert col.get("nonexistent") is None


# ── Evo Statistics Tests (Gap 18) ─────────────────────────────────────
