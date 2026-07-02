# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — SC-NAS Engine Tests


import importlib
import sys
from collections.abc import Callable
from types import ModuleType

import pytest

from sc_neurocore.nas import sc_nas_engine as nas_module
from sc_neurocore.nas.sc_nas_engine import (
    DecorrelationStrategy,
    EvolutionaryNAS,
    FPGAResourceBudget,
    LayerConfig,
    NASObjective,
    NASReport,
    NASVerilogEmitter,
    NeuronType,
    SCCandidate,
    SCFitnessEvaluator,
    pareto_front,
    run_nas,
)
from sc_neurocore.optimizer.sc_optimizer import LayerProfile
from sc_neurocore.optimizer.surrogate_sc_optimizer import (
    SurrogateLayerConfig,
    SurrogateOptimizerReport,
)

RustTournament = Callable[[list[float], int, int, int], list[int]]


def _surrogate_cfg(length: int) -> SurrogateLayerConfig:
    return SurrogateLayerConfig(
        bitstream_length=length,
        decorrelator="LFSR",
        mode="SC",
        precision_bits=8,
        lfsr_polynomial="x16+x14+x13+x11+1",
        luts_used=100,
        power_used=1.0,
        latency_cycles=length,
        accuracy_score=0.99,
        utility_score=0.95,
    )


class _FakeSurrogateOptimiser:
    def __init__(self) -> None:
        self.calls: list[list[LayerProfile]] = []

    def optimise(self, network: list[LayerProfile]) -> SurrogateOptimizerReport:
        self.calls.append(network)
        return SurrogateOptimizerReport(
            config={
                profile.id: _surrogate_cfg(128 + index * 128)
                for index, profile in enumerate(network)
            },
            total_luts=100 * len(network),
            total_power_mw=float(len(network)),
            total_latency_cycles=256,
            mean_accuracy=0.99,
            training_points=16,
            target_name="unit-fpga",
        )


# ── LayerConfig Tests ────────────────────────────────────────────────


class TestLayerConfig:
    def test_lut_cost_increases_with_neurons(self) -> None:
        a = LayerConfig(32, NeuronType.LIF, 256, DecorrelationStrategy.LFSR)
        b = LayerConfig(64, NeuronType.LIF, 256, DecorrelationStrategy.LFSR)
        assert b.lut_cost > a.lut_cost

    def test_ff_cost_increases_with_bitstream_length(self) -> None:
        a = LayerConfig(32, NeuronType.LIF, 128, DecorrelationStrategy.LFSR)
        b = LayerConfig(32, NeuronType.LIF, 1024, DecorrelationStrategy.LFSR)
        assert b.ff_cost > a.ff_cost

    def test_power_scales_with_length(self) -> None:
        a = LayerConfig(32, NeuronType.LIF, 256, DecorrelationStrategy.LFSR)
        b = LayerConfig(32, NeuronType.LIF, 512, DecorrelationStrategy.LFSR)
        assert b.power_cost > a.power_cost

    def test_hh_costlier_than_lif(self) -> None:
        lif = LayerConfig(32, NeuronType.LIF, 256, DecorrelationStrategy.LFSR)
        hh = LayerConfig(32, NeuronType.HH, 256, DecorrelationStrategy.LFSR)
        assert hh.lut_cost > lif.lut_cost
        assert hh.power_cost > lif.power_cost

    def test_neuron_type_ordering(self) -> None:
        costs = {}
        for nt in NeuronType:
            l = LayerConfig(32, nt, 256, DecorrelationStrategy.LFSR)
            costs[nt] = l.lut_cost
        assert costs[NeuronType.LIF] < costs[NeuronType.IZHIKEVICH]
        assert costs[NeuronType.IZHIKEVICH] < costs[NeuronType.ADEX]
        assert costs[NeuronType.ADEX] < costs[NeuronType.HH]

    def test_dsp_cost(self) -> None:
        lif = LayerConfig(32, NeuronType.LIF, 256, DecorrelationStrategy.LFSR)
        hh = LayerConfig(32, NeuronType.HH, 256, DecorrelationStrategy.LFSR)
        assert lif.dsp_cost == 0
        assert hh.dsp_cost == 32 * 4

    def test_bram_cost(self) -> None:
        l = LayerConfig(64, NeuronType.LIF, 1024, DecorrelationStrategy.LFSR)
        expected = (64 * 1024) / 8192.0
        assert abs(l.bram_cost_kb - expected) < 0.01


# ── SCCandidate Tests ────────────────────────────────────────────────


class TestSCCandidate:
    def test_evaluate_resources(self) -> None:
        c = SCCandidate(
            layers=[
                LayerConfig(32, NeuronType.LIF, 256, DecorrelationStrategy.LFSR),
                LayerConfig(64, NeuronType.ADEX, 512, DecorrelationStrategy.SOBOL),
            ]
        )
        c.evaluate_resources()
        assert c.total_luts > 0
        assert c.total_ffs > 0
        assert c.total_dsp > 0
        assert c.total_bram_kb > 0
        assert c.total_power_mw > 0

    def test_meets_budget_within_limits(self) -> None:
        c = SCCandidate(
            layers=[
                LayerConfig(16, NeuronType.LIF, 64, DecorrelationStrategy.LFSR),
            ]
        )
        budget = FPGAResourceBudget(max_luts=1_000_000)
        assert c.meets_budget(budget)

    def test_exceeds_budget(self) -> None:
        c = SCCandidate(
            layers=[
                LayerConfig(256, NeuronType.HH, 4096, DecorrelationStrategy.HYBRID),
            ]
            * 10
        )
        budget = FPGAResourceBudget(max_luts=100)
        assert not c.meets_budget(budget)

    def test_fingerprint_deterministic(self) -> None:
        c = SCCandidate(
            layers=[
                LayerConfig(32, NeuronType.LIF, 256, DecorrelationStrategy.LFSR),
            ]
        )
        assert c.fingerprint == c.fingerprint

    def test_fingerprint_differs_for_different_configs(self) -> None:
        a = SCCandidate(
            layers=[
                LayerConfig(32, NeuronType.LIF, 256, DecorrelationStrategy.LFSR),
            ]
        )
        b = SCCandidate(
            layers=[
                LayerConfig(64, NeuronType.ADEX, 512, DecorrelationStrategy.SOBOL),
            ]
        )
        assert a.fingerprint != b.fingerprint

    def test_dsp_budget_check(self) -> None:
        c = SCCandidate(
            layers=[
                LayerConfig(256, NeuronType.HH, 256, DecorrelationStrategy.LFSR),
            ]
        )
        budget = FPGAResourceBudget(max_dsp=10)
        assert not c.meets_budget(budget)


# ── Fitness Evaluator Tests ──────────────────────────────────────────


class TestSCFitnessEvaluator:
    def test_longer_bitstreams_higher_accuracy(self) -> None:
        ev = SCFitnessEvaluator(seed=42)
        short = SCCandidate(
            layers=[
                LayerConfig(32, NeuronType.LIF, 64, DecorrelationStrategy.LFSR),
            ]
        )
        long = SCCandidate(
            layers=[
                LayerConfig(32, NeuronType.LIF, 4096, DecorrelationStrategy.LFSR),
            ]
        )
        acc_short = ev.evaluate(short)
        acc_long = ev.evaluate(long)
        assert acc_long > acc_short

    def test_sobol_decorrelation_bonus(self) -> None:
        ev = SCFitnessEvaluator(seed=42)
        lfsr = SCCandidate(
            layers=[
                LayerConfig(32, NeuronType.LIF, 256, DecorrelationStrategy.LFSR),
            ]
        )
        sobol = SCCandidate(
            layers=[
                LayerConfig(32, NeuronType.LIF, 256, DecorrelationStrategy.SOBOL),
            ]
        )
        assert ev.evaluate(sobol) > ev.evaluate(lfsr)

    def test_accuracy_bounded_0_1(self) -> None:
        ev = SCFitnessEvaluator(seed=42)
        c = SCCandidate(
            layers=[
                LayerConfig(32, NeuronType.LIF, 128, DecorrelationStrategy.LFSR),
            ]
        )
        acc = ev.evaluate(c)
        assert 0.0 <= acc <= 1.0


# ── Pareto Front Tests ───────────────────────────────────────────────


class TestParetoFront:
    def test_empty_input(self) -> None:
        assert pareto_front([]) == []

    def test_single_candidate(self) -> None:
        c = SCCandidate(
            layers=[LayerConfig(32, NeuronType.LIF, 256, DecorrelationStrategy.LFSR)],
            accuracy=0.9,
            total_luts=1000,
        )
        front = pareto_front([c])
        assert len(front) == 1

    def test_dominated_candidate_excluded(self) -> None:
        a = SCCandidate(layers=[], accuracy=0.95, total_luts=500, total_power_mw=10)
        b = SCCandidate(layers=[], accuracy=0.90, total_luts=600, total_power_mw=15)
        front = pareto_front([a, b])
        assert len(front) == 1
        assert front[0] is a

    def test_non_dominated_both_kept(self) -> None:
        a = SCCandidate(layers=[], accuracy=0.95, total_luts=1000, total_power_mw=20)
        b = SCCandidate(layers=[], accuracy=0.90, total_luts=500, total_power_mw=10)
        front = pareto_front([a, b])
        assert len(front) == 2

    def test_crowding_distance_assigned(self) -> None:
        candidates = [
            SCCandidate(layers=[], accuracy=0.90, total_luts=100, total_power_mw=5),
            SCCandidate(layers=[], accuracy=0.93, total_luts=200, total_power_mw=10),
            SCCandidate(layers=[], accuracy=0.96, total_luts=300, total_power_mw=15),
            SCCandidate(layers=[], accuracy=0.99, total_luts=400, total_power_mw=20),
        ]
        front = pareto_front(candidates)
        assert any(c.crowding_distance == float("inf") for c in front)

    def test_crowding_distance_interior(self) -> None:
        candidates = [
            SCCandidate(layers=[], accuracy=0.90, total_luts=100, total_power_mw=5),
            SCCandidate(layers=[], accuracy=0.93, total_luts=200, total_power_mw=10),
            SCCandidate(layers=[], accuracy=0.96, total_luts=300, total_power_mw=15),
            SCCandidate(layers=[], accuracy=0.99, total_luts=400, total_power_mw=20),
        ]
        front = pareto_front(candidates)
        interior = [c for c in front if c.crowding_distance != float("inf")]
        for c in interior:
            assert c.crowding_distance > 0


# ── Evolutionary NAS Tests ───────────────────────────────────────────


class TestEvolutionaryNAS:
    def test_search_returns_non_empty_front(self) -> None:
        report = run_nas(population_size=10, num_generations=5, seed=42)
        assert len(report.pareto_front) > 0

    def test_search_history_recorded(self) -> None:
        report = run_nas(population_size=10, num_generations=5, seed=42)
        assert len(report.search_history) == 5

    def test_best_accuracy_positive(self) -> None:
        report = run_nas(population_size=10, num_generations=10, seed=42)
        assert report.best_accuracy > 0.0

    def test_most_efficient_exists(self) -> None:
        report = run_nas(population_size=10, num_generations=5, seed=42)
        assert report.most_efficient is not None

    def test_budget_constraint_respected(self) -> None:
        budget = FPGAResourceBudget(max_luts=10_000_000)
        report = run_nas(budget=budget, population_size=10, num_generations=5, seed=42)
        for c in report.pareto_front:
            assert c.total_luts <= budget.max_luts

    def test_search_can_score_candidates_with_surrogate_optimizer(self) -> None:
        surrogate = _FakeSurrogateOptimiser()
        report = run_nas(
            population_size=6,
            num_generations=2,
            seed=42,
            surrogate_optimizer=surrogate,
        )

        assert surrogate.calls
        assert len(report.pareto_front) > 0
        assert all(candidate.accuracy == 0.99 for candidate in report.pareto_front)

    def test_summary_format(self) -> None:
        report = run_nas(population_size=10, num_generations=3, seed=42)
        s = report.summary()
        assert "SC-NAS Report" in s
        assert "Pareto front size" in s

    def test_wall_time_recorded(self) -> None:
        report = run_nas(population_size=10, num_generations=3, seed=42)
        assert report.wall_time_s > 0.0

    def test_mutation_preserves_min_layers(self) -> None:
        nas = EvolutionaryNAS(NASObjective(), FPGAResourceBudget(), seed=42)
        for _ in range(50):
            parent = nas._random_candidate()
            child = nas._mutate(parent, 1)
            assert len(child.layers) >= 2

    def test_crossover_produces_valid_candidate(self) -> None:
        nas = EvolutionaryNAS(NASObjective(), FPGAResourceBudget(), seed=42)
        a = nas._random_candidate()
        b = nas._random_candidate()
        child = nas._crossover(a, b, 1)
        assert len(child.layers) >= 1
        child.evaluate_resources()
        assert child.total_luts > 0

    def test_convergence_early_stop(self) -> None:
        report = run_nas(
            population_size=10,
            num_generations=100,
            seed=42,
            convergence_patience=3,
        )
        assert len(report.search_history) < 100

    def test_history_has_dsp_bram(self) -> None:
        report = run_nas(population_size=10, num_generations=3, seed=42)
        assert "best_dsp" in report.search_history[0]
        assert "best_bram_kb" in report.search_history[0]

    def test_neuron_count_mutation(self) -> None:
        nas = EvolutionaryNAS(NASObjective(), FPGAResourceBudget(), seed=42)
        parent = nas._random_candidate()
        original_neurons = [l.neurons for l in parent.layers]
        mutated_any = False
        for _ in range(50):
            child = nas._mutate(parent, 1)
            if any(
                c.neurons != o
                for c, o in zip(child.layers, original_neurons)
                if len(child.layers) == len(parent.layers)
            ):
                mutated_any = True
                break
        # With 50 tries and 1/6 chance of neuron_count mutation, extremely likely
        assert mutated_any


# ── NAS Verilog Emitter Tests ────────────────────────────────────────


class TestNASVerilogEmitter:
    def _make_candidate(self) -> SCCandidate:
        c = SCCandidate(
            layers=[
                LayerConfig(32, NeuronType.LIF, 256, DecorrelationStrategy.LFSR),
                LayerConfig(64, NeuronType.IZHIKEVICH, 512, DecorrelationStrategy.SOBOL),
            ],
            accuracy=0.95,
        )
        c.evaluate_resources()
        return c

    def test_emit_contains_module(self) -> None:
        c = self._make_candidate()
        v = NASVerilogEmitter.emit(c)
        assert "module sc_nas_network" in v
        assert "endmodule" in v

    def test_emit_has_parameters(self) -> None:
        c = self._make_candidate()
        v = NASVerilogEmitter.emit(c)
        assert "L0_NEURONS    = 32" in v
        assert "L1_NEURONS    = 64" in v
        assert "L0_BITSTREAM  = 256" in v

    def test_emit_has_neuron_modules(self) -> None:
        c = self._make_candidate()
        v = NASVerilogEmitter.emit(c)
        assert "sc_lif_neuron" in v
        assert "sc_izhikevich_neuron" in v

    def test_emit_has_resource_comment(self) -> None:
        c = self._make_candidate()
        v = NASVerilogEmitter.emit(c)
        assert "LUTs" in v
        assert "DSPs" in v
        assert "BRAM" in v

    def test_emit_custom_name(self) -> None:
        c = self._make_candidate()
        v = NASVerilogEmitter.emit(c, module_name="my_net")
        assert "module my_net" in v

    def test_emit_pareto(self) -> None:
        c1 = self._make_candidate()
        c2 = self._make_candidate()
        result = NASVerilogEmitter.emit_pareto([c1, c2])
        assert len(result) == 2
        assert "sc_nas_pareto_0" in result
        assert "sc_nas_pareto_1" in result

    def test_emit_all_neuron_types(self) -> None:
        for nt in NeuronType:
            c = SCCandidate(
                layers=[
                    LayerConfig(16, nt, 128, DecorrelationStrategy.LFSR),
                ],
                accuracy=0.8,
            )
            c.evaluate_resources()
            v = NASVerilogEmitter.emit(c)
            assert "module" in v
            assert "endmodule" in v


class TestSCNASEngineEdgeBranches:
    """Resource-utilisation ratios, empty-front report accessors, and the
    Rust-evolution tournament path (flag + import branch)."""

    @staticmethod
    def _candidate(fitness: float) -> SCCandidate:
        return SCCandidate(
            layers=[LayerConfig(32, NeuronType.LIF, 256, DecorrelationStrategy.LFSR)],
            fitness=fitness,
        )

    def test_resource_budget_utilisation_ratios(self) -> None:
        budget = FPGAResourceBudget(max_luts=1000, max_ffs=2000, max_bram_kb=100, max_dsp=50)
        util = budget.utilisation(luts=500, ffs=500, bram=25, dsp=25)
        assert util["luts"] == 0.5
        assert util["ffs"] == 0.25
        assert util["bram"] == 0.25
        assert util["dsp"] == 0.5

    def test_report_accessors_on_empty_pareto_front(self) -> None:
        report = NASReport(pareto_front=[], search_history=[])
        assert report.best_accuracy == 0.0
        assert report.most_efficient is None

    def test_tournament_select_uses_rust_evo_when_available(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        nas = EvolutionaryNAS(
            objective=NASObjective(),
            budget=FPGAResourceBudget(),
            population_size=24,
            num_generations=1,
            seed=7,
        )
        population = [self._candidate(float(index)) for index in range(24)]
        captured: dict[str, list[float]] = {}

        def _fake_evo(fitness: list[float], n: int, k: int, seed: int) -> list[int]:
            captured["fitness"] = list(fitness)
            return [3]

        # The Rust tournament path activates only when the flag is set AND the
        # population exceeds the 20-candidate threshold.
        monkeypatch.setattr(nas_module, "_HAS_RUST_EVO", True)
        monkeypatch.setattr(nas_module, "py_evo_tournament", _fake_evo, raising=False)

        chosen = nas._tournament_select(population, k=3)
        assert chosen is population[3]
        assert len(captured["fitness"]) == 24

    def test_module_import_enables_rust_evo_when_extension_is_available(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured: dict[str, list[float]] = {}

        def _fake_evo(fitness: list[float], n: int, k: int, seed: int) -> list[int]:
            captured["fitness"] = list(fitness)
            assert n == 1
            assert k == 3
            assert seed >= 0
            return [5]

        class _RustExtension(ModuleType):
            py_evo_tournament: RustTournament

        rust_extension = _RustExtension("sc_neurocore_engine")
        rust_extension.py_evo_tournament = _fake_evo
        monkeypatch.setitem(sys.modules, "sc_neurocore_engine", rust_extension)

        importlib.reload(nas_module)
        try:
            assert nas_module._HAS_RUST_EVO is True
            nas = nas_module.EvolutionaryNAS(
                objective=nas_module.NASObjective(),
                budget=nas_module.FPGAResourceBudget(),
                population_size=24,
                num_generations=1,
                seed=11,
            )
            population = [self._candidate(float(index)) for index in range(24)]

            chosen = nas._tournament_select(population, k=3)

            assert chosen is population[5]
            assert captured["fitness"] == [candidate.fitness for candidate in population]
        finally:
            monkeypatch.delitem(sys.modules, "sc_neurocore_engine", raising=False)
            importlib.reload(nas_module)
