# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Source/config provenance header

# Tests for sc_neurocore.nas (hardware-aware NAS + formal equivalence)

from __future__ import annotations

import numpy as np

from sc_neurocore.nas.search_space import (
    Architecture,
    SearchSpace,
    NEURON_CHOICES,
    WIDTH_CHOICES,
    L_CHOICES,
)
from sc_neurocore.nas.search import (
    nas,
    NASResult,
    _evaluate,
    _dominates,
    _non_dominated_sort,
    _crowding_distance,
)
from sc_neurocore.nas.equiv import (
    check_equivalence,
    generate_miter,
    generate_sby,
    EquivResult,
)


class TestArchitecture:
    def test_fields(self) -> None:
        a = Architecture(
            n_inputs=64,
            layer_widths=[32, 16],
            neuron_types=["StochasticLIFNeuron", "StochasticLIFNeuron"],
            bitstream_lengths=[128, 64],
            delay_ranges=[2, 0],
        )
        assert a.n_layers == 2
        assert a.layer_sizes == [(64, 32), (32, 16)]
        assert a.total_params == 64 * 32 + 32 * 16

    def test_single_layer(self) -> None:
        a = Architecture(
            n_inputs=10,
            layer_widths=[5],
            neuron_types=["StochasticLIFNeuron"],
            bitstream_lengths=[256],
            delay_ranges=[0],
        )
        assert a.n_layers == 1
        assert a.total_params == 50


class TestSearchSpace:
    def test_defaults(self) -> None:
        sp = SearchSpace(n_inputs=64, n_outputs=10)
        assert sp.min_layers == 1
        assert sp.max_layers == 4
        assert len(sp.width_choices) == len(WIDTH_CHOICES)

    def test_random_architecture(self) -> None:
        sp = SearchSpace(n_inputs=64, n_outputs=10)
        rng = np.random.RandomState(42)
        arch = sp.random_architecture(rng)
        assert arch.n_inputs == 64
        assert arch.layer_widths[-1] == 10
        assert arch.n_layers >= 1
        assert arch.n_layers <= 4
        for nt in arch.neuron_types:
            assert nt in NEURON_CHOICES
        for L in arch.bitstream_lengths:
            assert L in L_CHOICES

    def test_mutate(self) -> None:
        sp = SearchSpace(n_inputs=64, n_outputs=10)
        rng = np.random.RandomState(42)
        original = sp.random_architecture(rng)
        mutated = sp.mutate(original, rng)
        assert mutated.n_inputs == original.n_inputs
        assert mutated.n_layers == original.n_layers

    def test_mutate_all_genes(self) -> None:
        sp = SearchSpace(n_inputs=64, n_outputs=10, min_layers=3, max_layers=3)
        # Run enough mutations to hit all gene types including width (gene=0)
        for seed in range(50):
            rng = np.random.RandomState(seed)
            original = sp.random_architecture(np.random.RandomState(0))
            sp.mutate(original, rng)

    def test_crossover_same_layers(self) -> None:
        sp = SearchSpace(n_inputs=32, n_outputs=8, min_layers=2, max_layers=2)
        rng = np.random.RandomState(42)
        a = sp.random_architecture(rng)
        b = sp.random_architecture(rng)
        child = sp.crossover(a, b, rng)
        assert child.n_layers == 2
        assert child.layer_widths[-1] == 8

    def test_crossover_different_layers(self) -> None:
        sp = SearchSpace(n_inputs=32, n_outputs=8)
        rng = np.random.RandomState(42)
        a = Architecture(
            n_inputs=32,
            layer_widths=[16, 8],
            neuron_types=["StochasticLIFNeuron"] * 2,
            bitstream_lengths=[128, 64],
            delay_ranges=[0, 0],
        )
        b = Architecture(
            n_inputs=32,
            layer_widths=[32, 16, 8],
            neuron_types=["SCIzhikevichNeuron"] * 3,
            bitstream_lengths=[256, 128, 64],
            delay_ranges=[1, 1, 1],
        )
        child = sp.crossover(a, b, rng)
        assert child.n_layers == 2  # min(2, 3)

    def test_space_size(self) -> None:
        sp = SearchSpace(n_inputs=64, n_outputs=10, min_layers=1, max_layers=2)
        assert sp.space_size > 0


class TestEvaluate:
    def test_default_proxy(self) -> None:
        arch = Architecture(
            n_inputs=64,
            layer_widths=[32, 10],
            neuron_types=["StochasticLIFNeuron"] * 2,
            bitstream_lengths=[128, 128],
            delay_ranges=[0, 0],
        )
        _evaluate(arch, "artix7")
        assert arch.fitness_luts > 0
        assert arch.fitness_energy_nj > 0
        assert 0 < arch.fitness_accuracy <= 1.0

    def test_custom_accuracy_fn(self) -> None:
        arch = Architecture(
            n_inputs=16,
            layer_widths=[8],
            neuron_types=["StochasticLIFNeuron"],
            bitstream_lengths=[64],
            delay_ranges=[0],
        )
        _evaluate(arch, "ice40", accuracy_fn=lambda a: 0.95)
        assert arch.fitness_accuracy == 0.95


class TestDominance:
    def test_dominates(self) -> None:
        a = Architecture(
            n_inputs=16,
            layer_widths=[8],
            neuron_types=["StochasticLIFNeuron"],
            bitstream_lengths=[64],
            delay_ranges=[0],
            fitness_accuracy=0.9,
            fitness_energy_nj=100.0,
        )
        b = Architecture(
            n_inputs=16,
            layer_widths=[8],
            neuron_types=["StochasticLIFNeuron"],
            bitstream_lengths=[64],
            delay_ranges=[0],
            fitness_accuracy=0.8,
            fitness_energy_nj=200.0,
        )
        assert _dominates(a, b)
        assert not _dominates(b, a)

    def test_no_dominance(self) -> None:
        a = Architecture(
            n_inputs=16,
            layer_widths=[8],
            neuron_types=["StochasticLIFNeuron"],
            bitstream_lengths=[64],
            delay_ranges=[0],
            fitness_accuracy=0.9,
            fitness_energy_nj=200.0,
        )
        b = Architecture(
            n_inputs=16,
            layer_widths=[8],
            neuron_types=["StochasticLIFNeuron"],
            bitstream_lengths=[64],
            delay_ranges=[0],
            fitness_accuracy=0.8,
            fitness_energy_nj=100.0,
        )
        assert not _dominates(a, b)
        assert not _dominates(b, a)


class TestNonDominatedSort:
    def _make_arch(self, acc: float, energy: float) -> Architecture:
        return Architecture(
            n_inputs=16,
            layer_widths=[8],
            neuron_types=["StochasticLIFNeuron"],
            bitstream_lengths=[64],
            delay_ranges=[0],
            fitness_accuracy=acc,
            fitness_energy_nj=energy,
        )

    def test_single_front(self) -> None:
        pop = [self._make_arch(0.9, 200), self._make_arch(0.8, 100)]
        fronts = _non_dominated_sort(pop)
        assert len(fronts) == 1
        assert len(fronts[0]) == 2

    def test_two_fronts(self) -> None:
        pop = [
            self._make_arch(0.9, 100),  # dominates c
            self._make_arch(0.8, 50),  # dominates c
            self._make_arch(0.7, 200),  # dominated by both
        ]
        fronts = _non_dominated_sort(pop)
        assert len(fronts) == 2
        assert len(fronts[0]) == 2
        assert len(fronts[1]) == 1


class TestCrowdingDistance:
    def _make_arch(self, acc: float, energy: float) -> Architecture:
        return Architecture(
            n_inputs=16,
            layer_widths=[8],
            neuron_types=["StochasticLIFNeuron"],
            bitstream_lengths=[64],
            delay_ranges=[0],
            fitness_accuracy=acc,
            fitness_energy_nj=energy,
        )

    def test_two_points(self) -> None:
        front = [self._make_arch(0.9, 100), self._make_arch(0.8, 200)]
        dist = _crowding_distance(front)
        assert all(d == float("inf") for d in dist)

    def test_three_points(self) -> None:
        front = [
            self._make_arch(0.9, 300),
            self._make_arch(0.85, 200),
            self._make_arch(0.8, 100),
        ]
        dist = _crowding_distance(front)
        assert dist[0] == float("inf") or dist[2] == float("inf")


class TestNAS:
    def test_small_search(self) -> None:
        sp = SearchSpace(n_inputs=16, n_outputs=4, min_layers=1, max_layers=2)
        result = nas(sp, target="artix7", population_size=10, generations=3, seed=42)
        assert isinstance(result, NASResult)
        assert len(result.pareto_front) > 0
        assert result.generations == 3
        assert result.total_evaluations > 0

    def test_best_accuracy(self) -> None:
        sp = SearchSpace(n_inputs=16, n_outputs=4, min_layers=1, max_layers=2)
        result = nas(sp, target="artix7", population_size=10, generations=3)
        best = result.best_accuracy()
        assert best is not None
        assert best.fitness_accuracy > 0

    def test_best_efficiency(self) -> None:
        sp = SearchSpace(n_inputs=16, n_outputs=4, min_layers=1, max_layers=2)
        result = nas(sp, target="artix7", population_size=10, generations=3)
        best = result.best_efficiency()
        assert best is not None
        assert best.fitness_energy_nj > 0

    def test_summary(self) -> None:
        sp = SearchSpace(n_inputs=16, n_outputs=4, min_layers=1, max_layers=2)
        result = nas(sp, target="ice40", population_size=10, generations=2)
        s = result.summary()
        assert "NAS Result" in s
        assert "Pareto front" in s

    def test_lut_constraint(self) -> None:
        sp = SearchSpace(n_inputs=16, n_outputs=4, min_layers=1, max_layers=2)
        result = nas(sp, target="ice40", population_size=10, generations=3, max_luts=5000)
        for arch in result.pareto_front:
            assert arch.fitness_luts > 0

    def test_custom_accuracy(self) -> None:
        sp = SearchSpace(n_inputs=8, n_outputs=2, min_layers=1, max_layers=1)
        result = nas(
            sp,
            target="artix7",
            population_size=6,
            generations=2,
            accuracy_fn=lambda a: 0.5 + 0.5 * min(a.total_params / 500, 1.0),
        )
        assert all(a.fitness_accuracy >= 0.5 for a in result.pareto_front)

    def test_empty_pareto_front_methods(self) -> None:
        r = NASResult(pareto_front=[], all_evaluated=[], generations=0, total_evaluations=0)
        assert r.best_accuracy() is None
        assert r.best_efficiency() is None


class TestEquivResult:
    def test_summary_pass(self) -> None:
        r = EquivResult(module="sc_lif_neuron", passed=True, depth=30, engine="z3", log="ok")
        assert "PROVED" in r.summary()

    def test_summary_fail(self) -> None:
        r = EquivResult(module="sc_lif_neuron", passed=False, depth=30, engine="z3", log="err")
        assert "FAILED" in r.summary()


class TestEquivChecker:
    def test_check_no_run(self) -> None:
        r = check_equivalence(run=False)
        assert r.passed is True
        assert "not run" in r.log

    def test_generate_miter(self) -> None:
        v = generate_miter("sc_lif_neuron", "sc_lif_reference", "equiv_test")
        assert "module equiv_test" in v
        assert "sc_lif_neuron" in v
        assert "sc_lif_reference" in v
        assert "assert" in v

    def test_generate_sby(self) -> None:
        s = generate_sby("equiv_test", ["a.v", "b.v", "c.v"], depth=20)
        assert "depth 20" in s
        assert "read -formal a.v" in s
        assert "prep -top equiv_test" in s

    def test_generate_miter_custom_width(self) -> None:
        v = generate_miter("dut", "ref", "top", data_width=8, fraction=4)
        assert "DATA_WIDTH = 8" in v
        assert "FRACTION = 4" in v
