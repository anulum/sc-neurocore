# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestReplicationEngine from former test_replication.py

"""Focused suite: TestReplicationEngine from former test_replication.py."""

from __future__ import annotations

from tests.test_evo_substrate.replication_support import *  # noqa: F403


class TestReplicationEngine:
    def _metrics_fn(self, genome: Genome) -> dict[str, float]:
        return {"accuracy": 0.5 + 0.01 * genome.topology.num_neurons}

    def test_seed(self) -> None:
        re = ReplicationEngine()
        g = Genome()
        org = re.seed(g)
        assert len(re.population) == 1
        assert org.genome.genome_id != ""

    def test_replicate(self) -> None:
        re = ReplicationEngine()
        parent = re.seed(Genome())
        child = re.replicate(parent)
        assert child is not None
        assert child.genome.parent_id == parent.genome.genome_id
        assert re.total_replications == 1

    def test_non_industrial_replicate_bypasses_formal_safety(self) -> None:
        re = ReplicationEngine(industrial_mode=False)
        re.safety_guard = FormalSafetyGuard(SafetyBounds(max_neurons=4))
        parent = re.seed(Genome())

        child = re.replicate(parent)

        assert child is not None
        assert re.safety_guard.checked == 0

    def test_replication_never_exceeds_population_capacity(self) -> None:
        mutation_engine = ReplicationEngine(max_population=1)
        mutation_parent = mutation_engine.seed(Genome())
        mutation_child = mutation_engine.replicate(mutation_parent)

        crossover_engine = ReplicationEngine(max_population=2)
        first_parent = crossover_engine.seed(Genome())
        second_parent = crossover_engine.seed(Genome())
        crossover_child = crossover_engine.replicate_crossover(first_parent, second_parent)

        assert mutation_child is not None
        assert len(mutation_engine.population) == 1
        assert crossover_child is not None
        assert len(crossover_engine.population) == 2

    def test_evaluate_all(self) -> None:
        re = ReplicationEngine()
        re.seed(Genome())
        re.evaluate_all(self._metrics_fn)
        assert re.population[0].fitness is not None

    def test_evaluate_all_ignores_dead_organisms(self) -> None:
        re = ReplicationEngine()
        organism = re.seed(Genome())
        organism.alive = False

        re.evaluate_all(self._metrics_fn)

        assert organism.fitness is None
        assert re.hall_of_fame.size == 0

    def test_select_and_cull(self) -> None:
        re = ReplicationEngine(max_population=20)
        for i in range(10):
            g = Genome()
            g.topology.num_neurons = i + 5
            re.seed(g)
        re.evaluate_all(self._metrics_fn)
        killed = re.select_and_cull(survival_fraction=0.5)
        assert killed > 0
        assert len(re.graveyard) == killed

    def test_evolve_generation(self) -> None:
        re = ReplicationEngine(max_population=8)
        for _ in range(4):
            re.seed(Genome())
        result = re.evolve_generation(self._metrics_fn)
        assert result["generation"] == 1
        assert result["population_size"] > 0
        assert result["best_fitness"] > 0

    def test_best_organism(self) -> None:
        re = ReplicationEngine()
        re.seed(Genome())
        re.evaluate_all(self._metrics_fn)
        assert re.best_organism is not None

    def test_mean_fitness(self) -> None:
        re = ReplicationEngine()
        for _ in range(4):
            re.seed(Genome())
        re.evaluate_all(self._metrics_fn)
        assert re.mean_fitness > 0

    def test_industrial_replicate_rejects_child_outside_safety_bounds(self) -> None:
        re = ReplicationEngine()
        re.safety_guard = FormalSafetyGuard(SafetyBounds(max_neurons=4))
        parent_genome = Genome()
        parent_genome.topology.num_neurons = 16
        parent = re.seed(parent_genome)

        child = re.replicate(parent)

        assert child is None
        assert re.total_replications == 0
        assert re.safety_guard.rejected == 1

    def test_replicate_crossover_records_child_lineage_when_safety_passes(self) -> None:
        re = ReplicationEngine(max_population=4)
        parent_a = re.seed(Genome())
        parent_b_genome = Genome()
        parent_b_genome.topology.num_neurons = 24
        parent_b = re.seed(parent_b_genome)

        child = re.replicate_crossover(parent_a, parent_b)

        assert child is not None
        assert child.genome.parent_id == (
            f"{parent_a.genome.genome_id}x{parent_b.genome.genome_id}"
        )
        assert re.total_replications == 1
        assert len(re.population) == 3

    def test_industrial_replicate_crossover_rejects_child_outside_safety_bounds(self) -> None:
        re = ReplicationEngine()
        re.safety_guard = FormalSafetyGuard(SafetyBounds(max_neurons=4))
        parent_a = re.seed(Genome())
        parent_b = re.seed(Genome())

        child = re.replicate_crossover(parent_a, parent_b)

        assert child is None
        assert re.total_replications == 0
        assert re.safety_guard.rejected == 1

    def test_runtime_fault_replay_plan_applies_replay_penalty(self) -> None:
        class ReplayPolicy:
            def evaluate(
                self,
                bitstreams: np.ndarray[Any, Any],
                *,
                layer_id: str,
                fault_model: FaultModel,
                ber: float,
                seed: int,
            ) -> DegradationPlan:
                audit = BitstreamAuditReport(
                    layer=layer_id,
                    stream_length=128,
                    num_neurons=1,
                    status=AuditSeverity.WARNING,
                )
                observation = SeededFaultObservation(
                    layer_id=layer_id,
                    seed=seed,
                    fault_model=fault_model,
                    ber=ber,
                    affected_bits=2,
                    bitstream_length=128,
                    affected_ratio=0.01,
                    audit=audit,
                )
                return evo_mod.DegradationPlan(
                    action=evo_mod.DegradationAction.REPLAY_WITH_SEED,
                    observation=observation,
                    recommended_bitstream_length=256,
                    replay_seed=seed,
                    reason="seeded replay required",
                )

        g = Genome()
        g.compute_id()
        org = Organism(
            genome=g,
            fitness=FitnessResult(g.genome_id, composite=1.0),
        )
        config = evo_mod.RuntimeFaultConfig(fitness_penalty_on_replay=0.5)
        re = ReplicationEngine(
            runtime_fault_config=config,
            degradation_policy=cast(Any, ReplayPolicy()),
        )

        check = re.verify_runtime_faults(org)

        assert org.genome.topology.bitstream_length == 256
        assert org.fitness is not None
        assert org.fitness.composite == pytest.approx(0.5)
        assert check.action == evo_mod.DegradationAction.REPLAY_WITH_SEED.value
        assert org.runtime_fault_checks == [check]

    def test_industrial_select_and_cull_applies_extinction_event(self) -> None:
        re = ReplicationEngine(max_population=8)
        re.extinction_detector = ExtinctionDetector(stagnation_gens=1, kill_fraction=0.5)
        for _ in range(4):
            org = re.seed(Genome())
            org.fitness = FitnessResult(org.genome.genome_id, composite=0.5)

        killed = re.select_and_cull(survival_fraction=1.0)

        assert killed == 0
        assert len(re.population) == 2
        assert re.extinction_detector.extinction_count == 1

    def test_non_industrial_generation_crosses_survivors_when_roll_is_low(self) -> None:
        class AlwaysCrossoverRng:
            def random(self) -> float:
                return 0.0

        re = ReplicationEngine(max_population=3, industrial_mode=False)
        first = re.seed(Genome())
        second_genome = Genome()
        second_genome.topology.num_neurons = 32
        second = re.seed(second_genome)
        first.fitness = FitnessResult(first.genome.genome_id, composite=0.9)
        second.fitness = FitnessResult(second.genome.genome_id, composite=0.8)
        re.mutator.rng = cast(Any, AlwaysCrossoverRng())

        result = re.evolve_generation(lambda genome: {"accuracy": 0.7})

        assert result["children"] == 1
        assert any("x" in org.genome.parent_id for org in re.population)

    def test_industrial_generation_continues_when_tournament_returns_no_parent(self) -> None:
        class NoParentTournament:
            def select(
                self,
                population: list[Organism],
                rng: np.random.Generator,
            ) -> None:
                return None

        re = ReplicationEngine(max_population=4)
        re.tournament = cast(Any, NoParentTournament())
        for _ in range(2):
            re.seed(Genome())

        result = re.evolve_generation(lambda genome: {"accuracy": 0.7})

        assert result["children"] == 0
        assert result["population_size"] == 2

    def test_generation_counts_no_child_when_formal_safety_rejects(self) -> None:
        re = ReplicationEngine(max_population=3)
        re.safety_guard = FormalSafetyGuard(SafetyBounds(max_neurons=4))
        re.seed(Genome())
        re.seed(Genome())

        result = re.evolve_generation(lambda genome: {"accuracy": 0.7})

        assert result["children"] == 0
        assert re.safety_guard.rejected > 0
