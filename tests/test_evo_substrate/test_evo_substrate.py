# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Evolutionary Substrate Tests

import numpy as np
import pytest

from sc_neurocore.evo_substrate.evo_substrate import (
    AgeRegulator,
    BloatPenalizer,
    CPPNGenome,
    CoevolutionArena,
    ComplexityTracker,
    CrossoverEngine,
    EvoStatisticsTracker,
    ExtinctionDetector,
    FitnessEvaluator,
    FitnessResult,
    FormalSafetyGuard,
    GenerationStats,
    Genome,
    GenomeSerializer,
    HWFitnessCollector,
    HWFitnessReport,
    HallOfFame,
    IslandModel,
    MutationConfig,
    MutationEngine,
    MutationType,
    NeuronGene,
    NoveltyArchive,
    Organism,
    OrganismEmitter,
    ParetoFront,
    PlasticityGene,
    ReplicationEngine,
    ResourceBudget,
    SafetyBounds,
    TileDeploymentTracker,
    TopologyGene,
    TournamentSelector,
    assign_species,
    compute_bloat,
    genome_complexity,
    genome_diff,
    genomic_distance,
    population_diversity,
    shared_fitness,
)


# ── Genome Tests ─────────────────────────────────────────────────────


class TestGenome:
    def test_to_vector(self):
        g = Genome()
        v = g.to_vector()
        assert len(v) == g.vector_dim

    def test_from_vector_roundtrip(self):
        g = Genome()
        v = g.to_vector()
        g2 = Genome.from_vector(v)
        assert abs(g2.topology.num_neurons - g.topology.num_neurons) < 1
        assert abs(g2.neuron.tau_fast - g.neuron.tau_fast) < 0.01

    def test_compute_id(self):
        g = Genome()
        gid = g.compute_id()
        assert len(gid) == 12
        assert g.genome_id == gid

    def test_id_deterministic(self):
        g1 = Genome()
        g2 = Genome()
        assert g1.compute_id() == g2.compute_id()

    def test_id_differs_on_change(self):
        g1 = Genome()
        g2 = Genome()
        g2.topology.num_neurons = 999
        assert g1.compute_id() != g2.compute_id()


# ── TopologyGene Tests ───────────────────────────────────────────────


class TestTopologyGene:
    def test_from_vector_clamps(self):
        v = np.array([0.0, 0.0, -1.0, -1.0, 0.0])
        tg = TopologyGene.from_vector(v)
        assert tg.num_neurons >= 2
        assert tg.num_layers >= 1
        assert tg.connectivity >= 0.01
        assert tg.bitstream_length >= 32


# ── NeuronGene Tests ─────────────────────────────────────────────────


class TestNeuronGene:
    def test_from_vector_clamps(self):
        v = np.zeros(8)
        ng = NeuronGene.from_vector(v)
        assert ng.tau_fast >= 0.5
        assert ng.theta >= 0.1


# ── PlasticityGene Tests ─────────────────────────────────────────────


class TestPlasticityGene:
    def test_from_vector_clamps(self):
        v = np.zeros(6)
        pg = PlasticityGene.from_vector(v)
        assert pg.stdp_lr > 0
        assert pg.stp_u_base >= 0.01


# ── MutationEngine Tests ─────────────────────────────────────────────


class TestMutationEngine:
    def test_point_mutation(self):
        me = MutationEngine(
            MutationConfig(point_rate=1.0, structural_rate=0, duplication_rate=0, swap_rate=0)
        )
        g = Genome()
        g.compute_id()
        child, mt = me.mutate(g)
        assert mt == MutationType.POINT
        assert child.parent_id == g.genome_id
        assert child.generation == g.generation + 1

    def test_structural_mutation(self):
        me = MutationEngine(MutationConfig(structural_rate=1.0), rng_seed=0)
        g = Genome()
        g.compute_id()
        child, mt = me.mutate(g)
        assert mt == MutationType.STRUCTURAL

    def test_child_has_new_id(self):
        me = MutationEngine()
        g = Genome()
        g.compute_id()
        child, _ = me.mutate(g)
        assert child.genome_id != ""

    def test_child_identity_reset(self):
        me = MutationEngine()
        g = Genome()
        g.identity_deep = 0.99
        g.compute_id()
        child, _ = me.mutate(g)
        assert child.identity_deep == 0.0

    def test_neuron_bounds_preserved(self):
        me = MutationEngine(MutationConfig(point_rate=1.0, point_sigma=10.0), rng_seed=42)
        g = Genome()
        g.compute_id()
        for _ in range(10):
            g, _ = me.mutate(g)
        assert g.neuron.tau_fast >= 0.5
        assert g.neuron.theta >= 0.1
        assert g.topology.num_neurons >= 2


# ── FitnessEvaluator Tests ──────────────────────────────────────────


class TestFitnessEvaluator:
    def test_evaluate(self):
        ev = FitnessEvaluator()
        g = Genome()
        g.compute_id()
        result = ev.evaluate(g, {"accuracy": 0.9})
        assert result.accuracy == 0.9
        assert result.composite > 0

    def test_energy_penalty(self):
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

    def test_composite_weights(self):
        r = FitnessResult("test", accuracy=1.0, energy_score=0.0, latency_score=0.0)
        c = r.compute_composite(w_acc=1.0, w_energy=0.0, w_latency=0.0)
        assert c == 1.0


# ── ReplicationEngine Tests ─────────────────────────────────────────


class TestReplicationEngine:
    def _metrics_fn(self, genome):
        return {"accuracy": 0.5 + 0.01 * genome.topology.num_neurons}

    def test_seed(self):
        re = ReplicationEngine()
        g = Genome()
        org = re.seed(g)
        assert len(re.population) == 1
        assert org.genome.genome_id != ""

    def test_replicate(self):
        re = ReplicationEngine()
        parent = re.seed(Genome())
        child = re.replicate(parent)
        assert child.genome.parent_id == parent.genome.genome_id
        assert re.total_replications == 1

    def test_evaluate_all(self):
        re = ReplicationEngine()
        re.seed(Genome())
        re.evaluate_all(self._metrics_fn)
        assert re.population[0].fitness is not None

    def test_select_and_cull(self):
        re = ReplicationEngine(max_population=20)
        for i in range(10):
            g = Genome()
            g.topology.num_neurons = i + 5
            re.seed(g)
        re.evaluate_all(self._metrics_fn)
        killed = re.select_and_cull(survival_fraction=0.5)
        assert killed > 0
        assert len(re.graveyard) == killed

    def test_evolve_generation(self):
        re = ReplicationEngine(max_population=8)
        for _ in range(4):
            re.seed(Genome())
        result = re.evolve_generation(self._metrics_fn)
        assert result["generation"] == 1
        assert result["population_size"] > 0
        assert result["best_fitness"] > 0

    def test_best_organism(self):
        re = ReplicationEngine()
        re.seed(Genome())
        re.evaluate_all(self._metrics_fn)
        assert re.best_organism is not None

    def test_mean_fitness(self):
        re = ReplicationEngine()
        for _ in range(4):
            re.seed(Genome())
        re.evaluate_all(self._metrics_fn)
        assert re.mean_fitness > 0


# ── OrganismEmitter Tests ───────────────────────────────────────────


class TestOrganismEmitter:
    def test_to_nir(self):
        g = Genome()
        g.compute_id()
        nir = OrganismEmitter.to_nir(g)
        assert "nodes" in nir
        assert "edges" in nir
        assert len(nir["nodes"]) == g.topology.num_neurons

    def test_nir_has_arcane_params(self):
        g = Genome()
        g.compute_id()
        nir = OrganismEmitter.to_nir(g)
        node = list(nir["nodes"].values())[0]
        assert node["type"] == "ArcaneNeuron"
        assert "tau_fast" in node

    def test_to_verilog(self):
        g = Genome()
        g.compute_id()
        v = OrganismEmitter.to_verilog(g)
        assert "module" in v
        assert f"NUM_NEURONS = {g.topology.num_neurons}" in v
        assert "sc_lif_neuron" in v

    def test_verilog_custom_name(self):
        g = Genome()
        g.compute_id()
        v = OrganismEmitter.to_verilog(g, module_name="test_org")
        assert "module test_org" in v


# ── Crossover Tests ─────────────────────────────────────────────────


class TestCrossover:
    def test_crossover_produces_child(self):
        cx = CrossoverEngine(rng_seed=42)
        a = Genome()
        a.compute_id()
        b = Genome()
        b.topology.num_neurons = 64
        b.compute_id()
        child = cx.crossover(a, b)
        assert child.genome_id != ""
        assert child.generation == 1

    def test_crossover_parent_id_format(self):
        cx = CrossoverEngine()
        a = Genome()
        a.compute_id()
        b = Genome()
        b.compute_id()
        child = cx.crossover(a, b)
        assert "x" in child.parent_id

    def test_crossover_mixes_genes(self):
        cx = CrossoverEngine(rng_seed=7)
        a = Genome()
        a.neuron.tau_fast = 1.0
        a.compute_id()
        b = Genome()
        b.neuron.tau_fast = 100.0
        b.compute_id()
        child = cx.crossover(a, b)
        assert child.neuron.tau_fast in (1.0, 100.0)  # mix


# ── Speciation Tests ────────────────────────────────────────────────


class TestSpeciation:
    def test_identical_genomes_same_species(self):
        orgs = [Organism(genome=Genome()) for _ in range(5)]
        for o in orgs:
            o.genome.compute_id()
        species = assign_species(orgs, threshold=0.5)
        assert len(species) == 1

    def test_different_genomes_separate_species(self):
        orgs = []
        for i in range(3):
            g = Genome()
            g.topology.num_neurons = (i + 1) * 200
            g.neuron.tau_fast = (i + 1) * 50.0
            g.compute_id()
            orgs.append(Organism(genome=g))
        species = assign_species(orgs, threshold=0.01)
        assert len(species) >= 2

    def test_genomic_distance_self(self):
        g = Genome()
        assert genomic_distance(g, g) == 0.0

    def test_genomic_distance_symmetric(self):
        a = Genome()
        b = Genome()
        b.topology.num_neurons = 100
        assert abs(genomic_distance(a, b) - genomic_distance(b, a)) < 1e-10


# ── Diversity Tests ─────────────────────────────────────────────────


class TestDiversity:
    def test_clones_zero_diversity(self):
        orgs = [Organism(genome=Genome()) for _ in range(5)]
        assert population_diversity(orgs) == 0.0

    def test_varied_population_positive_diversity(self):
        orgs = []
        for i in range(5):
            g = Genome()
            g.topology.num_neurons = 10 + i * 50
            orgs.append(Organism(genome=g))
        assert population_diversity(orgs) > 0.0

    def test_single_organism_zero(self):
        assert population_diversity([Organism(genome=Genome())]) == 0.0


# ── Lineage Tests ───────────────────────────────────────────────────


class TestLineage:
    def test_lineage_recorded_on_seed(self):
        re = ReplicationEngine()
        re.seed(Genome())
        assert re.lineage.num_records == 1

    def test_lineage_recorded_on_replicate(self):
        re = ReplicationEngine()
        parent = re.seed(Genome())
        re.replicate(parent)
        assert re.lineage.num_records == 2

    def test_get_ancestors(self):
        re = ReplicationEngine()
        parent = re.seed(Genome())
        child = re.replicate(parent)
        chain = re.lineage.get_ancestors(child.genome.genome_id)
        assert len(chain) >= 1


# ── Elitism Tests ───────────────────────────────────────────────────


class TestElitism:
    def test_best_survives_cull(self):
        re = ReplicationEngine(max_population=20, elitism=1)
        for i in range(10):
            g = Genome()
            g.topology.num_neurons = 10 + i * 10
            re.seed(g)
        re.evaluate_all(lambda g: {"accuracy": g.topology.num_neurons / 200.0})
        best_id = re.best_organism.genome.genome_id
        re.select_and_cull(survival_fraction=0.3)
        remaining_ids = [o.genome.genome_id for o in re.population]
        assert best_id in remaining_ids

    def test_diversity_in_evolve_result(self):
        re = ReplicationEngine(max_population=8)
        for _ in range(4):
            re.seed(Genome())
        result = re.evolve_generation(lambda g: {"accuracy": 0.5})
        assert "diversity" in result


# ── Safety Bounds Tests (Gap 1) ─────────────────────────────────────


class TestSafetyBounds:
    def test_clamp(self):
        sb = SafetyBounds(max_neurons=64)
        g = Genome()
        g.topology.num_neurons = 999
        sb.clamp(g)
        assert g.topology.num_neurons == 64

    def test_within_bounds(self):
        sb = SafetyBounds()
        g = Genome()
        assert sb.is_within_bounds(g)

    def test_out_of_bounds(self):
        sb = SafetyBounds(max_neurons=10)
        g = Genome()
        g.topology.num_neurons = 100
        assert not sb.is_within_bounds(g)


# ── Tile Deployment Tests (Gap 2) ────────────────────────────────────


class TestTileDeployment:
    def test_deploy(self):
        tracker = TileDeploymentTracker(num_tiles=4)
        g = Genome()
        g.compute_id()
        org = Organism(genome=g)
        alloc = tracker.deploy(org, 0)
        assert alloc.deployed
        assert org.tile_id == 0

    def test_free_tiles(self):
        tracker = TileDeploymentTracker(num_tiles=4)
        assert len(tracker.free_tiles) == 4
        g = Genome()
        g.compute_id()
        tracker.deploy(Organism(genome=g), 1)
        assert len(tracker.free_tiles) == 3

    def test_evict(self):
        tracker = TileDeploymentTracker(num_tiles=4)
        g = Genome()
        g.compute_id()
        tracker.deploy(Organism(genome=g), 0)
        tracker.evict(0)
        assert 0 in tracker.free_tiles

    def test_utilisation(self):
        tracker = TileDeploymentTracker(num_tiles=4)
        g = Genome()
        g.compute_id()
        tracker.deploy(Organism(genome=g), 0)
        assert tracker.utilisation == 0.25


# ── Hall of Fame Tests (Gap 3) ────────────────────────────────────────


class TestHallOfFame:
    def test_update(self):
        hof = HallOfFame(max_size=3)
        g = Genome()
        g.compute_id()
        org = Organism(genome=g, fitness=FitnessResult(g.genome_id, composite=0.8))
        assert hof.update(org)
        assert hof.best_fitness == 0.8

    def test_max_size(self):
        hof = HallOfFame(max_size=2)
        for i in range(5):
            g = Genome()
            g.topology.num_neurons = i + 10
            g.compute_id()
            org = Organism(genome=g, fitness=FitnessResult(g.genome_id, composite=i * 0.1))
            hof.update(org)
        assert hof.size == 2


# ── Island Model Tests (Gap 4) ────────────────────────────────────────


class TestIslandModel:
    def test_add_organism(self):
        im = IslandModel(num_islands=3)
        g = Genome()
        g.compute_id()
        im.add_organism(0, Organism(genome=g))
        assert im.total_population == 1

    def test_migrate(self):
        im = IslandModel(num_islands=2, migration_rate=1.0)
        g = Genome()
        g.compute_id()
        im.add_organism(0, Organism(genome=g))
        rng = np.random.default_rng(42)
        im.migrate(rng)
        assert im.total_population >= 2  # original + migrant


# ── Genome Serialization Tests (Gap 5) ───────────────────────────────


class TestGenomeSerializer:
    def test_roundtrip(self):
        g = Genome()
        g.compute_id()
        d = GenomeSerializer.to_dict(g)
        g2 = GenomeSerializer.from_dict(d)
        assert g2.genome_id == g.genome_id
        np.testing.assert_array_almost_equal(g2.to_vector(), g.to_vector(), decimal=4)

    def test_dict_keys(self):
        g = Genome()
        g.compute_id()
        d = GenomeSerializer.to_dict(g)
        assert "vector" in d
        assert "genome_id" in d


# ── Novelty Search Tests (Gap 6) ──────────────────────────────────────


class TestNoveltyArchive:
    def test_empty_archive_high_score(self):
        na = NoveltyArchive()
        assert na.novelty_score(np.array([1.0, 2.0])) == 1.0

    def test_add_novel(self):
        na = NoveltyArchive(threshold=0.01)
        assert na.maybe_add(np.array([1.0, 0.0]))
        assert na.size == 1

    def test_add_duplicate_rejected(self):
        na = NoveltyArchive(threshold=0.5)
        na.maybe_add(np.array([1.0, 0.0]))
        assert not na.maybe_add(np.array([1.0, 0.0]))  # identical


# ── Resource Budget Tests (Gap 7) ─────────────────────────────────────


class TestResourceBudget:
    def test_within_budget(self):
        rb = ResourceBudget(max_neurons=1024)
        g = Genome()
        ok, violations = rb.check(g)
        assert ok
        assert violations == []

    def test_exceeds_budget(self):
        rb = ResourceBudget(max_neurons=8)
        g = Genome()  # default 16 neurons
        ok, violations = rb.check(g)
        assert not ok
        assert len(violations) > 0


# ── Extinction Tests (Gap 8) ──────────────────────────────────────────


class TestExtinctionDetector:
    def test_no_extinction_early(self):
        ed = ExtinctionDetector(stagnation_gens=5)
        for i in range(3):
            assert ed.check(0.5) is False

    def test_detects_stagnation(self):
        ed = ExtinctionDetector(stagnation_gens=5)
        for _ in range(10):
            ed.check(0.5)  # all same fitness
        assert ed.extinction_count > 0

    def test_apply_kills(self):
        ed = ExtinctionDetector(kill_fraction=0.5)
        pop = [Organism(genome=Genome()) for _ in range(10)]
        rng = np.random.default_rng(42)
        killed = ed.apply(pop, rng)
        assert killed == 5


# ── Co-Evolution Tests (Gap 9) ────────────────────────────────────────


class TestCoevolution:
    def test_arena(self):
        arena = CoevolutionArena()
        g1 = Genome()
        g1.topology.num_neurons = 32
        g1.compute_id()
        g2 = Genome()
        g2.topology.num_neurons = 8
        g2.compute_id()
        arena.add_predator(Organism(genome=g1))
        arena.add_prey(Organism(genome=g2))
        assert arena.total_organisms == 2

    def test_interactions(self):
        arena = CoevolutionArena()
        g1 = Genome()
        g1.topology.num_neurons = 32
        g1.compute_id()
        g2 = Genome()
        g2.topology.num_neurons = 8
        g2.compute_id()
        arena.add_predator(Organism(genome=g1))
        arena.add_prey(Organism(genome=g2))
        results = arena.evaluate_interactions()
        assert len(results) == 2


# ── Formal Safety Guard Tests (Gap 10) ────────────────────────────────


class TestFormalSafetyGuard:
    def test_passes_valid(self):
        guard = FormalSafetyGuard()
        g = Genome()
        g.compute_id()
        result = guard.check(g)
        assert result.passed
        assert result.violations == []

    def test_rejects_invalid(self):
        guard = FormalSafetyGuard(SafetyBounds(max_neurons=10))
        g = Genome()  # 16 neurons > 10
        g.compute_id()
        result = guard.check(g)
        assert not result.passed
        assert not result.neuron_count_ok
        assert guard.rejected == 1

    def test_rejection_rate(self):
        guard = FormalSafetyGuard(SafetyBounds(max_neurons=10))
        g = Genome()
        g.compute_id()
        guard.check(g)  # fails
        g2 = Genome()
        g2.topology.num_neurons = 5
        g2.compute_id()
        guard.check(g2)  # passes
        assert guard.rejection_rate == 0.5


# ── Tournament Selection Tests (Gap 11) ──────────────────────────────


class TestTournamentSelector:
    def test_select(self):
        ts = TournamentSelector(tournament_size=2)
        pop = []
        for i in range(5):
            g = Genome()
            g.topology.num_neurons = i + 10
            g.compute_id()
            org = Organism(genome=g, fitness=FitnessResult(g.genome_id, composite=i * 0.1))
            pop.append(org)
        rng = np.random.default_rng(42)
        winner = ts.select(pop, rng)
        assert winner is not None

    def test_select_n(self):
        ts = TournamentSelector(tournament_size=3)
        pop = []
        for i in range(10):
            g = Genome()
            g.topology.num_neurons = i + 5
            g.compute_id()
            pop.append(Organism(genome=g, fitness=FitnessResult(g.genome_id, composite=i * 0.05)))
        rng = np.random.default_rng(0)
        selected = ts.select_n(pop, 4, rng)
        assert len(selected) == 4


# ── Pareto Front Tests (Gap 12) ───────────────────────────────────────


class TestParetoFront:
    def test_add_non_dominated(self):
        pf = ParetoFront()
        g = Genome()
        g.compute_id()
        org = Organism(
            genome=g,
            fitness=FitnessResult(g.genome_id, accuracy=0.9, energy_score=0.5, latency_score=0.8),
        )
        assert pf.update(org)
        assert pf.size == 1

    def test_dominated_rejected(self):
        pf = ParetoFront()
        g1 = Genome()
        g1.compute_id()
        org1 = Organism(
            genome=g1,
            fitness=FitnessResult(g1.genome_id, accuracy=0.9, energy_score=0.9, latency_score=0.9),
        )
        pf.update(org1)
        g2 = Genome()
        g2.topology.num_neurons = 8
        g2.compute_id()
        org2 = Organism(
            genome=g2,
            fitness=FitnessResult(g2.genome_id, accuracy=0.5, energy_score=0.5, latency_score=0.5),
        )
        assert not pf.update(org2)


# ── Age Regulation Tests (Gap 13) ─────────────────────────────────────


class TestAgeRegulator:
    def test_young_survive(self):
        ar = AgeRegulator(max_age=10)
        pop = [Organism(genome=Genome(), birth_generation=5)]
        killed = ar.apply(pop, current_generation=10)
        assert killed == 0

    def test_old_culled(self):
        ar = AgeRegulator(max_age=5)
        pop = [Organism(genome=Genome(), birth_generation=0)]
        killed = ar.apply(pop, current_generation=10)
        assert killed == 1
        assert not pop[0].alive


# ── Bloat Control Tests (Gap 14) ──────────────────────────────────────


class TestBloatControl:
    def test_compute_bloat(self):
        g = Genome()
        bm = compute_bloat(g)
        assert bm.total_params > 0
        assert bm.bloat_score > 0

    def test_penalizer_no_penalty(self):
        bp = BloatPenalizer(threshold=100.0)
        g = Genome()
        assert bp.penalize(0.9, g) == 0.9

    def test_penalizer_reduces(self):
        bp = BloatPenalizer(threshold=0.01)
        g = Genome()
        assert bp.penalize(0.9, g) < 0.9


# ── Fitness Sharing Tests (Gap 15) ────────────────────────────────────


class TestFitnessSharing:
    def test_shared_fitness_reduces(self):
        pop = []
        for _ in range(5):
            g = Genome()
            g.compute_id()
            pop.append(Organism(genome=g, fitness=FitnessResult(g.genome_id, composite=0.8)))
        sf = shared_fitness(pop[0], pop, sigma=1.0)
        assert sf < 0.8  # shared among 5 clones

    def test_unique_keeps_full(self):
        g1 = Genome()
        g1.topology.num_neurons = 4
        g1.compute_id()
        g2 = Genome()
        g2.topology.num_neurons = 1000
        g2.compute_id()
        org1 = Organism(genome=g1, fitness=FitnessResult(g1.genome_id, composite=0.8))
        org2 = Organism(genome=g2, fitness=FitnessResult(g2.genome_id, composite=0.5))
        sf = shared_fitness(org1, [org1, org2], sigma=0.0001)
        assert sf > 0.5  # only shares with itself


# ── CPPN Tests (Gap 16) ───────────────────────────────────────────────


class TestCPPN:
    def test_query(self):
        cppn = CPPNGenome()
        val = cppn.query(0.5, 0.5)
        assert 0.0 <= val <= 1.0  # sigmoid output

    def test_weight_matrix(self):
        cppn = CPPNGenome()
        w = cppn.generate_weight_matrix(4, 4)
        assert w.shape == (4, 4)

    def test_structure(self):
        cppn = CPPNGenome()
        assert cppn.num_nodes == 3
        assert cppn.num_edges == 2


# ── HW Fitness Tests (Gap 17) ─────────────────────────────────────────


class TestHWFitness:
    def test_report(self):
        r = HWFitnessReport("test_id", fpga_accuracy=0.9, fmax_mhz=200.0)
        assert r.hw_composite > 0

    def test_collector(self):
        col = HWFitnessCollector()
        col.submit(HWFitnessReport("g1", fpga_accuracy=0.8))
        assert col.total_reports == 1
        assert col.get("g1") is not None
        assert col.get("nonexistent") is None


# ── Evo Statistics Tests (Gap 18) ─────────────────────────────────────


class TestEvoStatistics:
    def test_record(self):
        est = EvoStatisticsTracker()
        est.record(GenerationStats(1, 10, 0.7, 0.5, 0.3))
        est.record(GenerationStats(2, 12, 0.8, 0.6, 0.25))
        assert est.generations_tracked == 2

    def test_trajectory(self):
        est = EvoStatisticsTracker()
        est.record(GenerationStats(1, 10, 0.5, 0.3, 0.2))
        est.record(GenerationStats(2, 10, 0.8, 0.5, 0.3))
        assert est.fitness_trajectory == [0.5, 0.8]
        assert est.improvement_rate() == pytest.approx(0.3)


# ── Genome Diff Tests (Gap 19) ────────────────────────────────────────


class TestGenomeDiff:
    def test_identical(self):
        g = Genome()
        d = genome_diff(g, g)
        assert d.is_identical
        assert d.neuron_delta == 0

    def test_different(self):
        a = Genome()
        b = Genome()
        b.topology.num_neurons = 64
        d = genome_diff(a, b)
        assert not d.is_identical
        assert d.neuron_delta == 48


# ── Complexity Metric Tests (Gap 20) ──────────────────────────────────


class TestComplexityMetric:
    def test_complexity_positive(self):
        g = Genome()
        assert genome_complexity(g) > 0

    def test_bigger_is_more_complex(self):
        small = Genome()
        small.topology.num_neurons = 4
        big = Genome()
        big.topology.num_neurons = 512
        big.topology.num_layers = 8
        assert genome_complexity(big) > genome_complexity(small)

    def test_tracker(self):
        ct = ComplexityTracker()
        pop = [Organism(genome=Genome()) for _ in range(5)]
        ct.record(0, pop)
        ct.record(1, pop)
        assert len(ct.mean_trajectory) == 2
