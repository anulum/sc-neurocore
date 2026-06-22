# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Self-Evolving Meta-Plasticity Tests

import numpy as np
import pytest

from sc_neurocore.meta_plasticity.meta_plasticity import (
    CheckpointStore,
    ContextRuleBank,
    CuriositySignal,
    EWCProtection,
    EngineConfig,
    FitnessTrajectory,
    HomeostaticParams,
    MetaControlSignal,
    MetaController,
    MetaLearningRate,
    MetaPlasticityEngine,
    MetaSignalType,
    NeuromodulatorState,
    NeuromodulatorType,
    PlasticityRuleSet,
    RuleConstraints,
    RuleEvolver,
    STDPParams,
    STPParams,
    SleepPhase,
    TaggingModel,
    inject_diversity,
    population_diversity,
)


# ── STDPParams Tests ─────────────────────────────────────────────────


class TestSTDPParams:
    def test_to_vector(self):
        p = STDPParams()
        v = p.to_vector()
        assert len(v) == 5

    def test_from_vector_roundtrip(self):
        p = STDPParams(tau_plus=15.0, a_plus=0.02, lr=0.005)
        v = p.to_vector()
        p2 = STDPParams.from_vector(v)
        assert abs(p2.tau_plus - 15.0) < 1e-6
        assert abs(p2.lr - 0.005) < 1e-6

    def test_from_vector_clamps(self):
        v = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        p = STDPParams.from_vector(v)
        assert p.tau_plus >= 1.0
        assert p.a_plus > 0


# ── STPParams Tests ──────────────────────────────────────────────────


class TestSTPParams:
    def test_to_vector(self):
        p = STPParams()
        assert len(p.to_vector()) == 3

    def test_from_vector_clamps_u(self):
        v = np.array([2.0, 100.0, 50.0])
        p = STPParams.from_vector(v)
        assert p.u_base <= 0.99


# ── HomeostaticParams Tests ──────────────────────────────────────────


class TestHomeostaticParams:
    def test_adapt_increases_gain(self):
        hp = HomeostaticParams(target_rate_hz=10.0, current_gain=1.0)
        hp.adapt(5.0)  # Below target
        assert hp.current_gain > 1.0

    def test_adapt_decreases_gain(self):
        hp = HomeostaticParams(target_rate_hz=5.0, current_gain=1.0)
        hp.adapt(10.0)  # Above target
        assert hp.current_gain < 1.0

    def test_gain_bounded(self):
        hp = HomeostaticParams(target_rate_hz=100.0, gain_adaptation_rate=1.0)
        for _ in range(100):
            hp.adapt(0.0)
        assert hp.current_gain <= 10.0


# ── PlasticityRuleSet Tests ──────────────────────────────────────────


class TestPlasticityRuleSet:
    def test_to_vector(self):
        rs = PlasticityRuleSet()
        v = rs.to_vector()
        assert len(v) == rs.vector_dim

    def test_from_vector_roundtrip(self):
        rs = PlasticityRuleSet()
        v = rs.to_vector()
        rs2 = PlasticityRuleSet.from_vector(v)
        assert abs(rs2.stdp.tau_plus - rs.stdp.tau_plus) < 1e-6
        assert abs(rs2.stp.u_base - rs.stp.u_base) < 1e-6

    def test_copy_independent(self):
        rs = PlasticityRuleSet()
        rs2 = rs.copy()
        rs2.stdp.lr = 999.0
        assert rs.stdp.lr != 999.0


# ── MetaController Tests ─────────────────────────────────────────────


class TestMetaController:
    def _feed_observations(self, mc, n=20, novelty=0.8, surprise=0.5, gci=0.5):
        for _ in range(n):
            mc.observe({"novelty": novelty, "surprise": surprise, "gci": gci})

    def test_no_op_few_observations(self):
        mc = MetaController()
        mc.observe({"novelty": 0.5})
        signals = mc.decide()
        assert signals[0].signal_type == MetaSignalType.NO_OP

    def test_increase_lr_on_high_novelty(self):
        mc = MetaController()
        self._feed_observations(mc, novelty=0.9, surprise=0.5, gci=0.5)
        signals = mc.decide()
        types = [s.signal_type for s in signals]
        assert MetaSignalType.INCREASE_LR in types

    def test_decrease_lr_on_low_novelty(self):
        mc = MetaController()
        self._feed_observations(mc, novelty=0.1, surprise=0.05, gci=0.5)
        signals = mc.decide()
        types = [s.signal_type for s in signals]
        assert MetaSignalType.DECREASE_LR in types

    def test_apply_increase_lr(self):
        mc = MetaController()
        rules = PlasticityRuleSet()
        old_lr = rules.stdp.lr
        sig = MetaControlSignal(MetaSignalType.INCREASE_LR, magnitude=0.5)
        mc.apply_signals(rules, [sig])
        assert rules.stdp.lr > old_lr

    def test_apply_decrease_lr(self):
        mc = MetaController()
        rules = PlasticityRuleSet()
        old_lr = rules.stdp.lr
        sig = MetaControlSignal(MetaSignalType.DECREASE_LR, magnitude=0.5)
        mc.apply_signals(rules, [sig])
        assert rules.stdp.lr < old_lr

    def test_apply_widen_window(self):
        mc = MetaController()
        rules = PlasticityRuleSet()
        old_tau = rules.stdp.tau_plus
        sig = MetaControlSignal(MetaSignalType.WIDEN_WINDOW, magnitude=5.0)
        mc.apply_signals(rules, [sig])
        assert rules.stdp.tau_plus > old_tau

    def test_apply_narrow_window(self):
        mc = MetaController()
        rules = PlasticityRuleSet()
        rules.stdp.tau_plus = 50.0
        sig = MetaControlSignal(MetaSignalType.NARROW_WINDOW, magnitude=5.0)
        mc.apply_signals(rules, [sig])
        assert rules.stdp.tau_plus < 50.0

    def test_signal_history(self):
        mc = MetaController()
        self._feed_observations(mc, novelty=0.9)
        mc.decide()
        assert len(mc.signal_history) > 0

    def test_widen_window_on_unstable_gci(self):
        mc = MetaController()
        for i in range(10):
            mc.observe({"novelty": 0.5, "surprise": 0.5, "gci": 0.1 if i % 2 else 0.9})
        types = [s.signal_type for s in mc.decide()]
        assert MetaSignalType.WIDEN_WINDOW in types

    def test_no_op_when_metrics_are_mid_range_and_stable(self):
        mc = MetaController()
        self._feed_observations(mc, novelty=0.5, surprise=0.5, gci=0.5)
        assert [s.signal_type for s in mc.decide()] == [MetaSignalType.NO_OP]

    def test_apply_increase_homeostatic(self):
        mc = MetaController()
        rules = PlasticityRuleSet()
        old = rules.homeostatic.gain_adaptation_rate
        mc.apply_signals(rules, [MetaControlSignal(MetaSignalType.INCREASE_HOMEOSTATIC)])
        assert rules.homeostatic.gain_adaptation_rate > old

    def test_apply_reset_stp(self):
        mc = MetaController()
        rules = PlasticityRuleSet()
        rules.stp.u_base = 0.9
        mc.apply_signals(rules, [MetaControlSignal(MetaSignalType.RESET_STP)])
        assert rules.stp.u_base == STPParams().u_base


# ── RuleEvolver Tests ────────────────────────────────────────────────


class TestRuleEvolver:
    def test_initial_population(self):
        ev = RuleEvolver(population_size=8)
        assert len(ev.population) == 8

    def test_evaluate_fitness(self):
        ev = RuleEvolver()
        rs = PlasticityRuleSet()
        fitness = ev.evaluate_fitness(rs, {"gci": 0.8, "gci_std": 0.05, "mean_surprise": 0.1})
        assert fitness > 0
        assert rs.fitness == fitness

    def test_crossover(self):
        ev = RuleEvolver()
        p1 = PlasticityRuleSet()
        p2 = PlasticityRuleSet()
        p2.stdp.lr = 0.05
        child = ev.crossover(p1, p2)
        assert child.generation == ev.generation + 1

    def test_mutate(self):
        ev = RuleEvolver(mutation_rate=1.0, mutation_scale=0.5)
        original = PlasticityRuleSet()
        mutated = ev.mutate(original)
        v1 = original.to_vector()
        v2 = mutated.to_vector()
        assert not np.allclose(v1, v2)

    def test_evolve(self):
        ev = RuleEvolver(population_size=8)
        for r in ev.population:
            r.fitness = np.random.default_rng(42).random()
        new_pop = ev.evolve()
        assert len(new_pop) == 8
        assert ev.generation == 1

    def test_best(self):
        ev = RuleEvolver(population_size=4)
        ev.population[0].fitness = 0.1
        ev.population[1].fitness = 0.9
        ev.population[2].fitness = 0.5
        ev.population[3].fitness = 0.3
        assert ev.best.fitness == 0.9

    def test_mean_fitness(self):
        ev = RuleEvolver(population_size=4)
        for i, r in enumerate(ev.population):
            r.fitness = float(i) / 3.0
        assert 0.0 < ev.mean_fitness < 1.0


# ── NeuromodulatorState Tests ────────────────────────────────────────


class TestNeuromodulatorState:
    def test_initial_levels(self):
        ns = NeuromodulatorState()
        for nm in NeuromodulatorType:
            assert ns.levels[nm] == 0.5

    def test_update_high_surprise(self):
        ns = NeuromodulatorState()
        ns.update(novelty=0.5, surprise=1.0, gci=0.5)
        assert ns.levels[NeuromodulatorType.DOPAMINE] > 0.5

    def test_update_bounded(self):
        ns = NeuromodulatorState()
        for _ in range(100):
            ns.update(novelty=1.0, surprise=1.0, gci=1.0)
        for nm in NeuromodulatorType:
            assert 0.0 <= ns.levels[nm] <= 1.0

    def test_modulation_factor_lr(self):
        ns = NeuromodulatorState()
        ns.levels[NeuromodulatorType.DOPAMINE] = 0.9
        assert ns.modulation_factor("lr") > 1.0

    def test_modulation_factor_default(self):
        ns = NeuromodulatorState()
        assert ns.modulation_factor("unknown") == 1.0

    def test_modulation_factor_tau(self):
        ns = NeuromodulatorState()
        # 0.8 + 0.4*(1 - ach=0.5) = 1.0
        assert ns.modulation_factor("tau") == pytest.approx(1.0)

    def test_modulation_factor_gain(self):
        ns = NeuromodulatorState()
        # 0.5 + 0.5*ne=0.5 = 0.75
        assert ns.modulation_factor("gain") == pytest.approx(0.75)


# ── MetaPlasticityEngine Tests ───────────────────────────────────────


class TestMetaPlasticityEngine:
    def test_single_step(self):
        engine = MetaPlasticityEngine()
        result = engine.step({"novelty": 0.5, "surprise": 0.1, "gci": 0.7})
        assert result["step"] == 1

    def test_meta_control_fires(self):
        cfg = EngineConfig(meta_interval=5, evolve_interval=1000)
        engine = MetaPlasticityEngine(config=cfg)
        for i in range(10):
            engine.step({"novelty": 0.9, "surprise": 0.5, "gci": 0.5})
        assert engine.rule_changes > 0

    def test_evolution_fires(self):
        cfg = EngineConfig(meta_interval=10, evolve_interval=5, enable_evolution=True)
        engine = MetaPlasticityEngine(config=cfg)
        for i in range(10):
            engine.step({"novelty": 0.5, "surprise": 0.1, "gci": 0.8, "gci_std": 0.02})
        assert engine.evolver.generation > 0

    def test_performance_log(self):
        engine = MetaPlasticityEngine()
        for _ in range(5):
            engine.step({"novelty": 0.5})
        assert len(engine.performance_log) == 5
        assert "stdp_lr" in engine.performance_log[0]

    def test_status(self):
        engine = MetaPlasticityEngine()
        engine.step({"novelty": 0.5})
        st = engine.status()
        assert "step" in st
        assert "rule_changes" in st
        assert "neuromod_dopamine" in st

    def test_neuromodulation_changes_lr(self):
        cfg = EngineConfig(meta_interval=1, evolve_interval=10000, enable_neuromodulation=True)
        engine = MetaPlasticityEngine(config=cfg)
        initial_lr = engine.rules.stdp.lr
        for _ in range(100):
            engine.step({"novelty": 0.9, "surprise": 0.9, "gci": 0.3})
        # LR should have been modulated
        assert engine.rules.stdp.lr != initial_lr


# ── Checkpoint Store Tests (Gap 1) ─────────────────────────────────────


class TestCheckpointStore:
    def test_save_and_count(self):
        store = CheckpointStore()
        rs = PlasticityRuleSet()
        rs.fitness = 0.8
        store.save(rs, step=100, tag="baseline")
        assert store.count == 1

    def test_restore_best(self):
        store = CheckpointStore()
        rs1 = PlasticityRuleSet()
        rs1.fitness = 0.3
        rs2 = PlasticityRuleSet()
        rs2.fitness = 0.9
        store.save(rs1, step=1)
        store.save(rs2, step=2)
        best = store.restore_best()
        assert best is not None
        assert best.fitness == 0.9

    def test_restore_by_tag(self):
        store = CheckpointStore()
        rs = PlasticityRuleSet()
        store.save(rs, step=1, tag="task_A")
        restored = store.restore_by_tag("task_A")
        assert restored is not None

    def test_max_checkpoints(self):
        store = CheckpointStore(max_checkpoints=3)
        for i in range(5):
            store.save(PlasticityRuleSet(), step=i)
        assert store.count == 3

    def test_restore_best_empty_store(self):
        assert CheckpointStore().restore_best() is None

    def test_restore_by_tag_missing(self):
        store = CheckpointStore()
        store.save(PlasticityRuleSet(), step=1, tag="present")
        assert store.restore_by_tag("absent") is None


# ── EWC Protection Tests (Gap 2) ──────────────────────────────────────


class TestEWCProtection:
    def test_no_anchor(self):
        ewc = EWCProtection()
        rs = PlasticityRuleSet()
        assert ewc.penalty(rs) == 0.0

    def test_penalty_after_consolidation(self):
        ewc = EWCProtection(importance=100.0)
        rs = PlasticityRuleSet()
        ewc.consolidate(rs)
        modified = rs.copy()
        modified.stdp.lr = 0.05
        assert ewc.penalty(modified) > 0

    def test_regularise_pulls_back(self):
        ewc = EWCProtection(importance=10000.0)
        rs = PlasticityRuleSet()
        ewc.consolidate(rs)
        modified = rs.copy()
        modified.stdp.lr = 0.1
        regularised = ewc.regularise(modified, max_penalty=0.01)
        assert abs(regularised.stdp.lr - rs.stdp.lr) < abs(modified.stdp.lr - rs.stdp.lr)

    def test_regularise_without_anchor_is_identity(self):
        ewc = EWCProtection()
        rs = PlasticityRuleSet()
        assert ewc.regularise(rs) is rs

    def test_regularise_below_threshold_is_identity(self):
        ewc = EWCProtection()
        rs = PlasticityRuleSet()
        ewc.consolidate(rs)  # anchor == current, so penalty is 0 <= max_penalty
        assert ewc.regularise(rs, max_penalty=10.0) is rs


# ── Curiosity Signal Tests (Gap 3) ────────────────────────────────────


class TestCuriositySignal:
    def test_first_update_high_curiosity(self):
        cs = CuriositySignal()
        c = cs.update(np.array([1.0, 2.0, 3.0]))
        assert c == 1.0

    def test_stable_input_low_curiosity(self):
        cs = CuriositySignal(alpha=0.5)
        state = np.array([1.0, 2.0, 3.0])
        cs.update(state)
        for _ in range(20):
            cs.update(state)
        assert cs.curiosity < 0.01

    def test_changing_input_high_curiosity(self):
        cs = CuriositySignal(alpha=0.1)
        cs.update(np.zeros(5))
        c = cs.update(np.ones(5) * 100)
        assert c > 0.5


# ── Meta-Learning Rate Tests (Gap 4) ──────────────────────────────────


class TestMetaLearningRate:
    def test_positive_delta_increases(self):
        mlr = MetaLearningRate(meta_lr=0.01)
        new = mlr.update(0.1)
        assert new > 0.01

    def test_negative_delta_decreases(self):
        mlr = MetaLearningRate(meta_lr=0.01)
        new = mlr.update(-0.1)
        assert new < 0.01

    def test_bounded(self):
        mlr = MetaLearningRate(meta_lr=0.01, max_meta_lr=0.1)
        for _ in range(100):
            mlr.update(1.0)
        assert mlr.meta_lr <= 0.1


# ── Sleep Phase Tests (Gap 5) ─────────────────────────────────────────


class TestSleepPhase:
    def test_record_and_buffer_size(self):
        sp = SleepPhase()
        sp.record({"novelty": 0.5})
        sp.record({"novelty": 0.8})
        assert sp.buffer_size == 2

    def test_sleep_replays(self):
        sp = SleepPhase(consolidation_rounds=3)
        for i in range(5):
            sp.record({"novelty": float(i) / 4})
        calls = []
        replays = sp.sleep(lambda m: calls.append(m))
        assert replays == 3
        assert len(calls) == 3
        assert not sp.is_sleeping

    def test_empty_buffer_no_replay(self):
        sp = SleepPhase()
        replays = sp.sleep(lambda m: None)
        assert replays == 0


# ── Synaptic Tagging Tests (Gap 6) ────────────────────────────────────


class TestSynapticTagging:
    def test_create_tag(self):
        tm = TaggingModel()
        tag = tm.create_tag(synapse_id=0, strength=0.8, time_ms=100.0)
        assert tag.tag_strength == 0.8
        assert not tag.captured

    def test_decay_reduces_strength(self):
        tm = TaggingModel(tag_decay_rate=0.1)
        tag = tm.create_tag(0, 0.8, 0.0)
        tm.decay_tags(10.0)
        assert tag.tag_strength < 0.8

    def test_consolidate_captures(self):
        tm = TaggingModel(capture_threshold=0.3)
        tm.create_tag(0, 0.5, 0.0)
        captured = tm.consolidate(consolidation_strength=0.8)
        assert captured == 1
        assert tm.tags[0].captured

    def test_consolidate_weak_signal(self):
        tm = TaggingModel(capture_threshold=0.3)
        tm.create_tag(0, 0.5, 0.0)
        captured = tm.consolidate(consolidation_strength=0.2)
        assert captured == 0

    def test_prune_expired(self):
        tm = TaggingModel()
        tm.create_tag(0, 0.001, 0.0)  # below 0.01 threshold
        pruned = tm.prune_expired()
        assert pruned == 1

    def test_active_tags(self):
        tm = TaggingModel()
        tm.create_tag(0, 0.5, 0.0)
        tm.create_tag(1, 0.005, 0.0)  # expired
        assert tm.active_tags == 1


# ── Population Diversity Tests (Gap 7) ────────────────────────────────


class TestPopulationDiversity:
    def test_identical_population_zero(self):
        ev = RuleEvolver(population_size=4)
        assert population_diversity(ev) < 1e-6

    def test_diverse_population(self):
        ev = RuleEvolver(population_size=4, mutation_rate=1.0, mutation_scale=1.0)
        ev.population[0].stdp.lr = 0.001
        ev.population[1].stdp.lr = 0.05
        ev.population[2].stdp.lr = 0.1
        ev.population[3].stdp.tau_plus = 50.0
        assert population_diversity(ev) > 0

    def test_inject_diversity(self):
        ev = RuleEvolver(population_size=4)
        d_before = population_diversity(ev)
        inject_diversity(ev, n_random=2)
        d_after = population_diversity(ev)
        assert d_after >= d_before

    def test_single_member_population_is_zero_diversity(self):
        # A single individual has no pairs to compare, so diversity is 0.
        ev = RuleEvolver(population_size=1)
        assert population_diversity(ev) == 0.0


# ── Context Rule Bank Tests (Gap 8) ───────────────────────────────────


class TestContextRuleBank:
    def test_store_and_switch(self):
        bank = ContextRuleBank()
        rs = PlasticityRuleSet()
        rs.stdp.lr = 0.05
        bank.store("task_A", rs)
        restored = bank.switch("task_A")
        assert restored is not None
        assert restored.stdp.lr == 0.05

    def test_missing_context(self):
        bank = ContextRuleBank()
        assert bank.switch("missing") is None

    def test_contexts_list(self):
        bank = ContextRuleBank()
        bank.store("A", PlasticityRuleSet())
        bank.store("B", PlasticityRuleSet())
        assert set(bank.contexts()) == {"A", "B"}
        assert bank.num_contexts == 2


# ── Fitness Trajectory Tests (Gap 9) ──────────────────────────────────


class TestFitnessTrajectory:
    def test_improving_trend(self):
        ft = FitnessTrajectory(window=10)
        for i in range(20):
            ft.record(float(i) / 19.0)
        assert ft.trend() > 0
        assert ft.is_improving

    def test_declining_trend(self):
        ft = FitnessTrajectory(window=10)
        for i in range(20):
            ft.record(1.0 - float(i) / 19.0)
        assert ft.trend() < 0

    def test_stagnant(self):
        ft = FitnessTrajectory(window=10)
        for _ in range(20):
            ft.record(0.5)
        assert ft.is_stagnant

    def test_best_ever(self):
        ft = FitnessTrajectory()
        ft.record(0.3)
        ft.record(0.9)
        ft.record(0.5)
        assert ft.best_ever == 0.9

    def test_trend_insufficient_history(self):
        assert FitnessTrajectory().trend() == 0.0

    def test_trend_single_point_window_has_no_slope(self):
        # A window of one collapses the x-axis to a single point with zero
        # variance, so no slope can be fit and the trend is 0.
        ft = FitnessTrajectory(window=1)
        ft.record(1.0)
        ft.record(2.0)
        assert ft.trend() == 0.0

    def test_is_stagnant_false_before_window_fills(self):
        ft = FitnessTrajectory(window=20)
        ft.record(0.5)
        assert ft.is_stagnant is False


# ── Rule Constraints Tests (Gap 10) ───────────────────────────────────


class TestRuleConstraints:
    def test_valid_rules(self):
        rc = RuleConstraints()
        rs = PlasticityRuleSet()
        assert rc.is_valid(rs)

    def test_invalid_lr(self):
        rc = RuleConstraints()
        rs = PlasticityRuleSet()
        rs.stdp.lr = 999.0
        assert not rc.is_valid(rs)

    def test_invalid_tau_with_valid_lr(self):
        # lr passes its range check so validation proceeds to the tau check,
        # which an out-of-range tau_plus fails.
        rc = RuleConstraints()
        rs = PlasticityRuleSet()
        rs.stdp.lr = 0.01
        rs.stdp.tau_plus = 1000.0
        assert not rc.is_valid(rs)

    def test_enforce_clamps(self):
        rc = RuleConstraints()
        rs = PlasticityRuleSet()
        rs.stdp.lr = 999.0
        rs.stdp.tau_plus = 0.001
        rs.bitstream.length = 1
        rc.enforce(rs)
        assert rs.stdp.lr <= 0.1
        assert rs.stdp.tau_plus >= 1.0
        assert rs.bitstream.length >= 32
