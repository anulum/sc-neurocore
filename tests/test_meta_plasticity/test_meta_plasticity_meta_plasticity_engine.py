# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestMetaPlasticityEngine from former test_meta_plasticity.py

"""Focused suite: TestMetaPlasticityEngine from former test_meta_plasticity.py."""

from __future__ import annotations

from meta_plasticity_support import *  # noqa: F403


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
