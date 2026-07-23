# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestMetaPlasticityIntegration from former test_autonomous_learning.py

"""Focused suite: TestMetaPlasticityIntegration from former test_autonomous_learning.py."""

from __future__ import annotations

from autonomous_learning_support import *  # noqa: F403

class TestMetaPlasticityIntegration:
    def test_meta_plasticity_engine_step(self):
        engine = MetaPlasticityEngine(config=EngineConfig(enable_evolution=False))
        metrics = {
            "novelty": 0.8,
            "surprise": 0.1,
            "gci": 0.7,
            "gci_std": 0.05,
            "mean_rate_hz": 4.5,
        }

        res = engine.step(metrics)
        assert res["step"] == 1
        assert "current_rules" in res
        assert engine.neuromod.levels is not None
