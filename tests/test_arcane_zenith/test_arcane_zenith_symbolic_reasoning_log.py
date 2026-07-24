# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSymbolicReasoningLog from former test_arcane_zenith.py

"""Focused suite: TestSymbolicReasoningLog from former test_arcane_zenith.py."""

from __future__ import annotations

from tests.test_arcane_zenith.arcane_zenith_support import *  # noqa: F403


class TestSymbolicReasoningLog:
    def test_reasoning_log_has_stable_schema_and_fields(self):
        core = create_arcane_neuron_with_zenith_plasticity(backend="torch")
        core.step(1.5)
        log = core.export_symbolic_reasoning_log()

        assert log["schema_version"] == "sc-neurocore.arcane-zenith.symbolic-reasoning-log.v1"
        assert isinstance(log["tick"], int)
        assert log["novelty_level"] in {"low", "medium", "high"}
        assert log["novelty_shift"] in {"rising", "falling", "steady"}
        assert log["confidence_trend"] in {"rising", "falling", "steady"}
        assert log["identity_shift"] in {"stable", "drifting"}
        assert log["adaptation_regime"] in {"conservative", "aggressive"}

    def test_reasoning_log_tick_progresses_after_steps(self):
        core = create_arcane_neuron_with_zenith_plasticity(backend="torch")
        core.step(0.5)
        first = core.export_symbolic_reasoning_log()
        core.step(0.5)
        second = core.export_symbolic_reasoning_log()
        assert second["tick"] > first["tick"]

    def test_episode_trace_embeds_symbolic_log(self):
        core = create_arcane_neuron_with_zenith_plasticity(backend="torch")
        result = core.run_meta_learning_episode([0.2, 0.4, 0.6], reset_before=True)
        for row in result["trace"]:
            assert "symbolic_log" in row
            embedded = row["symbolic_log"]
            assert (
                embedded["schema_version"] == "sc-neurocore.arcane-zenith.symbolic-reasoning-log.v1"
            )
