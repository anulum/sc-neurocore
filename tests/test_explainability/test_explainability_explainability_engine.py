# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestExplainabilityEngine from former test_explainability.py

"""Focused suite: TestExplainabilityEngine from former test_explainability.py."""

from __future__ import annotations

from explainability_support import *  # noqa: F403


class TestExplainabilityEngine:
    def test_explain_spike_records_decision(self):
        engine = ExplainabilityEngine(seed=0xACE1)
        node = engine.explain_spike(
            neuron_id="V1_0",
            threshold_q16=32768,
            bitstream_length=256,
            spike_threshold_count=100,
        )
        assert node.decision in (SpikeDecision.SPIKE, SpikeDecision.NO_SPIKE)
        assert node.bitstream_length == 256

    def test_provenance_has_three_stages(self):
        engine = ExplainabilityEngine(seed=0xACE1)
        engine.explain_spike("V1_0", 32768, 256, 100)
        assert engine.provenance.num_steps == 3
        stages = [s["stage"] for s in engine.provenance.to_list()]
        assert stages == ["input", "encoding", "decision"]

    def test_verify_produces_report(self):
        engine = ExplainabilityEngine(seed=0xACE1)
        engine.explain_spike("V1_0", 32768, 256, 100)
        report = engine.verify()
        assert isinstance(report, VerifiabilityReport)
        assert report.replay_matches is True
        assert report.num_steps == 3

    def test_replay_bitstream_deterministic(self):
        engine = ExplainabilityEngine(seed=0xACE1)
        bs1 = engine.replay_bitstream(32768, 100)
        bs2 = engine.replay_bitstream(32768, 100)
        np.testing.assert_array_equal(bs1, bs2)

    def test_multiple_neurons(self):
        engine = ExplainabilityEngine(seed=0x1234)
        engine.explain_spike("n0", 32768, 128, 50)
        engine.explain_spike("n1", 16384, 128, 30)
        assert engine.tree.num_nodes == 2
        assert engine.provenance.num_steps == 6

    def test_chain_hash_stable(self):
        engine = ExplainabilityEngine(seed=0xACE1)
        engine.explain_spike("n0", 32768, 64, 20)
        h1 = engine.provenance.chain_hash
        assert len(h1) == 64

    def test_decision_tree_serialisable(self):
        engine = ExplainabilityEngine(seed=0xACE1)
        engine.explain_spike("n0", 32768, 64, 20)
        d = engine.tree.to_dict()
        assert d["neuron_id"] == "n0"
        assert "bitstream_hash" in d
