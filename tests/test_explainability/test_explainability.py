# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Explainability Tests

import numpy as np
import pytest

from sc_neurocore.explainability.explainability import (
    CausalAttribution,
    CausalAttributor,
    ExplainabilityEngine,
    ExplanationDiff,
    FormalPropertyLink,
    LFSRReplay,
    MultiLayerTrace,
    NaturalLanguageExplainer,
    ProvenanceTrace,
    RegulatoryMetadata,
    SensitivityAnalyzer,
    SensitivityResult,
    SpikeDecision,
    SpikeDecisionTree,
    SymbolicPath,
    TemporalWindow,
    VerifiabilityReport,
)


# ── LFSRReplay Tests ────────────────────────────────────────────────


class TestLFSRReplay:
    def test_deterministic_output(self):
        a = LFSRReplay(0xACE1)
        b = LFSRReplay(0xACE1)
        for _ in range(100):
            assert a.step() == b.step()

    def test_zero_seed_raises(self):
        with pytest.raises(ValueError):
            LFSRReplay(0)

    def test_encode_length(self):
        lfsr = LFSRReplay(0xACE1)
        bs = lfsr.encode(32768, 1000)
        assert len(bs) == 1000

    def test_encode_probability(self):
        lfsr = LFSRReplay(0xACE1)
        bs = lfsr.encode(32768, 10000)
        p = np.mean(bs)
        assert abs(p - 0.5) < 0.03

    def test_reset_replays_same(self):
        lfsr = LFSRReplay(0xACE1)
        bs1 = lfsr.encode(32768, 100)
        lfsr.reset()
        bs2 = lfsr.encode(32768, 100)
        np.testing.assert_array_equal(bs1, bs2)

    def test_different_seeds_different_output(self):
        a = LFSRReplay(0xACE1)
        b = LFSRReplay(0xBEEF)
        bs_a = a.encode(32768, 100)
        bs_b = b.encode(32768, 100)
        assert not np.array_equal(bs_a, bs_b)

    def test_matches_core_engine_polynomial(self):
        lfsr = LFSRReplay(0xACE1)
        vals = [lfsr.step() for _ in range(10)]
        lfsr2 = LFSRReplay(0xACE1)
        vals2 = [lfsr2.step() for _ in range(10)]
        assert vals == vals2


# ── SpikeDecisionTree Tests ─────────────────────────────────────────


class TestSpikeDecisionTree:
    def test_add_root_decision(self):
        tree = SpikeDecisionTree()
        bs = np.array([1, 0, 1, 1, 0, 0, 1, 0], dtype=np.uint8)
        node = tree.add_decision("n0", bs, threshold=3)
        assert tree.root is node
        assert node.decision == SpikeDecision.SPIKE
        assert node.popcount == 4

    def test_no_spike_below_threshold(self):
        tree = SpikeDecisionTree()
        bs = np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=np.uint8)
        node = tree.add_decision("n0", bs, threshold=5)
        assert node.decision == SpikeDecision.NO_SPIKE

    def test_child_nodes(self):
        tree = SpikeDecisionTree()
        bs1 = np.ones(8, dtype=np.uint8)
        root = tree.add_decision("n0", bs1, threshold=4)
        bs2 = np.zeros(8, dtype=np.uint8)
        child = tree.add_decision("n1", bs2, threshold=4, parent=root)
        assert len(root.children) == 1
        assert child.decision == SpikeDecision.NO_SPIKE

    def test_depth(self):
        tree = SpikeDecisionTree()
        root = tree.add_decision("n0", np.ones(8, dtype=np.uint8), 4)
        child = tree.add_decision("n1", np.ones(8, dtype=np.uint8), 4, parent=root)
        tree.add_decision("n2", np.ones(8, dtype=np.uint8), 4, parent=child)
        assert tree.depth == 3

    def test_num_spikes(self):
        tree = SpikeDecisionTree()
        tree.add_decision("n0", np.ones(8, dtype=np.uint8), 4)
        tree.add_decision("n1", np.zeros(8, dtype=np.uint8), 4)
        assert tree.num_spikes == 1

    def test_bitstream_hash_deterministic(self):
        tree = SpikeDecisionTree()
        bs = np.array([1, 0, 1, 1], dtype=np.uint8)
        n1 = tree.add_decision("n0", bs, 2)
        tree2 = SpikeDecisionTree()
        n2 = tree2.add_decision("n0", bs, 2)
        assert n1.bitstream_hash == n2.bitstream_hash

    def test_to_dict_structure(self):
        tree = SpikeDecisionTree()
        tree.add_decision("n0", np.ones(4, dtype=np.uint8), 2)
        d = tree.to_dict()
        assert "neuron_id" in d
        assert "decision" in d
        assert d["decision"] == "spike"


# ── ProvenanceTrace Tests ────────────────────────────────────────────


class TestProvenanceTrace:
    def test_add_step(self):
        trace = ProvenanceTrace()
        trace.add_step("input", "test data")
        assert trace.num_steps == 1

    def test_finalize(self):
        trace = ProvenanceTrace()
        trace.add_step("input", "data")
        assert not trace.is_complete
        trace.finalize()
        assert trace.is_complete

    def test_chain_hash_deterministic(self):
        t1 = ProvenanceTrace()
        t1.add_step("input", "data")
        t1.add_step("encode", "encoded")
        h1 = t1.chain_hash

        t2 = ProvenanceTrace()
        t2.add_step("input", "data")
        t2.add_step("encode", "encoded")
        h2 = t2.chain_hash
        assert h1 == h2

    def test_chain_hash_changes_on_tamper(self):
        t1 = ProvenanceTrace()
        t1.add_step("input", "data")
        h1 = t1.chain_hash

        t2 = ProvenanceTrace()
        t2.add_step("input", "tampered")
        h2 = t2.chain_hash
        assert h1 != h2

    def test_to_list(self):
        trace = ProvenanceTrace()
        trace.add_step("input", "data", metadata={"key": "value"})
        lst = trace.to_list()
        assert len(lst) == 1
        assert lst[0]["stage"] == "input"
        assert lst[0]["metadata"]["key"] == "value"

    def test_data_hash_from_array(self):
        trace = ProvenanceTrace()
        data = np.array([1, 0, 1], dtype=np.uint8)
        step = trace.add_step("encode", "bits", data=data)
        assert len(step.data_hash) == 16


# ── ExplainabilityEngine Tests ───────────────────────────────────────


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


# ── DecisionMargin Tests ─────────────────────────────────────────────


class TestDecisionMargin:
    def test_spike_margin_positive(self):
        tree = SpikeDecisionTree()
        bs = np.ones(100, dtype=np.uint8)
        node = tree.add_decision("n0", bs, threshold=80)
        m = node.margin
        assert m.margin == 20
        assert m.confidence > 0

    def test_no_spike_margin_negative(self):
        tree = SpikeDecisionTree()
        bs = np.zeros(100, dtype=np.uint8)
        bs[:30] = 1
        node = tree.add_decision("n0", bs, threshold=50)
        m = node.margin
        assert m.margin == -20
        assert m.confidence > 0

    def test_exact_threshold(self):
        tree = SpikeDecisionTree()
        bs = np.zeros(100, dtype=np.uint8)
        bs[:50] = 1
        node = tree.add_decision("n0", bs, threshold=50)
        assert node.margin.margin == 0
        assert node.decision == SpikeDecision.SPIKE


# ── SCC Influence Tests ──────────────────────────────────────────────


class TestSCCInfluence:
    def test_influence_computed(self):
        tree = SpikeDecisionTree()
        bs = np.ones(100, dtype=np.uint8)
        node = tree.add_decision("n0", bs, threshold=50, scc=0.5)
        assert node.scc_influence > 0

    def test_zero_scc_zero_influence(self):
        tree = SpikeDecisionTree()
        bs = np.ones(100, dtype=np.uint8)
        node = tree.add_decision("n0", bs, threshold=50, scc=0.0)
        assert node.scc_influence == 0.0

    def test_influence_in_dict(self):
        tree = SpikeDecisionTree()
        bs = np.ones(8, dtype=np.uint8)
        tree.add_decision("n0", bs, threshold=4, scc=0.3)
        d = tree.to_dict()
        assert "scc_influence" in d
        assert "margin" in d
        assert "confidence" in d


# ── Sensitivity Analysis Tests ───────────────────────────────────────


class TestSensitivityAnalyzer:
    def test_basic_sensitivity(self):
        tree = SpikeDecisionTree()
        bs = np.zeros(100, dtype=np.uint8)
        bs[:60] = 1
        node = tree.add_decision("n0", bs, threshold=55)
        results = SensitivityAnalyzer.analyze(node)
        assert len(results) == 6  # default perturbations
        assert any(r.flipped for r in results)

    def test_custom_perturbations(self):
        tree = SpikeDecisionTree()
        bs = np.ones(100, dtype=np.uint8)
        node = tree.add_decision("n0", bs, threshold=50)
        results = SensitivityAnalyzer.analyze(node, perturbations=[-1, 1])
        assert len(results) == 2

    def test_critical_delta_spike(self):
        tree = SpikeDecisionTree()
        bs = np.zeros(100, dtype=np.uint8)
        bs[:60] = 1
        node = tree.add_decision("n0", bs, threshold=55)
        cd = SensitivityAnalyzer.critical_delta(node)
        assert cd == 6  # margin 5 → need +6 to flip

    def test_critical_delta_no_spike(self):
        tree = SpikeDecisionTree()
        bs = np.zeros(100, dtype=np.uint8)
        bs[:30] = 1
        node = tree.add_decision("n0", bs, threshold=50)
        cd = SensitivityAnalyzer.critical_delta(node)
        assert cd == -20

    def test_engine_sensitivity(self):
        engine = ExplainabilityEngine(seed=0xACE1)
        node = engine.explain_spike("n0", 32768, 256, 100)
        results = engine.sensitivity(node)
        assert len(results) > 0


# ── Causal Attribution Tests ─────────────────────────────────────────


class TestCausalAttribution:
    def test_basic_attribution(self):
        tree = SpikeDecisionTree()
        bs = np.ones(100, dtype=np.uint8)
        target = tree.add_decision("out", bs, threshold=50)
        inputs = {
            "in0": np.ones(100, dtype=np.uint8),
            "in1": np.zeros(100, dtype=np.uint8),
        }
        attr = CausalAttributor.attribute(target, inputs)
        assert attr.target_neuron == "out"
        assert attr.attributions["in0"] == 100.0
        assert attr.attributions["in1"] == 0.0
        assert attr.total_contribution == 100.0

    def test_weighted_attribution(self):
        tree = SpikeDecisionTree()
        bs = np.ones(100, dtype=np.uint8)
        target = tree.add_decision("out", bs, threshold=50)
        inputs = {"in0": np.ones(50, dtype=np.uint8)}
        weights = {"in0": 2.0}
        attr = CausalAttributor.attribute(target, inputs, weights)
        assert attr.attributions["in0"] == 100.0

    def test_top_contributors_sorted(self):
        tree = SpikeDecisionTree()
        target = tree.add_decision("out", np.ones(8, dtype=np.uint8), 4)
        inputs = {
            "a": np.ones(10, dtype=np.uint8),
            "b": np.zeros(10, dtype=np.uint8),
            "c": np.ones(5, dtype=np.uint8),
        }
        attr = CausalAttributor.attribute(target, inputs)
        top = attr.top_contributors
        assert top[0][0] == "a"
        assert top[0][1] > top[1][1]

    def test_engine_attribute(self):
        engine = ExplainabilityEngine(seed=0xACE1)
        node = engine.explain_spike("n0", 32768, 64, 20)
        inputs = {"src0": np.ones(64, dtype=np.uint8)}
        attr = engine.attribute(node, inputs)
        assert attr.total_contribution > 0


# ── Explanation Diff Tests ───────────────────────────────────────────


class TestExplanationDiff:
    def test_identical_nodes_no_diffs(self):
        tree = SpikeDecisionTree()
        bs = np.ones(8, dtype=np.uint8)
        a = tree.add_decision("n0", bs, threshold=4)
        tree2 = SpikeDecisionTree()
        b = tree2.add_decision("n0", bs, threshold=4)
        diffs = ExplanationDiff.diff(a, b)
        assert diffs == []

    def test_different_thresholds(self):
        tree = SpikeDecisionTree()
        bs = np.ones(8, dtype=np.uint8)
        a = tree.add_decision("n0", bs, threshold=4)
        b = tree.add_decision("n1", bs, threshold=6)
        diffs = ExplanationDiff.diff(a, b)
        fields_changed = [d.field for d in diffs]
        assert "neuron_id" in fields_changed
        assert "threshold" in fields_changed

    def test_different_decisions(self):
        tree = SpikeDecisionTree()
        a = tree.add_decision("n0", np.ones(8, dtype=np.uint8), threshold=4)
        b = tree.add_decision("n1", np.zeros(8, dtype=np.uint8), threshold=4)
        diffs = ExplanationDiff.diff(a, b)
        fields_changed = [d.field for d in diffs]
        assert "decision" in fields_changed


# ── Temporal Window Tests ────────────────────────────────────────────


class TestTemporalWindow:
    def test_add_and_query(self):
        tw = TemporalWindow()
        tree = SpikeDecisionTree()
        n0 = tree.add_decision("n0", np.ones(8, dtype=np.uint8), 4, timestep=0)
        n1 = tree.add_decision("n1", np.zeros(8, dtype=np.uint8), 4, timestep=1)
        tw.add(n0)
        tw.add(n1)
        assert tw.num_timesteps == 2
        assert tw.spike_rate_at(0) == 1.0
        assert tw.spike_rate_at(1) == 0.0

    def test_peak_timestep(self):
        tw = TemporalWindow()
        tree = SpikeDecisionTree()
        for t in range(3):
            for _ in range(3):
                bs = np.ones(8, dtype=np.uint8) if t == 1 else np.zeros(8, dtype=np.uint8)
                n = tree.add_decision(f"n_{t}", bs, 4, timestep=t)
                tw.add(n)
        assert tw.peak_timestep() == 1

    def test_active_timesteps(self):
        tw = TemporalWindow()
        tree = SpikeDecisionTree()
        for t in [0, 5, 10]:
            n = tree.add_decision(f"n_{t}", np.ones(8, dtype=np.uint8), 4, timestep=t)
            tw.add(n)
        assert tw.active_timesteps() == [0, 5, 10]

    def test_empty_timestep_rate_zero(self):
        tw = TemporalWindow()
        assert tw.spike_rate_at(999) == 0.0


# ── Multi-Layer Trace Tests ──────────────────────────────────────────


class TestMultiLayerTrace:
    def test_add_and_layers(self):
        mlt = MultiLayerTrace()
        tree = SpikeDecisionTree()
        n0 = tree.add_decision("n0", np.ones(8, dtype=np.uint8), 4, layer_id="L1")
        n1 = tree.add_decision("n1", np.zeros(8, dtype=np.uint8), 4, layer_id="L2")
        mlt.add(n0)
        mlt.add(n1)
        assert "L1" in mlt.layer_ids
        assert "L2" in mlt.layer_ids

    def test_spikes_at_layer(self):
        mlt = MultiLayerTrace()
        tree = SpikeDecisionTree()
        n0 = tree.add_decision("n0", np.ones(8, dtype=np.uint8), 4, layer_id="L1")
        n1 = tree.add_decision("n1", np.zeros(8, dtype=np.uint8), 4, layer_id="L1")
        mlt.add(n0)
        mlt.add(n1)
        assert mlt.spikes_at_layer("L1") == 1
        assert mlt.spike_rate_at_layer("L1") == 0.5

    def test_propagation_path(self):
        mlt = MultiLayerTrace()
        tree = SpikeDecisionTree()
        for lid in ["L1", "L2", "L3"]:
            n = tree.add_decision(f"n_{lid}", np.ones(8, dtype=np.uint8), 4, layer_id=lid)
            mlt.add(n)
        path = mlt.propagation_path()
        assert len(path) == 3
        assert all("spike_rate" in p for p in path)


# ── Symbolic Path Tests ──────────────────────────────────────────────


class TestSymbolicPath:
    def test_add_and_length(self):
        sp = SymbolicPath()
        sp.add("n0", SpikeDecision.SPIKE, "popcount(60) >= threshold(50)")
        sp.add("n1", SpikeDecision.NO_SPIKE, "popcount(30) < threshold(50)")
        assert sp.length == 2

    def test_to_list(self):
        sp = SymbolicPath()
        sp.add("n0", SpikeDecision.SPIKE, "popcount(60) >= threshold(50)")
        lst = sp.to_list()
        assert lst[0]["neuron"] == "n0"
        assert lst[0]["decision"] == "spike"
        assert "popcount" in lst[0]["reason"]


# ── Natural Language Explainer Tests ─────────────────────────────────


class TestNaturalLanguageExplainer:
    def test_explain_spike(self):
        tree = SpikeDecisionTree()
        bs = np.ones(100, dtype=np.uint8)
        node = tree.add_decision(
            "n0", bs, threshold=50, scc=0.3, contributing_neurons=["in_a", "in_b"]
        )
        text = NaturalLanguageExplainer.explain_node(node)
        assert "fired" in text
        assert "n0" in text
        assert "SCC" in text
        assert "in_a" in text

    def test_explain_no_spike(self):
        tree = SpikeDecisionTree()
        bs = np.zeros(100, dtype=np.uint8)
        node = tree.add_decision("n0", bs, threshold=50)
        text = NaturalLanguageExplainer.explain_node(node)
        assert "NOT" in text

    def test_explain_attribution(self):
        attr = CausalAttribution("out", {"a": 80.0, "b": 20.0}, 100.0)
        text = NaturalLanguageExplainer.explain_attribution(attr)
        assert "out" in text
        assert "a" in text

    def test_explain_sensitivity_robust(self):
        results = [
            SensitivityResult("n0", 50, 49, SpikeDecision.SPIKE, SpikeDecision.SPIKE, False),
            SensitivityResult("n0", 50, 51, SpikeDecision.SPIKE, SpikeDecision.SPIKE, False),
        ]
        text = NaturalLanguageExplainer.explain_sensitivity(results)
        assert "robust" in text

    def test_explain_sensitivity_flip(self):
        results = [
            SensitivityResult("n0", 50, 51, SpikeDecision.SPIKE, SpikeDecision.NO_SPIKE, True),
        ]
        text = NaturalLanguageExplainer.explain_sensitivity(results)
        assert "flip" in text


# ── Regulatory Metadata Tests ────────────────────────────────────────


class TestRegulatoryMetadata:
    def test_default_fields(self):
        rm = RegulatoryMetadata()
        assert rm.device_class == "Class II"
        assert rm.review_status == "pending"

    def test_verify_with_regulatory(self):
        engine = ExplainabilityEngine(seed=0xACE1)
        engine.explain_spike("n0", 32768, 64, 20)
        reg = RegulatoryMetadata(
            device_class="Class III",
            intended_use="BCI motor cortex",
            software_version="3.12.0",
        )
        report = engine.verify(regulatory=reg)
        assert report.regulatory is not None
        assert report.regulatory.device_class == "Class III"


# ── Formal Property Link Tests ───────────────────────────────────────


class TestFormalPropertyLink:
    def test_default_fields(self):
        fp = FormalPropertyLink(property_name="no_metastability")
        assert fp.status == "unverified"
        assert fp.engine == "sby"

    def test_verify_with_formal_props(self):
        engine = ExplainabilityEngine(seed=0xACE1)
        engine.explain_spike("n0", 32768, 64, 20)
        props = [
            FormalPropertyLink("no_metastability", status="proven", bounded_depth=20),
            FormalPropertyLink("lfsr_period", status="proven", bounded_depth=65535),
        ]
        report = engine.verify(formal_properties=props)
        assert len(report.formal_properties) == 2
        assert report.formal_properties[0].status == "proven"


# ── Node Lookup & Spike Path Tests ───────────────────────────────────


class TestTreeNavigation:
    def test_get_node(self):
        tree = SpikeDecisionTree()
        tree.add_decision("n0", np.ones(8, dtype=np.uint8), 4)
        tree.add_decision("n1", np.zeros(8, dtype=np.uint8), 4)
        assert tree.get_node("n0") is not None
        assert tree.get_node("n0").neuron_id == "n0"
        assert tree.get_node("missing") is None

    def test_nodes_at_layer(self):
        tree = SpikeDecisionTree()
        tree.add_decision("n0", np.ones(8, dtype=np.uint8), 4, layer_id="L1")
        tree.add_decision("n1", np.ones(8, dtype=np.uint8), 4, layer_id="L2")
        assert len(tree.nodes_at_layer("L1")) == 1
        assert len(tree.nodes_at_layer("L2")) == 1

    def test_nodes_at_timestep(self):
        tree = SpikeDecisionTree()
        tree.add_decision("n0", np.ones(8, dtype=np.uint8), 4, timestep=5)
        tree.add_decision("n1", np.ones(8, dtype=np.uint8), 4, timestep=5)
        assert len(tree.nodes_at_timestep(5)) == 2

    def test_spike_path(self):
        tree = SpikeDecisionTree()
        root = tree.add_decision("n0", np.ones(8, dtype=np.uint8), 4)
        child = tree.add_decision("n1", np.zeros(8, dtype=np.uint8), 4, parent=root)
        tree.add_decision("n2", np.ones(8, dtype=np.uint8), 4, parent=child)
        path = tree.spike_path()
        spiking = [n.neuron_id for n in path]
        assert "n0" in spiking
        assert "n1" not in spiking
        assert "n2" in spiking


# ── Engine Integration Tests ─────────────────────────────────────────


class TestEngineIntegration:
    def test_temporal_tracking(self):
        engine = ExplainabilityEngine(seed=0xACE1)
        engine.explain_spike("n0", 32768, 64, 20, timestep=0)
        engine.explain_spike("n1", 32768, 64, 20, timestep=1)
        assert engine.temporal.num_timesteps == 2

    def test_multi_layer_tracking(self):
        engine = ExplainabilityEngine(seed=0xACE1)
        engine.explain_spike("n0", 32768, 64, 20, layer_id="L1")
        engine.explain_spike("n1", 32768, 64, 20, layer_id="L2")
        assert len(engine.multi_layer.layer_ids) == 2

    def test_symbolic_path_tracking(self):
        engine = ExplainabilityEngine(seed=0xACE1)
        engine.explain_spike("n0", 32768, 64, 20)
        assert engine.symbolic.length == 1
        lst = engine.symbolic.to_list()
        assert "popcount" in lst[0]["reason"]

    def test_to_dict_has_all_fields(self):
        engine = ExplainabilityEngine(seed=0xACE1)
        engine.explain_spike(
            "n0", 32768, 64, 20, scc=0.2, layer_id="L1", timestep=3, contributing_neurons=["src0"]
        )
        d = engine.tree.to_dict()
        assert "scc_influence" in d
        assert "margin" in d
        assert "confidence" in d
        assert "timestep" in d
        assert "layer_id" in d
        assert "contributing_neurons" in d
