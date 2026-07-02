# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Explainability Validation Tests

"""Validation and edge-case coverage for deterministic explainability replay."""

from __future__ import annotations

from typing import cast

import numpy as np
import numpy.typing as npt
import pytest

from sc_neurocore.explainability.explainability import (
    ExplainabilityEngine,
    MultiLayerTrace,
    SpikeDecisionTree,
)


def test_empty_decision_tree_reports_identity_values() -> None:
    """Empty decision trees expose deterministic identity values."""
    tree = SpikeDecisionTree()

    assert tree.depth == 0
    assert tree.to_dict() == {}


def test_decision_node_leaf_status_tracks_child_edges() -> None:
    """Parent and child nodes report leaf status from their actual edges."""
    tree = SpikeDecisionTree()
    root = tree.add_decision("root", np.ones(8, dtype=np.uint8), threshold=4)
    child = tree.add_decision("child", np.zeros(8, dtype=np.uint8), threshold=4, parent=root)

    assert root.is_leaf is False
    assert child.is_leaf is True


def test_empty_multi_layer_rate_is_zero() -> None:
    """Missing layers return a zero spike rate instead of raising."""
    trace = MultiLayerTrace()

    assert trace.spike_rate_at_layer("missing") == 0.0


def test_verify_reports_mismatched_replay_evidence() -> None:
    """Public verification fails closed when stored replay evidence is tampered."""
    engine = ExplainabilityEngine(seed=0xACE1)
    node = engine.explain_spike("n0", threshold_q16=32768, bitstream_length=32, spike_threshold_count=16)
    stored = cast(npt.NDArray[np.uint8], engine._replayed_bitstreams[node.neuron_id])
    tampered = stored.copy()
    tampered[0] = np.uint8(1 - int(tampered[0]))

    engine._replayed_bitstreams[node.neuron_id] = tampered

    report = engine.verify()

    assert report.replay_matches is False
    assert report.all_hashes_valid is False


def test_explain_spike_rejects_non_positive_bitstream_length_before_trace() -> None:
    """Invalid stream lengths are rejected before provenance is mutated."""
    engine = ExplainabilityEngine(seed=0xACE1)

    with pytest.raises(ValueError, match="bitstream_length must be positive"):
        engine.explain_spike("n0", threshold_q16=32768, bitstream_length=0, spike_threshold_count=0)

    assert engine.provenance.num_steps == 0
    assert engine.tree.num_nodes == 0


def test_explain_spike_rejects_threshold_outside_q16_range_before_trace() -> None:
    """Replay thresholds must remain inside the unsigned Q16 domain."""
    engine = ExplainabilityEngine(seed=0xACE1)

    with pytest.raises(ValueError, match="threshold_q16 must be an integer in \\[0, 65535\\]"):
        engine.explain_spike("n0", threshold_q16=65536, bitstream_length=8, spike_threshold_count=4)

    assert engine.provenance.num_steps == 0
    assert engine.tree.num_nodes == 0


def test_explain_spike_rejects_spike_threshold_outside_stream_bounds() -> None:
    """Spike-count thresholds must be reachable within the replayed stream."""
    engine = ExplainabilityEngine(seed=0xACE1)

    with pytest.raises(ValueError, match="spike_threshold_count must be between 0 and bitstream_length"):
        engine.explain_spike("n0", threshold_q16=32768, bitstream_length=8, spike_threshold_count=9)

    assert engine.provenance.num_steps == 0
    assert engine.tree.num_nodes == 0


def test_replay_bitstream_rejects_invalid_external_parameters() -> None:
    """External bitstream replay applies the same Q16 and length guardrails."""
    engine = ExplainabilityEngine(seed=0xACE1)

    with pytest.raises(ValueError, match="threshold_q16 must be an integer in \\[0, 65535\\]"):
        engine.replay_bitstream(threshold_q16=-1, length=8)

    with pytest.raises(ValueError, match="length must be positive"):
        engine.replay_bitstream(threshold_q16=32768, length=0)
