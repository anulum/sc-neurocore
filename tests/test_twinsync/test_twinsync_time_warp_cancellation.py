# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Time-warp rollback cancels, re-executes and detects lag as its names say

"""Hold the time-warp engine's rollback, anti-messages, starvation check and state deltas to their names.

Anti-messages were queued and then applied like ordinary events: they cancelled nothing, advanced the target's
clock and counted as processed. A rollback left the rolled-back events in the processed list and never ran them
again. The starvation check compared each node with the global virtual time, which is never above a node's own
time, so it could not report anything. A state delta of a two-dimensional array repeated whole rows once per
changed element. These tests hold the cancellation of a rolled-back node's sent events, the re-execution of its
rolled-back inputs, lag measured from the furthest node, and an element-wise delta that rebuilds the state.
"""

from __future__ import annotations

import numpy as np
import pytest

from sc_neurocore.digital_twin.twinsync import DeltaCheckpoint, EventType, TimeWarpEngine, TwinEvent


def _drain(engine: TimeWarpEngine) -> list[TwinEvent]:
    """Process every queued event and return what was dequeued, in order."""
    out = []
    while (event := engine.process_next()) is not None:
        out.append(event)
    return out


def test_every_injected_event_gets_its_own_id() -> None:
    """The engine numbers events without an id, so an anti-message can name the one it cancels."""
    engine = TimeWarpEngine(2)
    a, b = TwinEvent(10, target_node=0), TwinEvent(20, target_node=1)
    engine.inject_event(a)
    engine.inject_event(b)
    assert a.event_id >= 0 and b.event_id >= 0 and a.event_id != b.event_id
    kept = TwinEvent(30, target_node=0, event_id=77)
    engine.inject_event(kept)
    assert kept.event_id == 77


def test_a_rollback_cancels_the_node_s_queued_sent_events() -> None:
    """An event the rolled-back node sent after the rollback time, still queued, is cancelled and never applied."""
    engine = TimeWarpEngine(2, checkpoint_interval_ns=1)
    engine.inject_event(TwinEvent(200, target_node=0))
    engine.process_next()
    sent = TwinEvent(300, source_node=0, target_node=1)
    engine.inject_event(sent)
    engine.inject_event(TwinEvent(100, target_node=0))
    _drain(engine)
    assert sent.cancelled is True
    assert engine.nodes[1].processed_events == 0
    assert engine.nodes[1].local_virtual_time_ns == 0


def test_an_anti_message_annihilates_a_processed_event_and_rolls_its_target_back() -> None:
    """A processed event the rolled-back node sent is undone at its receiver; the anti-message itself is not applied."""
    engine = TimeWarpEngine(2, checkpoint_interval_ns=1)
    sent = TwinEvent(300, source_node=0, target_node=1)
    engine.inject_event(sent)
    engine.inject_event(TwinEvent(400, target_node=0))
    _drain(engine)
    assert engine.nodes[1].processed_events == 1
    engine.inject_event(TwinEvent(100, target_node=0))
    dequeued = _drain(engine)

    antis = [e for e in dequeued if e.event_type is EventType.ANTI_MESSAGE]
    assert [a.payload["cancels"] for a in antis] == [sent.event_id]
    assert antis[0] in engine.anti_messages
    assert sent not in engine.processed
    assert all(e.event_type is not EventType.ANTI_MESSAGE for e in engine.processed)
    assert engine.nodes[1].processed_events == 0
    assert engine.nodes[1].local_virtual_time_ns < 300
    assert engine.nodes[1].rollback_count == 1


def test_a_rollback_runs_the_rolled_back_inputs_again_in_time_order() -> None:
    """Events the node processed after its restored time return to the queue and run again after the straggler."""
    engine = TimeWarpEngine(1, checkpoint_interval_ns=1)
    for t in (200, 300):
        engine.inject_event(TwinEvent(t, target_node=0))
    _drain(engine)
    engine.inject_event(TwinEvent(250, target_node=0))
    _drain(engine)
    assert [e.virtual_time_ns for e in engine.processed] == [200, 250, 300]
    assert engine.nodes[0].processed_events == 3
    assert engine.nodes[0].local_virtual_time_ns == 300
    assert engine.verify_causal_order() == []


def test_starvation_is_lag_behind_the_furthest_node() -> None:
    """A node more than the threshold behind the furthest-ahead node is starving; exactly the threshold is not."""
    engine = TimeWarpEngine(3)
    engine.nodes[0].local_virtual_time_ns = 50_000
    engine.nodes[1].local_virtual_time_ns = 49_000
    engine.nodes[2].local_virtual_time_ns = 0
    assert engine.detect_starvation(threshold_ns=1_000) == [2]
    assert engine.detect_starvation(threshold_ns=50_000) == []


def test_a_two_dimensional_delta_stores_changed_elements_and_rebuilds_the_state() -> None:
    """Flat element indices and their values, no repeated rows; applying the delta to the base gives the new state."""
    base = np.zeros((4, 5))
    new = base.copy()
    new[2, 1] = 7.0
    new[2, 3] = -1.5
    delta = DeltaCheckpoint.compute_delta(base, new, 0, 1, 1000, 0)
    assert delta.num_changes == 2
    np.testing.assert_array_equal(delta.changed_indices, [11, 13])
    np.testing.assert_array_equal(delta.changed_values, [7.0, -1.5])
    assert delta.size_bytes == delta.changed_indices.nbytes + delta.changed_values.nbytes
    assert delta.state_shape == (4, 5)
    np.testing.assert_array_equal(delta.apply(base), new)
    np.testing.assert_array_equal(base, np.zeros((4, 5)))


def test_a_delta_needs_two_states_of_one_shape() -> None:
    """States of different shapes have no element-wise diff, and a delta applies only to its own shape."""
    with pytest.raises(ValueError, match="shape"):
        DeltaCheckpoint.compute_delta(np.zeros(3), np.zeros(4), 0, 1, 0, 0)
    delta = DeltaCheckpoint.compute_delta(np.zeros(3), np.ones(3), 0, 1, 0, 0)
    with pytest.raises(ValueError, match="shape"):
        delta.apply(np.zeros(4))


def test_an_anti_message_for_no_processed_event_changes_nothing() -> None:
    """An anti-message whose event is not in the processed list (never run, or already undone) is only recorded."""
    engine = TimeWarpEngine(2)
    engine.inject_event(TwinEvent(100, target_node=1))
    engine.process_next()
    anti = TwinEvent(50, event_type=EventType.ANTI_MESSAGE, target_node=1, payload={"cancels": 999})
    engine.inject_event(anti)
    assert engine.process_next() is anti
    assert [e.virtual_time_ns for e in engine.processed] == [100]
    assert engine.nodes[1].rollback_count == 0
    assert engine.nodes[1].local_virtual_time_ns == 100


def test_an_anti_message_for_an_event_of_an_unknown_node_only_removes_it() -> None:
    """A processed entry naming a node the engine does not have is removed without a rollback."""
    engine = TimeWarpEngine(1)
    stray = TwinEvent(100, target_node=5, event_id=42)
    engine.processed = [stray]
    engine.inject_event(TwinEvent(100, event_type=EventType.ANTI_MESSAGE, payload={"cancels": 42}))
    engine.process_next()
    assert engine.processed == []
    assert engine.total_rollbacks == 0


def test_an_engine_without_nodes_reports_no_starvation() -> None:
    """With no node there is no lag to report."""
    assert TimeWarpEngine(0).detect_starvation(threshold_ns=0) == []


def test_a_checkpoint_bookmark_survives_another_node_s_rollback() -> None:
    """Undoing one node's events must not shift what a later rollback of another node undoes (reader's probe)."""
    engine = TimeWarpEngine(2, checkpoint_interval_ns=2)
    for vt, node in [(10, 1), (20, 1), (30, 1), (40, 0), (50, 0), (60, 0), (70, 0)]:
        engine.inject_event(TwinEvent(vt, source_node=9, target_node=node))
    _drain(engine)
    engine.inject_event(TwinEvent(5, source_node=9, target_node=1))
    _drain(engine)
    engine.inject_event(TwinEvent(55, source_node=9, target_node=0))
    engine.process_next()
    assert [e.virtual_time_ns for e in engine.processed if e.target_node == 0] == [40, 50, 55]
    assert sorted(e.virtual_time_ns for e in engine.event_queue) == [60, 70]
    _drain(engine)
    assert [e.virtual_time_ns for e in engine.processed if e.target_node == 0] == [
        40,
        50,
        55,
        60,
        70,
    ]


def test_an_anti_message_cancels_a_positive_its_receiver_has_re_queued() -> None:
    """When the receiver's own rollback put the named event back in the queue, the anti-message cancels it there."""
    engine = TimeWarpEngine(2, checkpoint_interval_ns=1000)
    sent = TwinEvent(100, source_node=0, target_node=1)
    engine.inject_event(sent)
    engine.process_next()
    engine.inject_event(TwinEvent(150, source_node=9, target_node=0))
    engine.process_next()
    engine.inject_event(TwinEvent(50, source_node=9, target_node=0))
    engine.process_next()
    engine.inject_event(TwinEvent(60, source_node=9, target_node=1))
    engine.process_next()
    assert sent in engine.event_queue and not sent.cancelled
    _drain(engine)
    assert sent.cancelled
    assert sent not in engine.processed
    assert engine.nodes[1].processed_events == 1


def test_a_checkpoint_saved_outside_the_engine_undoes_by_virtual_time() -> None:
    """A checkpoint the engine did not bookmark falls back to undoing the node's events after its time."""
    engine = TimeWarpEngine(1, checkpoint_interval_ns=1000)
    for vt in (100, 200, 300):
        engine.inject_event(TwinEvent(vt, target_node=0))
    _drain(engine)
    engine.checkpoint_mgr.save(0, 200)
    engine.inject_event(TwinEvent(250, target_node=0))
    engine.process_next()
    assert [e.virtual_time_ns for e in engine.processed] == [100, 200, 250]
    assert [e.virtual_time_ns for e in engine.event_queue] == [300]
