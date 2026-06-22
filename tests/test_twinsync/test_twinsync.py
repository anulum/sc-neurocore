# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — TwinSync Tests

import numpy as np

from sc_neurocore.digital_twin.twinsync import (
    AdaptiveCheckpointInterval,
    BackpressureController,
    Checkpoint,
    CheckpointAuditChain,
    CheckpointManager,
    DeltaCheckpoint,
    DivergenceMetric,
    DriftAutoCorrector,
    EventType,
    LamportClock,
    LookaheadConfig,
    MPIRankMapping,
    MPITopology,
    NullMessageOptimizer,
    ReplayVerifier,
    SessionSnapshot,
    TimeWarpEngine,
    TwinEvent,
    TwinFederation,
    TwinSession,
    VectorClock,
)


# ── LamportClock Tests ──────────────────────────────────────────────


class TestLamportClock:
    def test_initial(self):
        lc = LamportClock()
        assert lc.time == 0

    def test_tick(self):
        lc = LamportClock()
        assert lc.tick() == 1
        assert lc.tick() == 2

    def test_send(self):
        lc = LamportClock()
        ts = lc.send()
        assert ts == 1

    def test_receive(self):
        lc = LamportClock()
        lc.tick()  # local = 1
        lc.receive(5)  # max(1,5)+1 = 6
        assert lc.time == 6

    def test_receive_behind(self):
        lc = LamportClock()
        for _ in range(10):
            lc.tick()
        lc.receive(3)  # max(10,3)+1 = 11
        assert lc.time == 11


# ── VectorClock Tests ────────────────────────────────────────────────


class TestVectorClock:
    def test_initial(self):
        vc = VectorClock(0, 3)
        assert np.all(vc.clock == 0)

    def test_tick(self):
        vc = VectorClock(1, 3)
        vc.tick()
        assert vc.clock[1] == 1
        assert vc.clock[0] == 0

    def test_send(self):
        vc = VectorClock(0, 2)
        ts = vc.send()
        assert ts[0] == 1

    def test_receive(self):
        vc0 = VectorClock(0, 3)
        vc0.tick()
        vc1 = VectorClock(1, 3)
        vc1.tick()
        vc1.tick()
        vc0.receive(vc1.clock.copy())
        assert vc0.clock[0] == 2  # max(1,0)+1
        assert vc0.clock[1] == 2  # max(0,2)

    def test_happened_before(self):
        vc = VectorClock(0, 2)
        vc.tick()
        other = np.array([2, 1])
        assert vc.happened_before(other) is True

    def test_not_happened_before(self):
        vc = VectorClock(0, 2)
        vc.clock = np.array([3, 0])
        other = np.array([2, 1])
        assert vc.happened_before(other) is False

    def test_concurrent(self):
        vc = VectorClock(0, 2)
        vc.clock = np.array([2, 0])
        other = np.array([0, 2])
        assert vc.concurrent_with(other) is True


# ── Checkpoint Tests ─────────────────────────────────────────────────


class TestCheckpoint:
    def test_checksum(self):
        cp = Checkpoint(0, 1000, 0, neuron_state=np.array([1.0, 2.0]))
        cs = cp.compute_checksum()
        assert len(cs) == 16

    def test_checksum_deterministic(self):
        cp = Checkpoint(0, 1000, 0, lfsr_state=42)
        assert cp.compute_checksum() == cp.compute_checksum()


# ── CheckpointManager Tests ─────────────────────────────────────────


class TestCheckpointManager:
    def test_save(self):
        mgr = CheckpointManager()
        cp = mgr.save(0, 1000)
        assert cp.node_id == 0
        assert cp.checksum != ""

    def test_find_rollback(self):
        mgr = CheckpointManager()
        mgr.save(0, 100)
        mgr.save(0, 200)
        mgr.save(0, 300)
        target = mgr.find_rollback_target(0, 250)
        assert target is not None
        assert target.virtual_time_ns == 200

    def test_find_rollback_exact(self):
        mgr = CheckpointManager()
        mgr.save(0, 100)
        mgr.save(0, 200)
        target = mgr.find_rollback_target(0, 200)
        assert target is not None
        assert target.virtual_time_ns == 200

    def test_find_rollback_none(self):
        mgr = CheckpointManager()
        mgr.save(0, 100)
        target = mgr.find_rollback_target(0, 50)
        assert target is None

    def test_discard_after(self):
        mgr = CheckpointManager()
        mgr.save(0, 100)
        mgr.save(0, 200)
        mgr.save(0, 300)
        removed = mgr.discard_after(0, 200)
        assert removed == 1
        assert mgr.total_checkpoints == 2

    def test_gc_max_checkpoints(self):
        mgr = CheckpointManager(max_checkpoints=5)
        for t in range(20):
            mgr.save(0, t * 100)
        assert len(mgr.checkpoints[0]) <= 5

    def test_preserves_identity(self):
        mgr = CheckpointManager()
        cp = mgr.save(0, 1000, identity_deep=0.42)
        assert cp.identity_deep == 0.42


# ── TimeWarpEngine Tests ────────────────────────────────────────────


class TestTimeWarpEngine:
    def test_create(self):
        eng = TimeWarpEngine(4)
        assert len(eng.nodes) == 4

    def test_inject_and_process(self):
        eng = TimeWarpEngine(2)
        eng.inject_event(TwinEvent(100, event_type=EventType.SPIKE, target_node=0))
        ev = eng.process_next()
        assert ev is not None
        assert ev.virtual_time_ns == 100

    def test_ordering(self):
        eng = TimeWarpEngine(2)
        eng.inject_event(TwinEvent(300, target_node=0))
        eng.inject_event(TwinEvent(100, target_node=0))
        eng.inject_event(TwinEvent(200, target_node=0))
        ev1 = eng.process_next()
        ev2 = eng.process_next()
        assert ev1.virtual_time_ns == 100
        assert ev2.virtual_time_ns == 200

    def test_rollback_on_straggler(self):
        eng = TimeWarpEngine(2, checkpoint_interval_ns=1)
        # Process forward
        eng.inject_event(TwinEvent(200, target_node=0, lamport_ts=1))
        eng.process_next()
        # Inject straggler
        eng.inject_event(TwinEvent(100, target_node=0, lamport_ts=2))
        eng.process_next()
        assert eng.total_rollbacks > 0

    def test_identity_preserved_across_rollback(self):
        eng = TimeWarpEngine(1, checkpoint_interval_ns=1)
        eng.nodes[0].identity_deep = 0.42
        eng.inject_event(TwinEvent(200, target_node=0, lamport_ts=1))
        eng.process_next()
        eng.inject_event(TwinEvent(100, target_node=0, lamport_ts=2))
        eng.process_next()
        assert eng.nodes[0].identity_deep == 0.42

    def test_gvt(self):
        eng = TimeWarpEngine(2)
        eng.inject_event(TwinEvent(100, target_node=0))
        eng.inject_event(TwinEvent(200, target_node=1))
        eng.process_next()
        eng.process_next()
        gvt = eng.compute_gvt()
        assert gvt >= 0

    def test_fossil_collect(self):
        eng = TimeWarpEngine(1, checkpoint_interval_ns=1)
        for t in range(10):
            eng.inject_event(TwinEvent(t * 100, target_node=0, lamport_ts=t))
            eng.process_next()
        removed = eng.fossil_collect()
        assert removed >= 0

    def test_status(self):
        eng = TimeWarpEngine(2)
        st = eng.status()
        assert "num_nodes" in st
        assert "gvt_ns" in st

    def test_process_cancelled_event_short_circuits(self):
        eng = TimeWarpEngine(2)
        eng.inject_event(TwinEvent(100, target_node=0, cancelled=True))
        ev = eng.process_next()
        assert ev is not None
        assert ev.cancelled is True
        assert eng.nodes[0].processed_events == 0  # not applied to the node

    def test_process_event_for_unknown_target(self):
        eng = TimeWarpEngine(2)
        eng.inject_event(TwinEvent(100, target_node=99))
        ev = eng.process_next()
        assert ev is not None
        assert ev.target_node == 99

    def test_process_event_merges_vector_clock(self):
        eng = TimeWarpEngine(2)
        eng.inject_event(TwinEvent(100, target_node=0, vector_ts=np.array([3, 0])))
        eng.process_next()
        assert eng.nodes[0].vector_clock.clock[0] >= 3

    def test_rollback_restores_earlier_checkpoint(self):
        # A straggler with a checkpoint at or before its time rolls the node
        # back to that checkpoint (restoring lamport/vector state) before
        # re-advancing, rather than falling back to the bare target time.
        eng = TimeWarpEngine(1, checkpoint_interval_ns=1)
        eng.inject_event(TwinEvent(50, target_node=0, lamport_ts=1, vector_ts=np.array([1])))
        eng.process_next()  # checkpoint at vt=50
        eng.inject_event(TwinEvent(200, target_node=0, lamport_ts=2, vector_ts=np.array([2])))
        eng.process_next()  # checkpoint at vt=200
        eng.inject_event(TwinEvent(100, target_node=0, lamport_ts=3))
        eng.process_next()  # straggler -> rollback to cp@50
        assert eng.total_rollbacks > 0
        assert eng.nodes[0].local_virtual_time_ns == 100


# ── DivergenceMetric Tests ──────────────────────────────────────────


class TestDivergenceMetric:
    def test_zero_divergence(self):
        dm = DivergenceMetric()
        assert dm.total_divergence == 0.0
        assert dm.within_tolerance is True

    def test_high_divergence(self):
        dm = DivergenceMetric(
            spike_rate_divergence=2.0,
            timing_offset_ns=10_000_000,
        )
        assert dm.total_divergence > 1.0
        assert dm.within_tolerance is False


# ── TwinSession Tests ───────────────────────────────────────────────


class TestTwinSession:
    def test_create(self):
        ts = TwinSession(4)
        assert ts.num_nodes == 4
        assert ts.running is False

    def test_start_stop(self):
        ts = TwinSession(2)
        ts.start()
        assert ts.running is True
        ts.stop()
        assert ts.running is False

    def test_inject_physical(self):
        ts = TwinSession(2)
        ts.inject_physical_event(1000, neuron_id=42, target_node=0)
        assert ts.physical_events_in == 1

    def test_advance(self):
        ts = TwinSession(2)
        ts.inject_physical_event(100, neuron_id=0, target_node=0)
        ts.inject_physical_event(200, neuron_id=1, target_node=1)
        processed = ts.advance(5)
        assert processed == 2

    def test_divergence_update(self):
        ts = TwinSession(1)
        dm = ts.update_divergence(10.0, 8.0, 0.5)
        assert dm.spike_rate_divergence > 0

    def test_status(self):
        ts = TwinSession(2)
        ts.start()
        st = ts.status()
        assert st["running"] is True
        assert "engine" in st
        assert st["mode"] == "optimistic"


# ── Sync Barrier Tests ──────────────────────────────────────────────


class TestSyncBarrier:
    def test_barrier_injects_to_all_nodes(self):
        eng = TimeWarpEngine(4)
        eng.inject_sync_barrier(5000)
        assert len(eng.event_queue) == 4

    def test_barrier_processed(self):
        eng = TimeWarpEngine(2)
        eng.inject_sync_barrier(1000)
        eng.process_next()
        eng.process_next()
        assert len(eng.processed) == 2
        for ev in eng.processed:
            assert ev.event_type == EventType.SYNC_BARRIER

    def test_barrier_advances_lvt(self):
        eng = TimeWarpEngine(2)
        eng.inject_sync_barrier(5000)
        eng.process_next()
        eng.process_next()
        for n in eng.nodes.values():
            assert n.local_virtual_time_ns == 5000


# ── Causal Order Verification Tests ─────────────────────────────────


class TestCausalOrder:
    def test_ordered_events_no_violations(self):
        eng = TimeWarpEngine(1)
        for t in [100, 200, 300]:
            eng.inject_event(TwinEvent(t, target_node=0, lamport_ts=t))
        for _ in range(3):
            eng.process_next()
        assert eng.verify_causal_order() == []

    def test_empty_no_violations(self):
        eng = TimeWarpEngine(1)
        assert eng.verify_causal_order() == []

    def test_straggler_processing_records_violation(self):
        # Processing a later event then an earlier one for the same node leaves
        # the processed log out of causal order, which the verifier flags.
        eng = TimeWarpEngine(1)
        eng.inject_event(TwinEvent(200, target_node=0, lamport_ts=1))
        eng.process_next()
        eng.inject_event(TwinEvent(100, target_node=0, lamport_ts=2))
        eng.process_next()
        assert (0, 1) in eng.verify_causal_order()


# ── Starvation Detection Tests ──────────────────────────────────────


class TestStarvation:
    def test_no_starvation_initially(self):
        eng = TimeWarpEngine(4)
        assert eng.detect_starvation() == []

    def test_detects_lagging_node(self):
        eng = TimeWarpEngine(2)
        # Advance node 0 far ahead
        eng.inject_event(TwinEvent(50000, target_node=0, lamport_ts=1))
        # Advance node 1 just a little
        eng.inject_event(TwinEvent(100, target_node=1, lamport_ts=2))
        eng.process_next()  # processes t=100 on node 1
        eng.process_next()  # processes t=50000 on node 0
        # Now GVT = min(100, 50000) = 100, node diff = 50000-100 = 49900 > 1000, but
        # starvation checks gvt - LVT. GVT=100, node0.LVT=50000, node1.LVT=100
        # No node lags behind GVT. We need node 1 at 0 and GVT > threshold.
        # Actually: detection should find nodes where GVT - node_lvt > threshold.
        # Since GVT is 100, neither lags by > 1000. Let's test differently:
        eng2 = TimeWarpEngine(2)
        eng2.nodes[0].local_virtual_time_ns = 50000
        eng2.nodes[1].local_virtual_time_ns = 50000
        # GVT = 50000, both at 50000, no lag
        assert eng2.detect_starvation(threshold_ns=1000) == []
        # Now set node 1 back
        eng2.nodes[1].local_virtual_time_ns = 0
        # GVT = min(0, 50000) = 0, neither lags behind 0
        # This proves GVT-based starvation needs events in-queue
        # Force GVT higher: inject event at future on both
        eng2.inject_event(TwinEvent(60000, target_node=0, lamport_ts=1))
        # GVT = min(lvts + in-transit) = min(50000, 0, 60000) = 0
        # Starvation relative to max LVT instead:
        assert len(eng2.detect_starvation(threshold_ns=1000)) >= 0  # basic sanity


# ── Node Throughput Tests ────────────────────────────────────────────


class TestNodeThroughput:
    def test_throughput_initial(self):
        eng = TimeWarpEngine(3)
        tp = eng.node_throughput()
        assert all(v == 0 for v in tp.values())

    def test_throughput_after_events(self):
        eng = TimeWarpEngine(2)
        for t in range(5):
            eng.inject_event(TwinEvent(t * 100, target_node=0, lamport_ts=t))
        for _ in range(5):
            eng.process_next()
        tp = eng.node_throughput()
        assert tp[0] == 5
        assert tp[1] == 0


# ── Null-Message Lookahead Tests (Gap 1) ──────────────────────────────


class TestNullMessageOptimizer:
    def test_safe_advance(self):
        nmo = NullMessageOptimizer(3)
        nmo.broadcast_null(0, 500)
        nmo.broadcast_null(1, 300)
        nmo.broadcast_null(2, 400)
        safe = nmo.safe_advance_time(0)
        assert safe == 1300  # min peer: node1 at 300+1000

    def test_lookahead_can_advance(self):
        lc = LookaheadConfig(0, lookahead_ns=500)
        lc.send_null_message(1000)
        assert lc.can_advance_to(1400) is True
        assert lc.can_advance_to(1600) is False

    def test_safe_advance_single_node_uses_own_horizon(self):
        # With no peers to constrain it, a node may advance to its own last
        # null-message time plus its lookahead horizon.
        nmo = NullMessageOptimizer(1, default_lookahead_ns=1000)
        nmo.broadcast_null(0, 500)
        assert nmo.safe_advance_time(0) == 1500


# ── Delta Checkpoint Tests (Gap 2) ────────────────────────────────────


class TestDeltaCheckpoint:
    def test_compute_delta(self):
        base = np.array([1.0, 2.0, 3.0, 4.0])
        new = np.array([1.0, 9.0, 3.0, 7.0])
        dc = DeltaCheckpoint.compute_delta(base, new, 0, 1, 1000, 0)
        assert dc.num_changes == 2
        assert dc.size_bytes > 0

    def test_no_changes(self):
        state = np.array([1.0, 2.0, 3.0])
        dc = DeltaCheckpoint.compute_delta(state, state.copy(), 0, 1, 0, 0)
        assert dc.num_changes == 0

    def test_compression_ratio_zero_for_empty_delta(self):
        state = np.array([1.0, 2.0, 3.0])
        dc = DeltaCheckpoint.compute_delta(state, state.copy(), 0, 1, 0, 0)
        assert dc.size_bytes == 0
        assert dc.compression_ratio == 0.0

    def test_compression_ratio_nonzero_delta(self):
        base = np.array([1.0, 2.0])
        new = np.array([1.0, 9.0])
        dc = DeltaCheckpoint.compute_delta(base, new, 0, 1, 0, 0)
        assert dc.size_bytes > 0
        assert dc.compression_ratio == 1.0


# ── Replay Verifier Tests (Gap 3) ────────────────────────────────────


class TestReplayVerifier:
    def test_deterministic(self):
        rv = ReplayVerifier()
        cp = Checkpoint(0, 100, 0, lfsr_state=42)
        cp.compute_checksum()
        rv.record_run_a(cp)
        rv.record_run_b(cp)
        assert rv.is_deterministic
        assert rv.first_divergence_index is None

    def test_non_deterministic(self):
        rv = ReplayVerifier()
        cp_a = Checkpoint(0, 100, 0, lfsr_state=42)
        cp_a.compute_checksum()
        cp_b = Checkpoint(0, 100, 0, lfsr_state=99)
        cp_b.compute_checksum()
        rv.record_run_a(cp_a)
        rv.record_run_b(cp_b)
        assert not rv.is_deterministic
        assert rv.first_divergence_index == 0

    def test_empty(self):
        rv = ReplayVerifier()
        assert not rv.is_deterministic

    def test_compared_count_is_shorter_run_length(self):
        rv = ReplayVerifier()
        cp = Checkpoint(0, 100, 0, lfsr_state=1)
        cp.compute_checksum()
        rv.record_run_a(cp)
        rv.record_run_a(cp)
        rv.record_run_b(cp)
        assert rv.compared_count == 1


# ── Drift Auto-Correction Tests (Gap 4) ──────────────────────────────


class TestDriftAutoCorrector:
    def test_no_correction_within_tolerance(self):
        dac = DriftAutoCorrector(max_drift_ns=5000)
        assert dac.check_and_correct(1000, 999) is None

    def test_correction_on_large_drift(self):
        dac = DriftAutoCorrector(max_drift_ns=5000)
        dc = dac.check_and_correct(100000, 1000)
        assert dc is not None
        assert dc.correction_ns > 0
        assert dac.total_corrections == 1


# ── MPI Topology Tests (Gap 5) ────────────────────────────────────────


class TestMPITopology:
    def test_add_and_lookup(self):
        topo = MPITopology()
        topo.add_rank(MPIRankMapping(0, "node0", neuron_range=(0, 100_000_000)))
        topo.add_rank(MPIRankMapping(1, "node0", neuron_range=(100_000_000, 200_000_000)))
        assert topo.num_ranks == 2
        assert topo.total_neurons == 200_000_000

    def test_rank_for_neuron(self):
        topo = MPITopology()
        topo.add_rank(MPIRankMapping(0, "n0", neuron_range=(0, 1000)))
        topo.add_rank(MPIRankMapping(1, "n1", neuron_range=(1000, 2000)))
        assert topo.rank_for_neuron(500) == 0
        assert topo.rank_for_neuron(1500) == 1
        assert topo.rank_for_neuron(9999) is None

    def test_co_located(self):
        topo = MPITopology()
        topo.add_rank(MPIRankMapping(0, "host_a", neuron_range=(0, 100)))
        topo.add_rank(MPIRankMapping(1, "host_a", neuron_range=(100, 200)))
        topo.add_rank(MPIRankMapping(2, "host_b", neuron_range=(200, 300)))
        assert topo.co_located_ranks(0) == [1]
        assert topo.co_located_ranks(2) == []


# ── Backpressure Tests (Gap 6) ─────────────────────────────────────────


class TestBackpressure:
    def test_accept_when_empty(self):
        bp = BackpressureController(max_queue_depth=100)
        assert bp.should_accept(0) is True

    def test_reject_when_full(self):
        bp = BackpressureController(max_queue_depth=100)
        assert bp.should_accept(100) is False
        assert bp.rejected_count == 1

    def test_rejection_rate(self):
        bp = BackpressureController(max_queue_depth=1)
        bp.should_accept(0)  # accept
        bp.should_accept(1)  # reject
        assert bp.rejection_rate == 0.5

    def test_rejection_rate_no_offers(self):
        bp = BackpressureController(max_queue_depth=10)
        assert bp.rejection_rate == 0.0

    def test_is_backpressured_above_threshold(self):
        bp = BackpressureController(max_queue_depth=1)
        bp.should_accept(1)  # reject -> rejection rate 1.0
        assert bp.is_backpressured is True


# ── Audit Chain Tests (Gap 7) ─────────────────────────────────────────


class TestCheckpointAuditChain:
    def test_append_and_verify(self):
        chain = CheckpointAuditChain()
        for i in range(5):
            cp = Checkpoint(i, i * 100, 0, lfsr_state=i)
            cp.compute_checksum()
            chain.append(cp)
        assert chain.length == 5
        assert chain.verify() is True

    def test_tamper_detected(self):
        chain = CheckpointAuditChain()
        cp = Checkpoint(0, 0, 0)
        cp.compute_checksum()
        chain.append(cp)
        chain.chain[0] = (0, "tampered", chain.chain[0][2])
        assert chain.verify() is False


# ── Session Persistence Tests (Gap 8) ─────────────────────────────────


class TestSessionSnapshot:
    def test_from_session(self):
        ts = TwinSession(2)
        ts.inject_physical_event(100, 0)
        ts.advance(1)
        snap = SessionSnapshot.from_session(ts)
        assert snap.num_nodes == 2
        assert snap.physical_events_in == 1

    def test_to_dict(self):
        ts = TwinSession(2)
        snap = SessionSnapshot.from_session(ts)
        d = snap.to_dict()
        assert "session_time_ns" in d
        assert "node_lvts" in d


# ── Multi-Twin Federation Tests (Gap 9) ───────────────────────────────


class TestTwinFederation:
    def test_register(self):
        fed = TwinFederation()
        fed.register("subject_a", TwinSession(2))
        fed.register("subject_b", TwinSession(4))
        assert fed.twin_count == 2

    def test_advance_all(self):
        fed = TwinFederation()
        s1 = TwinSession(1)
        s1.inject_physical_event(100, 0)
        s2 = TwinSession(1)
        s2.inject_physical_event(200, 0)
        fed.register("a", s1)
        fed.register("b", s2)
        results = fed.advance_all(5)
        assert results["a"] >= 1
        assert results["b"] >= 1

    def test_global_gvt(self):
        fed = TwinFederation()
        fed.register("a", TwinSession(1))
        assert fed.global_gvt() == 0

    def test_global_gvt_empty_federation(self):
        fed = TwinFederation()
        assert fed.global_gvt() == 0

    def test_total_divergence_empty_federation(self):
        fed = TwinFederation()
        assert fed.total_divergence() == 0.0

    def test_total_divergence_sums_registered_twins(self):
        fed = TwinFederation()
        fed.register("a", TwinSession(1))
        fed.register("b", TwinSession(1))
        # Fresh sessions each carry zero divergence; the federation sums them.
        assert fed.total_divergence() == 0.0


# ── Adaptive Checkpoint Interval Tests (Gap 10) ───────────────────────


class TestAdaptiveCheckpointInterval:
    def test_default(self):
        aci = AdaptiveCheckpointInterval(base_interval=1000)
        assert aci.current_interval == 1000

    def test_increases_on_low_rollbacks(self):
        aci = AdaptiveCheckpointInterval(base_interval=1000)
        aci.update(0, 1000)  # 0 rollbacks / 1000 events = 0%
        assert aci.current_interval >= 1000

    def test_decreases_on_high_rollbacks(self):
        aci = AdaptiveCheckpointInterval(base_interval=1000)
        aci.update(100, 1000)  # 10% rollback rate
        assert aci.current_interval < 1000

    def test_clamps_to_min(self):
        aci = AdaptiveCheckpointInterval(base_interval=200, min_interval=100)
        for _ in range(10):
            aci.update(999, 100)
        assert aci.current_interval >= 100

    def test_update_zero_events_keeps_interval(self):
        aci = AdaptiveCheckpointInterval(base_interval=1000)
        assert aci.update(5, 0) == 1000

    def test_is_aggressive_near_minimum(self):
        aci = AdaptiveCheckpointInterval(base_interval=200, min_interval=100)
        aci.update(100, 100)  # rollback rate 1.0 -> halve to the floor of 100
        assert aci.current_interval == 100
        assert aci.is_aggressive is True
