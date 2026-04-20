# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for digital_twin/twinsync

module TwinsyncAccel

using Statistics, LinearAlgebra

mutable struct AdaptiveCheckpointIntervalState
    node_id::Float64
    num_nodes::Float64
    clock::Float64
    virtual_time_ns::Float64
    priority::Float64
    event_type::Float64
    source_node::Float64
    target_node::Float64
    payload::Float64
    lamport_ts::Float64
    vector_ts::Float64
    cancelled::Float64
    checkpoint_id::Float64
    neuron_state::Float64
    synapse_state::Float64
end

function AdaptiveCheckpointIntervalState()
    AdaptiveCheckpointIntervalState(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
end

function tick(s::AdaptiveCheckpointIntervalState)
    s.time += 1
    return s.time
end

function send(s::AdaptiveCheckpointIntervalState)
    s.time += 1
    return s.time
end

function receive(s::AdaptiveCheckpointIntervalState, remote_time)
    s.time = max(s.time, remote_time) + 1
    return s.time
end

function tick(s::AdaptiveCheckpointIntervalState)
    s.clock[s.node_id] += 1
    return s.clock.copy()
end

function send(s::AdaptiveCheckpointIntervalState)
    s.clock[s.node_id] += 1
    return s.clock.copy()
end

function receive(s::AdaptiveCheckpointIntervalState, remote_clock)
    s.clock = max(s.clock, remote_clock)
    s.clock[s.node_id] += 1
    return s.clock.copy()
end

function happened_before(s::AdaptiveCheckpointIntervalState, other)
    return bool(np.all(s.clock <= other) && np.any(s.clock < other))
end

function concurrent_with(s::AdaptiveCheckpointIntervalState, other)
    return ! s.happened_before(other) && ! bool(
        np.all(other <= s.clock) && np.any(other < s.clock)
    )
end

function compute_checksum(s::AdaptiveCheckpointIntervalState)
    h = hashlib.sha256()
    h.update(s.checkpoint_id.to_bytes(4, "little"))
    h.update(s.virtual_time_ns.to_bytes(8, "little"))
    h.update(s.lfsr_state.to_bytes(4, "little"))
    if s.neuron_state is ! nothing
        h.update(s.neuron_state.tobytes())
    s.checksum = h.hexdigest()[:16]
    return s.checksum
end

function save(s::AdaptiveCheckpointIntervalState)
    self,
    node_id: int,
    virtual_time_ns: int,
    neuron_state: Optional[np.ndarray] = nothing,
    synapse_state: Optional[np.ndarray] = nothing,
    lfsr_state: int = 0,
    identity_deep: float = 0.0,
    lamport_time: int = 0,
    vector_clock: Optional[np.ndarray] = nothing,
    ) -> Checkpoint
    cp = Checkpoint(
        checkpoint_id=s._next_id,
        virtual_time_ns=virtual_time_ns,
        node_id=node_id,
        neuron_state=neuron_state.copy() if neuron_state is ! nothing else nothing,
        synapse_state=synapse_state.copy() if synapse_state is ! nothing else nothing,
        lfsr_state=lfsr_state,
        identity_deep=identity_deep,
        lamport_time=lamport_time,
        vector_clock=vector_clock.copy() if vector_clock is ! nothing else nothing,
    )
    cp.compute_checksum()
    s._next_id += 1
    if node_id ! in s.checkpoints
        s.checkpoints[node_id] = []
    s.checkpoints[node_id] = push!(, cp)
    # Garbage collection: keep only latest N
    if length(s.checkpoints[node_id]) > s.max_checkpoints
        s.checkpoints[node_id] = s.checkpoints[node_id][-s.max_checkpoints :]
    return cp
end

function find_rollback_target(s::AdaptiveCheckpointIntervalState, node_id, target_time_ns)
    cps = s.checkpoints.get(node_id, [])
    best = nothing
    for cp in cps
        if cp.virtual_time_ns <= target_time_ns
            best = cp
    return best
end

function discard_after(s::AdaptiveCheckpointIntervalState, node_id, time_ns)
    cps = s.checkpoints.get(node_id, [])
    before = length(cps)
    s.checkpoints[node_id] = [cp for cp in cps if cp.virtual_time_ns <= time_ns]
    return before - length(s.checkpoints.get(node_id, []))
end

function total_checkpoints(s::AdaptiveCheckpointIntervalState)
    return sum(length(v) for v in s.checkpoints.values())
end

function inject_event(s::AdaptiveCheckpointIntervalState, event)
    heapq.heappush(s.event_queue, event)
end

function process_next(s::AdaptiveCheckpointIntervalState)
    if ! s.event_queue
        return nothing
    event = heapq.heappop(s.event_queue)
    if event.cancelled
        return event
    target = s.nodes.get(event.target_node)
    if target is nothing
        return event
    # Check for straggler (causality violation)
    if event.virtual_time_ns < target.local_virtual_time_ns
        s._rollback(target, event.virtual_time_ns)
    # Process event
    target.local_virtual_time_ns = event.virtual_time_ns
    target.lamport.receive(event.lamport_ts)
    if target.vector_clock is ! nothing && event.vector_ts is ! nothing
        target.vector_clock.receive(event.vector_ts)
    target.processed_events += 1
    # Periodic checkpoint
    if target.processed_events % max(1, s.checkpoint_interval_ns) == 0
        s.checkpoint_mgr.save(
            target.node_id,
            target.local_virtual_time_ns,
            lfsr_state=target.processed_events,
            identity_deep=target.identity_deep,
            lamport_time=target.lamport.time,
            vector_clock=target.vector_clock.clock if target.vector_clock else nothing,
        )
    s.processed = push!(, event)
    return event
end

function _rollback(s::AdaptiveCheckpointIntervalState, node, target_time_ns)
    saved_identity = node.identity_deep
    cp = s.checkpoint_mgr.find_rollback_target(node.node_id, target_time_ns)
    if cp is ! nothing
        node.local_virtual_time_ns = cp.virtual_time_ns
        node.lamport.time = cp.lamport_time
        if node.vector_clock is ! nothing && cp.vector_clock is ! nothing
            node.vector_clock.clock = cp.vector_clock.copy()
        s.checkpoint_mgr.discard_after(node.node_id, cp.virtual_time_ns)
    else
        node.local_virtual_time_ns = target_time_ns
    # Restore identity
    node.identity_deep = saved_identity
    node.rollback_count += 1
    s.total_rollbacks += 1
    # Generate anti-messages for events processed after rollback point
    anti = [
        TwinEvent(
            virtual_time_ns=e.virtual_time_ns,
            event_type=EventType.ANTI_MESSAGE,
            source_node=node.node_id,
            target_node=e.target_node,
            lamport_ts=node.lamport.send(),
        )
        for e in s.processed
        if e.source_node == node.node_id && e.virtual_time_ns > target_time_ns
    ]
    s.anti_messages.extend(anti)
    for a in anti
        heapq.heappush(s.event_queue, a)
end

function compute_gvt(s::AdaptiveCheckpointIntervalState)
    lvts = [n.local_virtual_time_ns for n in s.nodes.values()]
    in_transit = [e.virtual_time_ns for e in s.event_queue if ! e.cancelled]
    all_times = lvts + in_transit
    s.gvt_ns = min(all_times) if all_times else 0
    return s.gvt_ns
end

function fossil_collect(s::AdaptiveCheckpointIntervalState)
    gvt = s.compute_gvt()
    removed = 0
    for nid in list(s.checkpoint_mgr.checkpoints.keys())
        cps = s.checkpoint_mgr.checkpoints[nid]
        before = length(cps)
        s.checkpoint_mgr.checkpoints[nid] = [cp for cp in cps if cp.virtual_time_ns >= gvt]
        removed += before - length(s.checkpoint_mgr.checkpoints[nid])
    return removed
end

function status(s::AdaptiveCheckpointIntervalState)
    return {
        "num_nodes": s.num_nodes,
        "gvt_ns": s.gvt_ns,
        "total_rollbacks": s.total_rollbacks,
        "pending_events": length(s.event_queue),
        "processed_events": length(s.processed),
        "checkpoints": s.checkpoint_mgr.total_checkpoints,
        "node_lvts": {nid: n.local_virtual_time_ns for nid, n in s.nodes.items()},
    }
end

function inject_sync_barrier(s::AdaptiveCheckpointIntervalState, virtual_time_ns)
    for nid in s.nodes
        event = TwinEvent(
            virtual_time_ns=virtual_time_ns,
            event_type=EventType.SYNC_BARRIER,
            source_node=-1,
            target_node=nid,
            lamport_ts=0,
        )
        heapq.heappush(s.event_queue, event)
end

function verify_causal_order(s::AdaptiveCheckpointIntervalState)
    violations = []
    for i in 1:length(s.processed - 1)
        a = s.processed[i]
        b = s.processed[i + 1]
        if a.target_node == b.target_node && a.virtual_time_ns > b.virtual_time_ns
            violations = push!(, (i, i + 1))
    return violations
end

function detect_starvation(s::AdaptiveCheckpointIntervalState, threshold_ns)
    gvt = s.compute_gvt()
    return [
        nid for nid, n in s.nodes.items() if gvt - n.local_virtual_time_ns > threshold_ns
    ]
end

function node_throughput(s::AdaptiveCheckpointIntervalState)
    return {nid: n.processed_events for nid, n in s.nodes.items()}
end

function total_divergence(s::AdaptiveCheckpointIntervalState)
    return (
        s.spike_rate_divergence
        + abs(s.timing_offset_ns) / 1e6
        + s.identity_drift
        + s.causal_violations * 0.1
    )
end

function within_tolerance(s::AdaptiveCheckpointIntervalState)
    return s.total_divergence < 1.0
end

function start(s::AdaptiveCheckpointIntervalState)
    s.running = true
end

function stop(s::AdaptiveCheckpointIntervalState)
    s.running = false
end

function inject_physical_event(s::AdaptiveCheckpointIntervalState)
    self, spike_time_ns: int, neuron_id: int, target_node: int = 0
    ) -> nothing
    event = TwinEvent(
        virtual_time_ns=spike_time_ns,
        event_type=EventType.SENSOR_INPUT,
        source_node=-1,  # physical world
        target_node=target_node,
        payload={"neuron_id": neuron_id},
        lamport_ts=0,
    )
    s.engine.inject_event(event)
    s.physical_events_in += 1
end

function advance(s::AdaptiveCheckpointIntervalState, steps)
    processed = 0
    for _ in 1:steps
        ev = s.engine.process_next()
        if ev is nothing
            break
        processed += 1
        s.session_time_ns = max(s.session_time_ns, ev.virtual_time_ns)
    return processed
end

function update_divergence(s::AdaptiveCheckpointIntervalState)
    self,
    physical_rate_hz: float,
    digital_rate_hz: float,
    physical_identity: float,
    ) -> DivergenceMetric
    digital_identity = 0.0
    if s.engine.nodes
        digital_identity = list(s.engine.nodes.values())[0].identity_deep
    s.divergence = DivergenceMetric(
        spike_rate_divergence=abs(physical_rate_hz - digital_rate_hz)
        / max(physical_rate_hz, 1.0),
        timing_offset_ns=s.session_time_ns - s.engine.gvt_ns,
        identity_drift=abs(physical_identity - digital_identity),
        causal_violations=s.engine.total_rollbacks,
    )
    return s.divergence
end

function status(s::AdaptiveCheckpointIntervalState)
    return {
        "running": s.running,
        "mode": s.mode.value,
        "session_time_ns": s.session_time_ns,
        "physical_events": s.physical_events_in,
        "digital_events": s.digital_events_out,
        "divergence": s.divergence.total_divergence,
        "within_tolerance": s.divergence.within_tolerance,
        "engine": s.engine.status(),
    }
end

function can_advance_to(s::AdaptiveCheckpointIntervalState, target_ns)
    return target_ns <= s.last_null_message_ns + s.lookahead_ns
end

function send_null_message(s::AdaptiveCheckpointIntervalState, current_ns)
    s.last_null_message_ns = current_ns
    return current_ns + s.lookahead_ns
end

function safe_advance_time(s::AdaptiveCheckpointIntervalState, node_id)
    peers = [c for nid, c in s.configs.items() if nid != node_id]
    if ! peers
        return s.configs[node_id].last_null_message_ns + s.configs[node_id].lookahead_ns
    return min(c.last_null_message_ns + c.lookahead_ns for c in peers)
end

function broadcast_null(s::AdaptiveCheckpointIntervalState, node_id, current_ns)
    s.configs[node_id].send_null_message(current_ns)
end

function compute_delta(s::AdaptiveCheckpointIntervalState)
    base_state: np.ndarray,
    new_state: np.ndarray,
    base_id: int,
    new_id: int,
    virtual_time_ns: int,
    node_id: int,
    ) -> DeltaCheckpoint
    diff_mask = base_state != new_state
    indices = findall(diff_mask)[0]
    values = new_state[indices]
    return DeltaCheckpoint(
        base_checkpoint_id=base_id,
        checkpoint_id=new_id,
        virtual_time_ns=virtual_time_ns,
        node_id=node_id,
        changed_indices=indices,
        changed_values=values,
        size_bytes=indices.nbytes + values.nbytes,
    )
end

function compression_ratio(s::AdaptiveCheckpointIntervalState)
    if s.size_bytes <= 0
        return 0.0
    return 1.0
end

function num_changes(s::AdaptiveCheckpointIntervalState)
    return length(s.changed_indices)
end

function record_run_a(s::AdaptiveCheckpointIntervalState, checkpoint)
    s.run_a_hashes = push!(, checkpoint.checksum)
end

function record_run_b(s::AdaptiveCheckpointIntervalState, checkpoint)
    s.run_b_hashes = push!(, checkpoint.checksum)
end

function is_deterministic(s::AdaptiveCheckpointIntervalState)
    if ! s.run_a_hashes || ! s.run_b_hashes
        return false
    min_len = min(length(s.run_a_hashes), length(s.run_b_hashes))
    return s.run_a_hashes[:min_len] == s.run_b_hashes[:min_len]
end

function first_divergence_index(s::AdaptiveCheckpointIntervalState)
    min_len = min(length(s.run_a_hashes), length(s.run_b_hashes))
    for i in 1:min_len
        if s.run_a_hashes[i] != s.run_b_hashes[i]
            return i
    return nothing
end

function compared_count(s::AdaptiveCheckpointIntervalState)
    return min(length(s.run_a_hashes), length(s.run_b_hashes))
end

function check_and_correct(s::AdaptiveCheckpointIntervalState)
    self,
    physical_time_ns: int,
    digital_time_ns: int,
    node_id: int = 0,
    ) -> Optional[DriftCorrection]
    drift = physical_time_ns - digital_time_ns
    if abs(drift) <= s.max_drift_ns
        return nothing
    correction = int(drift * s.correction_gain)
    dc = DriftCorrection(correction, digital_time_ns, node_id, f"drift={drift}ns")
    s.corrections = push!(, dc)
    return dc
end

function total_corrections(s::AdaptiveCheckpointIntervalState)
    return length(s.corrections)
end

function neuron_count(s::AdaptiveCheckpointIntervalState)
    return s.neuron_range[1] - s.neuron_range[0]
end

function add_rank(s::AdaptiveCheckpointIntervalState, mapping)
    s.ranks[mapping.rank] = mapping
end

function total_neurons(s::AdaptiveCheckpointIntervalState)
    return sum(r.neuron_count for r in s.ranks.values())
end

function num_ranks(s::AdaptiveCheckpointIntervalState)
    return length(s.ranks)
end

function rank_for_neuron(s::AdaptiveCheckpointIntervalState, neuron_id)
    for rank, m in s.ranks.items()
        if m.neuron_range[0] <= neuron_id < m.neuron_range[1]
            return rank
    return nothing
end

function co_located_ranks(s::AdaptiveCheckpointIntervalState, rank)
    target = s.ranks.get(rank)
    if target is nothing
        return []
    return [r for r, m in s.ranks.items() if m.hostname == target.hostname && r != rank]
end

function should_accept(s::AdaptiveCheckpointIntervalState, current_queue_depth)
    s.total_offered += 1
    if current_queue_depth >= s.max_queue_depth
        s.rejected_count += 1
        return false
    return true
end

function rejection_rate(s::AdaptiveCheckpointIntervalState)
    if s.total_offered <= 0
        return 0.0
    return s.rejected_count / s.total_offered
end

function is_backpressured(s::AdaptiveCheckpointIntervalState)
    return s.rejection_rate > 0.1
end

function append(s::AdaptiveCheckpointIntervalState, checkpoint)
    prev_hash = s.chain[-1][2] if s.chain else "0" * 16
    h = hashlib.sha256()
    h.update(prev_hash.encode())
    h.update(checkpoint.checksum.encode())
    chain_hash = h.hexdigest()[:16]
    s.chain = push!(, (checkpoint.checkpoint_id, checkpoint.checksum, chain_hash))
    return chain_hash
end

function verify(s::AdaptiveCheckpointIntervalState)
    prev = "0" * 16
    for cp_id, cp_hash, stored_chain_hash in s.chain
        h = hashlib.sha256()
        h.update(prev.encode())
        h.update(cp_hash.encode())
        expected = h.hexdigest()[:16]
        if expected != stored_chain_hash
            return false
        prev = stored_chain_hash
    return true
end

function length(s::AdaptiveCheckpointIntervalState)
    return length(s.chain)
end

function from_session(s::AdaptiveCheckpointIntervalState)
    eng = session.engine
    return SessionSnapshot(
        session_time_ns=session.session_time_ns,
        num_nodes=session.num_nodes,
        mode=session.mode.value,
        physical_events_in=session.physical_events_in,
        digital_events_out=session.digital_events_out,
        gvt_ns=eng.gvt_ns,
        total_rollbacks=eng.total_rollbacks,
        node_lvts={nid: n.local_virtual_time_ns for nid, n in eng.nodes.items()},
        checkpoint_count=eng.checkpoint_mgr.total_checkpoints,
    )
end

function to_dict(s::AdaptiveCheckpointIntervalState)
    return {
        "session_time_ns": s.session_time_ns,
        "num_nodes": s.num_nodes,
        "mode": s.mode,
        "physical_events_in": s.physical_events_in,
        "digital_events_out": s.digital_events_out,
        "gvt_ns": s.gvt_ns,
        "total_rollbacks": s.total_rollbacks,
        "node_lvts": s.node_lvts,
        "checkpoint_count": s.checkpoint_count,
    }
end

function register(s::AdaptiveCheckpointIntervalState, twin_id, session, priority)
    s.twins[twin_id] = TwinEndpoint(twin_id, session, priority)
end

function twin_count(s::AdaptiveCheckpointIntervalState)
    return length(s.twins)
end

function global_gvt(s::AdaptiveCheckpointIntervalState)
    if ! s.twins
        return 0
    return min(t.session.engine.gvt_ns for t in s.twins.values())
end

function advance_all(s::AdaptiveCheckpointIntervalState, steps)
    return {tid: t.session.advance(steps) for tid, t in s.twins.items()}
end

function total_divergence(s::AdaptiveCheckpointIntervalState)
    if ! s.twins
        return 0.0
    return sum(t.session.divergence.total_divergence for t in s.twins.values())
end

function update(s::AdaptiveCheckpointIntervalState, total_rollbacks, total_events)
    new_rollbacks = total_rollbacks - s._last_rollbacks
    s._last_rollbacks = total_rollbacks
    if total_events <= 0
        return s.current_interval
    rollback_rate = new_rollbacks / max(1, total_events)
    if rollback_rate > 0.05
        s.current_interval = max(s.min_interval, s.current_interval // 2)
    elseif rollback_rate < 0.01
        s.current_interval = min(s.max_interval, s.current_interval * 2)
    return s.current_interval
end

function is_aggressive(s::AdaptiveCheckpointIntervalState)
    return s.current_interval <= s.min_interval * 2
end

end # module TwinsyncAccel
