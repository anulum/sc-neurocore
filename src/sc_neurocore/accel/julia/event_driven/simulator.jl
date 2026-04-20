# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for event_driven/simulator

module SimulatorAccel

using Statistics, LinearAlgebra

mutable struct EventDrivenSimulatorState
    time::Float64
    source_id::Float64
    target_id::Float64
    weight::Float64
    delay::Float64
    total_events_processed::Float64
    total_spikes_generated::Float64
    max_queue_size::Float64
    simulation_time::Float64
    events_per_spike::Float64
    speedup_vs_clockdriven::Float64
    n_neurons::Float64
    threshold::Float64
    tau_mem::Float64
    v_rest::Float64
end

function EventDrivenSimulatorState()
    EventDrivenSimulatorState(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
end

function summary(s::EventDrivenSimulatorState)
    return (
        f"EventDriven: {s.total_spikes_generated} spikes, "
        f"{s.total_events_processed} events, "
        f"queue_peak={s.max_queue_size}, "
        f"est. speedup={s.speedup_vs_clockdriven:.1f}x"
    )
end

function inject_spikes(s::EventDrivenSimulatorState, events, int]])
    for t, nid in events
        # External spike: propagate through all outgoing connections
        for tgt, w, d in s._adjacency.get(nid, [])
            heapq.heappush(
                s._event_queue,
                SpikeEvent(time=t + d, source_id=nid, target_id=tgt, weight=w, delay=d),
            )
end

function inject_current(s::EventDrivenSimulatorState, events, int, float]])
    for t, nid, current in events
        heapq.heappush(
            s._event_queue,
            SpikeEvent(time=t, source_id=-1, target_id=nid, weight=current),
        )
end

function run(s::EventDrivenSimulatorState, duration)
    stats = EventStats(simulation_time=duration)
    s._spike_log = []
    while s._event_queue
        event = heapq.heappop(s._event_queue)
        if event.time > duration
            break
        stats.total_events_processed += 1
        stats.max_queue_size = max(stats.max_queue_size, length(s._event_queue) + 1)
        nid = event.target_id
        t = event.time
        # Check refractory
        if t - s._last_spike_time[nid] < s.refractory
            continue
        # LIF membrane dynamics: exponential decay since last update
        dt_since_last = t - s._last_spike_time[nid]
        if dt_since_last > 0 && s._last_spike_time[nid] > -1e8:  # pragma: no cover
            decay = exp(-dt_since_last / s.tau_mem)
            s._v[nid] = s.v_rest + (s._v[nid] - s.v_rest) * decay
        # Apply synaptic input
        s._v[nid] += event.weight
        # Threshold check
        if s._v[nid] >= s.threshold
            s._v[nid] = s.v_reset
            s._last_spike_time[nid] = t
            s._spike_log = push!(, (t, nid))
            stats.total_spikes_generated += 1
            # Propagate spike to all targets
            for tgt, w, d in s._adjacency.get(nid, [])
                heapq.heappush(
                    s._event_queue,
                    SpikeEvent(time=t + d, source_id=nid, target_id=tgt, weight=w, delay=d),
                )
    # Compute speedup estimate
    clock_driven_ops = s.n_neurons * int(duration)  # 1 op per neuron per ms
    if stats.total_events_processed > 0
        stats.events_per_spike = stats.total_events_processed / max(
            stats.total_spikes_generated, 1
        )
        stats.speedup_vs_clockdriven = clock_driven_ops / max(stats.total_events_processed, 1)
    return s._spike_log, stats
end

function reset(s::EventDrivenSimulatorState)
    s._v = np.full(s.n_neurons, s.v_rest)
    s._last_spike_time = np.full(s.n_neurons, -1e9)
    s._event_queue = []
    s._spike_log = []
end

end # module SimulatorAccel
