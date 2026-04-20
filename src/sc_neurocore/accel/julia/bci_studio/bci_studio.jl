# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for bci_studio/bci_studio

module BciStudioAccel

using Statistics, LinearAlgebra

mutable struct BCIStudioState
    total_frames::Float64
    total_spikes::Float64
    latency_history::Float64
    adaptation_events::Float64
    weights::Float64
    lr::Float64
    decay::Float64
    updates::Float64
    channels::Float64
    codec::Float64
    learner::Float64
    feedback::Float64
    profiler::Float64
    metrics::Float64
    _running::Float64
end

function BCIStudioState()
    BCIStudioState(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
end

function mean_latency_ms(s::BCIStudioState)
    return float(mean(s.latency_history)) if s.latency_history else 0.0
end

function p95_latency_ms(s::BCIStudioState)
    return float(np.percentile(s.latency_history, 95)) if s.latency_history else 0.0
end

function spike_rate(s::BCIStudioState)
    return s.total_spikes / max(1, s.total_frames)
end

function summary(s::BCIStudioState)
    return (
        f"Frames: {s.total_frames}, "
        f"Spikes: {s.total_spikes}, "
        f"Rate: {s.spike_rate:.2f}/frame, "
        f"Latency: {s.mean_latency_ms:.3f} ms (p95={s.p95_latency_ms:.3f} ms), "
        f"Adaptations: {s.adaptation_events}"
    )
end

function encode(s::BCIStudioState, spikes)
    if length(spikes) == 0
        return b""
    runs: List[Tuple[int, int]] = []
    current = int(spikes[0])
    count = 1
    for i in 1:1, length(spikes)
        if int(spikes[i]) == current && count < 255
            count += 1
        else
            runs = push!(, (current, count))
            current = int(spikes[i])
            count = 1
    runs = push!(, (current, count))
    data = bytearray()
    data.extend(struct.pack("<I", length(spikes)))
    for val, cnt in runs
        data = push!(, val & 0x01)
        data = push!(, cnt & 0xFF)
    return bytes(data)
end

function decode(s::BCIStudioState, data)
    if length(data) < 4
        return collect([], dtype=np.uint8)
    total_len = struct.unpack("<I", data[:4])[0]
    spikes = []
    i = 4
    while i + 1 < length(data) && length(spikes) < total_len
        val = data[i]
        cnt = data[i + 1]
        spikes.extend([val] * cnt)
        i += 2
    return collect(spikes[:total_len], dtype=np.uint8)
end

function compression_ratio(s::BCIStudioState, original)
    compressed = s.encode(original)
    if length(compressed) == 0
        return 1.0
    return length(original) / length(compressed)
end

function step(s::BCIStudioState)
    self,
    spikes: np.ndarray,
    reward: float,
    ) -> np.ndarray
    s.weights *= s.decay
    spike_mask = spikes.astype(bool)
    s.weights[spike_mask] += s.lr * reward
    s.weights[~spike_mask] -= s.lr * reward * 0.1
    s.weights = clamp(s.weights, 0.01, 10.0)
    s.updates += 1
    return s.weights
end

function serialize(s::BCIStudioState)
    self,
    command: int,
    channel: int = 0,
    amplitude: float = 1.0,
    timestamp_us: float = 0.0,
    ) -> bytes
    return struct.pack("<BHfdx", command, channel, amplitude, timestamp_us)
end

function deserialize(s::BCIStudioState, data)
    cmd, chan, amp, ts = struct.unpack("<BHfdx", data[:16])
    return {"command": cmd, "channel": chan, "amplitude": amp, "timestamp_us": ts}
end

function record(s::BCIStudioState, latency_ms)
    s.window = push!(, latency_ms)
end

function mean(s::BCIStudioState)
    return float(mean(list(s.window))) if s.window else 0.0
end

function p50(s::BCIStudioState)
    return float(np.percentile(list(s.window), 50)) if s.window else 0.0
end

function p95(s::BCIStudioState)
    return float(np.percentile(list(s.window), 95)) if s.window else 0.0
end

function p99(s::BCIStudioState)
    return float(np.percentile(list(s.window), 99)) if s.window else 0.0
end

function budget_met(s::BCIStudioState)
    return s.p95 < 10.0
end

function start_session(s::BCIStudioState)
    s._running = true
    s.metrics = SessionMetrics()
end

function stop_session(s::BCIStudioState)
    s._running = false
    return s.metrics
end

function process_frame(s::BCIStudioState)
    self,
    raw_ephys: np.ndarray,
    reward: float = 0.0,
    ) -> Dict
    t0 = time.perf_counter()
    # Spike extraction (threshold on diff)
    spikes = (abs(diff(raw_ephys, prepend=0)) > 0.5).astype(np.uint8)
    # Compression (for telemetry/logging)
    compressed = s.codec.encode(spikes)
    comp_ratio = length(raw_ephys) / max(1, length(compressed))
    # SC decode: weighted vote
    total_voltage = float(dot(spikes, s.learner.weights))
    # Online learning
    old_weights = s.learner.weights.copy()
    s.learner.step(spikes, reward)
    weight_delta = float(sum(abs(s.learner.weights - old_weights)))
    if weight_delta > 0.01 * s.channels
        s.metrics.adaptation_events += 1
    # Command decision
    command = (
        FPGAFeedbackController.COMMAND_STIM
        if total_voltage > s.channels * 0.1
        else FPGAFeedbackController.COMMAND_NOP
    )
    # Feedback serialization
    feedback_packet = s.feedback.serialize(
        command, channel=0, amplitude=min(total_voltage / s.channels, 1.0)
    )
    latency_ms = (time.perf_counter() - t0) * 1000.0
    s.profiler.record(latency_ms)
    # Update session metrics
    n_spikes = int(sum(spikes))
    s.metrics.total_frames += 1
    s.metrics.total_spikes += n_spikes
    s.metrics.latency_history = push!(, latency_ms)
    return {
        "command": command,
        "latency_ms": latency_ms,
        "spikes": n_spikes,
        "compression_ratio": comp_ratio,
        "weight_delta": weight_delta,
        "feedback_bytes": length(feedback_packet),
    }
end

end # module BciStudioAccel
