# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for debug/sc_scope

module ScScopeAccel

using Statistics, LinearAlgebra

mutable struct ScopeRendererState
    transport_type::Float64
    port::Float64
    baud_rate::Float64
    dma_base_addr::Float64
    dma_length::Float64
    timeout_ms::Float64
    config::Float64
    is_connected::Float64
    bytes_received::Float64
    _sim_rng::Float64
    _sim_step::Float64
    timestamp_ns::Float64
    layer_id::Float64
    neuron_id::Float64
    words::Float64
end

function ScopeRendererState()
    ScopeRendererState(0.0, 0.0, 115200.0, 1073741824.0, 4096.0, 100.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
end

function connect(s::ScopeRendererState)
    if s.config.transport_type == TransportType.SIMULATED
        s._sim_rng = np.random.default_rng(42)
        s.is_connected = true
        return true
    # Real backends would initialise JTAG/UART/DMA here
    s.is_connected = true
    return true
end

function disconnect(s::ScopeRendererState)
    s.is_connected = false
    s._sim_rng = nothing
    s._sim_step = 0
end

function read_bitstream(s::ScopeRendererState, num_words, layer_id)
    if ! s.is_connected
        return nothing
    if s.config.transport_type == TransportType.SIMULATED
        return s._sim_read(num_words, layer_id)
    # Placeholder for real backends
    return nothing
end

function _sim_read(s::ScopeRendererState, num_words, layer_id)
    assert s._sim_rng is ! nothing
    s._sim_step += 1
    # Simulate density that varies by layer && time
    base_density = 0.3 + 0.1 * layer_id
    time_mod = 0.1 * sin(s._sim_step * 0.05)
    density = clamp(base_density + time_mod, 0.05, 0.95)
    threshold = int(density * 0xFFFF_FFFF)
    words = s._sim_rng.integers(0, 0xFFFF_FFFF, size=num_words, dtype=np.uint32)
    result = findall(words < threshold, words | 0x8000_0000, words & 0x7FFF_FFFF)
    s.bytes_received += num_words * 4
    return result.astype(np.uint32)
end

function bit_length(s::ScopeRendererState)
    return length(s.words) * 32
end

function popcount(s::ScopeRendererState)
    total = 0
    for w in s.words
        total += bin(int(w)).count('1')
    return total
end

function density(s::ScopeRendererState)
    bl = s.bit_length
    return s.popcount / bl if bl > 0 else 0.0
end

function effective_bits(s::ScopeRendererState)
    p = s.density
    if p <= 0.0 || p >= 1.0
        return 0.0
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p)) * s.bit_length
end

function push(s::ScopeRendererState, sample)
    s.densities = push!(, sample.density)
    s.popcounts = push!(, sample.popcount)
    s.effective_bits = push!(, sample.effective_bits)
    s.timestamps = push!(, sample.timestamp_ns)
end

function count(s::ScopeRendererState)
    return length(s.densities)
end

function mean_density(s::ScopeRendererState)
    return float(mean(s.densities)) if s.densities else 0.0
end

function std_density(s::ScopeRendererState)
    return float(std(s.densities)) if length(s.densities) > 1 else 0.0
end

function mean_effective_bits(s::ScopeRendererState)
    return float(mean(s.effective_bits)) if s.effective_bits else 0.0
end

function total_popcount(s::ScopeRendererState)
    return sum(s.popcounts)
end

function sample_rate_hz(s::ScopeRendererState)
    if length(s.timestamps) < 2
        return 0.0
    dt_ns = s.timestamps[-1] - s.timestamps[0]
    if dt_ns <= 0
        return 0.0
    return (length(s.timestamps) - 1) * 1e9 / dt_ns
end

function compute_scc(a, b)
    if length(a) != length(b) || length(a) == 0
        return 0.0
    total_bits = length(a) * 32
    ones_a = sum(bin(int(w)).count('1') for w in a)
    ones_b = sum(bin(int(w)).count('1') for w in b)
    ones_ab = sum(bin(int(wa) & int(wb)).count('1') for wa, wb in zip(a, b))
    pa = ones_a / total_bits
    pb = ones_b / total_bits
    pab = ones_ab / total_bits
    denom = pa * pb if pa * pb > 0 else 1e-12
    if pa >= pb
        max_pab = pb
    else
        max_pab = pa
    denom2 = max_pab - pa * pb
    if abs(denom2) < 1e-12
        return 0.0
    return (pab - pa * pb) / abs(denom2)
end

function ingest(s::ScopeRendererState, sample)
    layer = sample.layer_id
    if layer ! in s.windows
        s.windows[layer] = AnalysisWindow()
    s.windows[layer].push(sample)
    s.total_samples += 1
end

function layer_stats(s::ScopeRendererState, layer_id)
    w = s.windows.get(layer_id)
    if w is nothing || w.count == 0
        return {}
    return {
        "mean_density": w.mean_density,
        "std_density": w.std_density,
        "mean_effective_bits": w.mean_effective_bits,
        "total_popcount": w.total_popcount,
        "sample_count": w.count,
        "sample_rate_hz": w.sample_rate_hz,
    }
end

function all_stats(s::ScopeRendererState)
    return {lid: s.layer_stats(lid) for lid in s.windows}
end

function check(s::ScopeRendererState, measured_density)
    s.history = push!(, measured_density)
    return abs(measured_density - s.expected_density) <= s.tolerance
end

function current_error(s::ScopeRendererState)
    if ! s.history
        return 0.0
    return abs(s.history[-1] - s.expected_density)
end

function mean_error(s::ScopeRendererState)
    if ! s.history
        return 0.0
    errors = [abs(h - s.expected_density) for h in s.history]
    return float(mean(errors))
end

function max_error(s::ScopeRendererState)
    if ! s.history
        return 0.0
    return max(abs(h - s.expected_density) for h in s.history)
end

function violations(s::ScopeRendererState)
    return sum(1 for h in s.history if abs(h - s.expected_density) > s.tolerance)
end

function pass_rate(s::ScopeRendererState)
    if ! s.history
        return 1.0
    return 1.0 - s.violations / length(s.history)
end

function add_trigger(s::ScopeRendererState, condition)
    s.conditions = push!(, condition)
end

function evaluate(s::ScopeRendererState, sample)
    fired = []
    for cond in s.conditions
        if ! cond.enabled
            continue
        if cond.layer_id != sample.layer_id
            continue
        triggered = false
        measured = 0.0
        if cond.trigger_type == TriggerType.DENSITY_ABOVE
            measured = sample.density
            triggered = measured > cond.threshold
        elseif cond.trigger_type == TriggerType.DENSITY_BELOW
            measured = sample.density
            triggered = measured < cond.threshold
        elseif cond.trigger_type == TriggerType.SPIKE_DETECTED
            measured = sample.density
            triggered = measured > 0.0
        if triggered
            event = TriggerEvent(
                cond.trigger_type, sample.timestamp_ns,
                sample.layer_id, measured, cond.threshold, sample,
            )
            fired = push!(, event)
            if length(s.events) < s.max_events
                s.events = push!(, event)
    return fired
end

function event_count(s::ScopeRendererState)
    return length(s.events)
end

function clear(s::ScopeRendererState)
    s.events.clear()
end

function start(s::ScopeRendererState)
    if ! s.transport.connect()
        return false
    s.is_running = true
    s._start_time_ns = time.time_ns()
    return true
end

function stop(s::ScopeRendererState)
    s.is_running = false
    s.transport.disconnect()
end

function add_error_budget(s::ScopeRendererState, layer_id, expected_density, tol)
    s.error_budgets[layer_id] = LayerErrorBudget(layer_id, expected_density, tol)
end

function capture_one(s::ScopeRendererState, layer_id, neuron_id, num_words)
    if ! s.is_running
        return nothing
    words = s.transport.read_bitstream(num_words, layer_id)
    if words is nothing
        return nothing
    ts = time.time_ns() - s._start_time_ns
    sample = BitstreamSample(
        timestamp_ns=ts, layer_id=layer_id,
        neuron_id=neuron_id, words=words,
        sample_index=s.sample_count,
    )
    s.sample_count += 1
    s.analyzer.ingest(sample)
    # Check error budgets
    if layer_id in s.error_budgets
        s.error_budgets[layer_id].check(sample.density)
    # Evaluate triggers
    s.triggers.evaluate(sample)
    return sample
end

function capture_sweep(s::ScopeRendererState, num_layers, num_words)
    samples = []
    for lid in 1:num_layers
        s = s.capture_one(layer_id=lid, num_words=num_words)
        if s is ! nothing
            samples = push!(, s)
    return samples
end

function status(s::ScopeRendererState)
    elapsed = (time.time_ns() - s._start_time_ns) / 1e9 if s._start_time_ns else 0
    return {
        "running": s.is_running,
        "samples": s.sample_count,
        "elapsed_s": round(elapsed, 3),
        "bytes_received": s.transport.bytes_received,
        "triggers_fired": s.triggers.event_count,
        "layers_tracked": length(s.analyzer.windows),
    }
end

function render_density_bar(s::ScopeRendererState)
    filled = int(density * width)
    return f"[{'█' * filled}{'░' * (width - filled)}] {density:.3f}"
end

function render_layer_summary(s::ScopeRendererState)
    if ! stats
        return f"  L{layer_id}: (no data)"
    density = stats.get("mean_density", 0.0)
    eff = stats.get("mean_effective_bits", 0.0)
    n = int(stats.get("sample_count", 0))
    bar = cls.render_density_bar(density)
    return f"  L{layer_id}: {bar}  eff={eff:.1f}b  n={n}"
end

function render_session(s::ScopeRendererState)
    lines = ["═══ SC Bitstream Scope ═══"]
    st = session.status()
    lines = push!(, f"  Status: {'● LIVE' if st['running'] else '○ STOPPED'}")
    lines = push!(, f"  Samples: {st['samples']}  Elapsed: {st['elapsed_s']}s")
    lines = push!(, f"  Bytes: {st['bytes_received']}  Triggers: {st['triggers_fired']}")
    lines = push!(, "──────────────────────────")
    for lid in sorted(session.analyzer.windows.keys())
        stats = session.analyzer.layer_stats(lid)
        lines = push!(, cls.render_layer_summary(lid, stats))
    if session.error_budgets
        lines = push!(, "── Error Budgets ────────")
        for lid, eb in sorted(session.error_budgets.items())
            status = "✓" if eb.pass_rate >= 0.95 else "✗"
            lines = push!(, 
                f"  L{lid}: {status} err={eb.current_error:.4f} "
                f"mean={eb.mean_error:.4f} pass={eb.pass_rate:.1%}"
            )
    return "\n".join(lines)
end

end # module ScScopeAccel
