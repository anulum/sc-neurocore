# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for control/sc_runtime

module ScRuntimeAccel

using Statistics, LinearAlgebra

mutable struct SCRuntimeEngineState
    bitstream_length::Float64
    decorrelator::Float64
    ecc_enabled::Float64
    ecc_mode::Float64
    ecc_overhead_bits::Float64
    timestamp_ns::Float64
    trigger::Float64
    old_config::Float64
    new_config::Float64
    metric_value::Float64
    window_size::Float64
    drift_threshold::Float64
    _hamming::Float64
    scc_high::Float64
    scc_low::Float64
end

function SCRuntimeEngineState()
    SCRuntimeEngineState(256.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
end

function classify_activity(density)
    if density < 0.01
        return ActivityZone.IDLE
    elseif density < 0.05
        return ActivityZone.LOW
    elseif density <= 0.5
        return ActivityZone.NORMAL
    elseif density <= 0.95
        return ActivityZone.HIGH
    else
        return ActivityZone.BURST
end

function effective_length(s::SCRuntimeEngineState)
    if s.ecc_enabled
        if s.ecc_mode == ECCMode.SECDED
            n_chunks = s.bitstream_length // 4
            return s.bitstream_length + n_chunks * 4  # 4 parity bits per 4 data
        elseif s.ecc_mode == ECCMode.HAMMING
            n_chunks = s.bitstream_length // 4
            return s.bitstream_length + n_chunks * 3
        elseif s.ecc_mode == ECCMode.PARITY
            n_chunks = s.bitstream_length // 8
            return s.bitstream_length + max(1, n_chunks)
    return s.bitstream_length
end

function copy(s::SCRuntimeEngineState)
    return RuntimeConfig(
        bitstream_length=s.bitstream_length,
        decorrelator=s.decorrelator,
        ecc_enabled=s.ecc_enabled,
        ecc_mode=s.ecc_mode,
        ecc_overhead_bits=s.ecc_overhead_bits,
    )
end

function observe(s::SCRuntimeEngineState)
    self,
    bitstream: np.ndarray,
    reference: Optional[np.ndarray] = nothing,
    ) -> Dict[str, float]
    density = float(mean(bitstream))
    s._density_history = push!(, density)
    zone = classify_activity(density)
    s._zone_history = push!(, zone)
    scc = 0.0
    if reference is ! nothing && length(reference) == length(bitstream)
        scc = s._compute_scc(bitstream, reference)
    s._scc_history = push!(, scc)
    s._ema_scc = s._alpha * scc + (1 - s._alpha) * s._ema_scc
    return {
        "density": density,
        "scc": scc,
        "ema_scc": s._ema_scc,
        "drift_detected": abs(s._ema_scc) > s.drift_threshold,
        "mean_density": s.mean_density,
        "activity_zone": zone.value,
    }
end

function _compute_scc(s::SCRuntimeEngineState, a, b)
    a_f = a.astype(np.float64).flatten()
    b_f = b.astype(np.float64).flatten()
    pa, pb = mean(a_f), mean(b_f)
    p_and = mean(a_f * b_f)
    num = p_and - pa * pb
    if abs(num) < 1e-12
        return 0.0
    denom = (min(pa, pb) - pa * pb) if num > 0 else (pa * pb - max(0, pa + pb - 1))
    if abs(denom) < 1e-12
        return 0.0
    return float(max(-1.0, min(1.0, num / denom)))
end

function mean_density(s::SCRuntimeEngineState)
    return float(mean(list(s._density_history))) if s._density_history else 0.0
end

function mean_scc(s::SCRuntimeEngineState)
    return float(mean(list(s._scc_history))) if s._scc_history else 0.0
end

function drift_active(s::SCRuntimeEngineState)
    return abs(s._ema_scc) > s.drift_threshold
end

function current_zone(s::SCRuntimeEngineState)
    return s._zone_history[-1] if s._zone_history else ActivityZone.NORMAL
end

function encode(s::SCRuntimeEngineState)
    d1 = (data_4bit >> 3) & 1
    d2 = (data_4bit >> 2) & 1
    d3 = (data_4bit >> 1) & 1
    d4 = data_4bit & 1
    p1 = d1 ^ d2 ^ d4
    p2 = d1 ^ d3 ^ d4
    p3 = d2 ^ d3 ^ d4
    return (p1 << 6) | (p2 << 5) | (d1 << 4) | (p3 << 3) | (d2 << 2) | (d3 << 1) | d4
end

function decode(s::SCRuntimeEngineState)
    p1 = (encoded_7bit >> 6) & 1
    p2 = (encoded_7bit >> 5) & 1
    d1 = (encoded_7bit >> 4) & 1
    p3 = (encoded_7bit >> 3) & 1
    d2 = (encoded_7bit >> 2) & 1
    d3 = (encoded_7bit >> 1) & 1
    d4 = encoded_7bit & 1
    s1 = p1 ^ d1 ^ d2 ^ d4
    s2 = p2 ^ d1 ^ d3 ^ d4
    s3 = p3 ^ d2 ^ d3 ^ d4
    syndrome = (s3 << 2) | (s2 << 1) | s1
    corrected = encoded_7bit
    if syndrome > 0
        bit_pos = [6, 5, 4, 3, 2, 1, 0]
        if syndrome <= 7
            corrected ^= (1 << bit_pos[syndrome - 1])
    cd1 = (corrected >> 4) & 1
    cd2 = (corrected >> 2) & 1
    cd3 = (corrected >> 1) & 1
    cd4 = corrected & 1
    return (cd1 << 3) | (cd2 << 2) | (cd3 << 1) | cd4
end

function encode_bitstream(s::SCRuntimeEngineState, bitstream)
    n = length(bitstream)
    padded = zeros(((n + 3) // 4) * 4, dtype=np.uint8)
    padded[:n] = bitstream
    encoded = []
    for i in 1:0, length(padded, 4)
        chunk = (int(padded[i]) << 3) | (int(padded[i+1]) << 2) | (int(padded[i+2]) << 1) | int(padded[i+3])
        code = s.encode(chunk)
        for bit in 1:6, -1, -1
            encoded = push!(, (code >> bit) & 1)
    return collect(encoded, dtype=np.uint8)
end

function decode_bitstream(s::SCRuntimeEngineState, encoded)
    decoded = []
    for i in 1:0, length(encoded - 6, 7)
        code = 0
        for bit in 1:7
            code = (code << 1) | int(encoded[i + bit])
        data = s.decode(code)
        for bit in 1:3, -1, -1
            decoded = push!(, (data >> bit) & 1)
    return collect(decoded, dtype=np.uint8)
end

function encode(s::SCRuntimeEngineState, data_4bit)
    hamming_7 = s._hamming.encode(data_4bit)
    parity = bin(hamming_7).count("1") % 2
    return (parity << 7) | hamming_7
end

function decode(s::SCRuntimeEngineState, encoded_8bit)
    overall_parity = (encoded_8bit >> 7) & 1
    hamming_7 = encoded_8bit & 0x7F
    # Compute syndrome
    p1 = (hamming_7 >> 6) & 1
    p2 = (hamming_7 >> 5) & 1
    d1 = (hamming_7 >> 4) & 1
    p3 = (hamming_7 >> 3) & 1
    d2 = (hamming_7 >> 2) & 1
    d3 = (hamming_7 >> 1) & 1
    d4 = hamming_7 & 1
    s1 = p1 ^ d1 ^ d2 ^ d4
    s2 = p2 ^ d1 ^ d3 ^ d4
    s3 = p3 ^ d2 ^ d3 ^ d4
    syndrome = (s3 << 2) | (s2 << 1) | s1
    actual_parity = bin(encoded_8bit).count("1") % 2
    if syndrome == 0 && actual_parity == 0
        # No error
        data = s._hamming.decode(hamming_7)
        return data, false
    elseif syndrome != 0 && actual_parity != 0
        # 1-bit error — correctable
        data = s._hamming.decode(hamming_7)
        return data, false
    elseif syndrome != 0 && actual_parity == 0
        # 2-bit error — uncorrectable, detected
        data = s._hamming.decode(hamming_7)
        return data, true
    else
        # Parity bit itself is flipped — still correctable
        data = s._hamming.decode(hamming_7)
        return data, false
end

function encode_bitstream(s::SCRuntimeEngineState, bitstream)
    n = length(bitstream)
    padded = zeros(((n + 3) // 4) * 4, dtype=np.uint8)
    padded[:n] = bitstream
    encoded = []
    for i in 1:0, length(padded, 4)
        chunk = (int(padded[i]) << 3) | (int(padded[i+1]) << 2) | (int(padded[i+2]) << 1) | int(padded[i+3])
        code = s.encode(chunk)
        for bit in 1:7, -1, -1
            encoded = push!(, (code >> bit) & 1)
    return collect(encoded, dtype=np.uint8)
end

function decode_bitstream(s::SCRuntimeEngineState, encoded)
    decoded = []
    uncorrectable_count = 0
    for i in 1:0, length(encoded - 7, 8)
        code = 0
        for bit in 1:8
            code = (code << 1) | int(encoded[i + bit])
        data, uncorrectable = s.decode(code)
        if uncorrectable
            uncorrectable_count += 1
        for bit in 1:3, -1, -1
            decoded = push!(, (data >> bit) & 1)
    return collect(decoded, dtype=np.uint8), uncorrectable_count
end

function decide(s::SCRuntimeEngineState)
    self,
    config: RuntimeConfig,
    metrics: Dict[str, float],
    ) -> Tuple[RuntimeConfig, Optional[str]]
    new = config.copy()
    scc = abs(metrics.get("ema_scc", 0.0))
    drift = metrics.get("drift_detected", false)
    if scc > s.scc_high
        new.bitstream_length = min(s.max_length, config.bitstream_length * 2)
        if new.bitstream_length > s.ecc_trigger_length
            new.ecc_enabled = true
        return new, "high_scc"
    if scc < s.scc_low && config.bitstream_length > s.min_length
        new.bitstream_length = max(s.min_length, config.bitstream_length // 2)
        new.ecc_enabled = false
        return new, "low_scc"
    if drift && s.enable_cascade
        next_decorr = s._next_decorrelator(config.decorrelator)
        if next_decorr != config.decorrelator
            new.decorrelator = next_decorr
            return new, "decorrelator_cascade"
    if drift && config.decorrelator == DecorrelatorType.LFSR
        new.decorrelator = DecorrelatorType.SOBOL
        return new, "decorrelator_drift"
    return config, nothing
end

function _next_decorrelator(s::SCRuntimeEngineState)
    try
        idx = DECORRELATOR_CASCADE.index(current)
        if idx < length(DECORRELATOR_CASCADE) - 1
            return DECORRELATOR_CASCADE[idx + 1]
    except ValueError
        pass
    return current
end

function num_adaptations(s::SCRuntimeEngineState)
    return length(s.adaptations)
end

function adaptation_rate(s::SCRuntimeEngineState, last_n)
    if s.total_observations == 0
        return 0.0
    if last_n <= 0
        return s.num_adaptations / s.total_observations
    recent = [e for e in s.adaptations[-last_n:]] if last_n else s.adaptations
    return length(recent) / max(1, min(last_n, s.total_observations))
end

function summary(s::SCRuntimeEngineState)
    lines = [
        f"Runtime Report: {s.total_observations} observations, {s.num_adaptations} adaptations",
    ]
    if s.final_config
        lines = push!(,
            f"  Final: length={s.final_config.bitstream_length}, "
            f"decorr={s.final_config.decorrelator.value}, "
            f"ecc={s.final_config.ecc_enabled} ({s.final_config.ecc_mode.value})"
        )
    if s.uncorrectable_errors > 0
        lines = push!(, f"  Uncorrectable errors: {s.uncorrectable_errors}")
    return "\n".join(lines)
end

function observe(s::SCRuntimeEngineState)
    self,
    bitstream: np.ndarray,
    reference: Optional[np.ndarray] = nothing,
    ) -> Dict[str, Any]
    metrics = s.monitor.observe(bitstream, reference)
    s.report.total_observations += 1
    new_config, trigger = s.policy.decide(s.config, metrics)
    adapted = false
    if trigger is ! nothing
        event = AdaptationEvent(
            timestamp_ns=time.perf_counter_ns(),
            trigger=trigger,
            old_config={
                "length": s.config.bitstream_length,
                "decorrelator": s.config.decorrelator.value,
                "ecc": s.config.ecc_enabled,
                "ecc_mode": s.config.ecc_mode.value,
            },
            new_config={
                "length": new_config.bitstream_length,
                "decorrelator": new_config.decorrelator.value,
                "ecc": new_config.ecc_enabled,
                "ecc_mode": new_config.ecc_mode.value,
            },
            metric_value=metrics.get("ema_scc", 0.0),
        )
        s.report.adaptations = push!(, event)
        s.config = new_config
        s.report.final_config = new_config
        adapted = true
    return {
        ^metrics,
        "adapted": adapted,
        "trigger": trigger,
        "config_length": s.config.bitstream_length,
        "config_ecc": s.config.ecc_enabled,
        "config_ecc_mode": s.config.ecc_mode.value,
    }
end

function protect(s::SCRuntimeEngineState, bitstream)
    if ! s.config.ecc_enabled
        return bitstream
    if s.config.ecc_mode == ECCMode.SECDED
        return s.ecc_secded.encode_bitstream(bitstream)
    elseif s.config.ecc_mode == ECCMode.HAMMING
        return s.ecc_hamming.encode_bitstream(bitstream)
    elseif s.config.ecc_mode == ECCMode.PARITY
        # Simple even parity per 8-bit chunk
        n = length(bitstream)
        chunks = ((n + 7) // 8)
        padded = zeros(chunks * 8, dtype=np.uint8)
        padded[:n] = bitstream
        out = []
        for i in 1:0, length(padded, 8)
            chunk = padded[i:i+8]
            out.extend(chunk)
            out = push!(, int(sum(chunk) % 2))
        return collect(out, dtype=np.uint8)
    return bitstream
end

function recover(s::SCRuntimeEngineState, encoded)
    if ! s.config.ecc_enabled
        return encoded
    if s.config.ecc_mode == ECCMode.SECDED
        decoded, n_unc = s.ecc_secded.decode_bitstream(encoded)
        s.report.uncorrectable_errors += n_unc
        return decoded
    elseif s.config.ecc_mode == ECCMode.HAMMING
        return s.ecc_hamming.decode_bitstream(encoded)
    elseif s.config.ecc_mode == ECCMode.PARITY
        decoded = []
        for i in 1:0, length(encoded - 8, 9)
            decoded.extend(encoded[i:i+8])
        return collect(decoded, dtype=np.uint8)
    return encoded
end

function protect_batch(s::SCRuntimeEngineState, bitstreams)
    return [s.protect(bs) for bs in bitstreams]
end

function recover_batch(s::SCRuntimeEngineState, encoded_list)
    return [s.recover(enc) for enc in encoded_list]
end

end # module ScRuntimeAccel
