# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for bioware/bioware

module BiowareAccel

using Statistics, LinearAlgebra

mutable struct HomeostaticPlasticityState
    layout::Float64
    num_channels::Float64
    sample_rate_hz::Float64
    voltage_gain::Float64
    noise_floor_uv::Float64
    spike_threshold_sigma::Float64
    electrode_pitch_um::Float64
    channel::Float64
    timestamp_s::Float64
    amplitude_uv::Float64
    unit_id::Float64
    waveform::Float64
    config::Float64
    refractory_samples::Float64
    _noise_estimates::Float64
end

function HomeostaticPlasticityState()
    HomeostaticPlasticityState(0.0, 60.0, 20000.0, 1000.0, 5.0, 5.0, 200.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 30.0, 0.0)
end

function from_layout(s::HomeostaticPlasticityState)
    presets = {
        MEALayout.MEA_60: dict(num_channels=60, electrode_pitch_um=200.0),
        MEALayout.MEA_120: dict(num_channels=120, electrode_pitch_um=100.0),
        MEALayout.MEA_256: dict(num_channels=256, electrode_pitch_um=100.0),
        MEALayout.MEA_4096: dict(num_channels=4096, electrode_pitch_um=17.5),
        MEALayout.CUSTOM: dict(num_channels=60, electrode_pitch_um=200.0),
    }
    return cls(layout=layout, ^presets[layout])
end

function estimate_noise(s::HomeostaticPlasticityState, voltage_data)
    mad = np.median(abs(voltage_data), axis=0) / 0.6745
    s._noise_estimates = mad
    return mad
end

function detect(s::HomeostaticPlasticityState, voltage_data)
    n_samples, n_channels = voltage_data.shape
    if s._noise_estimates is nothing
        s.estimate_noise(voltage_data)
    assert s._noise_estimates is ! nothing
    spikes = []
    dt = 1.0 / s.config.sample_rate_hz
    sigma = s.config.spike_threshold_sigma
    for ch in 1:n_channels
        threshold = sigma * s._noise_estimates[ch]
        above = abs(voltage_data[:, ch]) > threshold
        crossings = findall(diff(above.astype(int)) == 1)[0]
        last_spike_idx = -s.refractory_samples - 1
        for idx in crossings
            if idx - last_spike_idx < s.refractory_samples
                continue
            last_spike_idx = idx
            amp = float(voltage_data[idx, ch])
            ts = idx * dt
            spikes = push!(,
                DetectedSpike(
                    channel=ch,
                    timestamp_s=ts,
                    amplitude_uv=amp,
                    unit_id=ch,
                )
            )
    return spikes
end

function transcode(s::HomeostaticPlasticityState)
    self,
    spikes: List[DetectedSpike],
    t_start_s: float = 0.0,
    ) -> List[AEREvent]
    events = []
    for spike in spikes
        neuron_id = s._map_channel(spike.channel)
        ts_hw = int((spike.timestamp_s - t_start_s) * s.hw_clock_hz) & 0xFFFF
        events = push!(,
            AEREvent(
                neuron_id=neuron_id,
                timestamp=ts_hw,
                valid=true,
            )
        )
    # Sort by timestamp (AER is time-ordered)
    events.sort(key=lambda e: e.timestamp)
    return events
end

function _map_channel(s::HomeostaticPlasticityState, channel)
    if s.channel_map is ! nothing
        return s.channel_map.get(channel, channel)
    return channel
end

function convert(s::HomeostaticPlasticityState, events)
    # Count events per neuron in the window
    counts: Dict[int, int] = {}
    for e in events
        if e.valid
            counts[e.neuron_id] = counts.get(e.neuron_id, 0) + 1
    max_count = max(counts.values()) if counts else 1
    bitstreams = {}
    for nid, count in counts.items()
        prob = count / max_count
        bitstreams[nid] = s._lfsr_encode(prob, nid)
    return bitstreams
end

function _lfsr_encode(s::HomeostaticPlasticityState, probability, neuron_id)
    threshold = int(clamp(probability, 0.0, 1.0) * 65535)
    seed = (s.lfsr_seed + neuron_id * 7919) & 0xFFFF
    if seed == 0
        seed = 1
    reg = seed
    bits = zeros(s.bitstream_length, dtype=np.uint8)
    for i in 1:s.bitstream_length
        bits[i] = 1 if reg < threshold else 0
        feedback = ((reg >> 15) ^ (reg >> 13) ^ (reg >> 12) ^ (reg >> 10)) & 1
        reg = ((reg << 1) | feedback) & 0xFFFF
    return bits
end

function encode(s::HomeostaticPlasticityState)
    self,
    bitstreams: Dict[int, np.ndarray],
    t_start_ms: float = 0.0,
    ) -> List[OptogeneticPulse]
    pulses = []
    total_power = 0.0
    for nid, bs in sorted(bitstreams.items())
        density = float(sum(bs)) / length(bs) if length(bs) > 0 else 0.0
        if density < 0.01
            continue
        intensity = density * s.max_intensity_mw_mm2
        if total_power + intensity > s.max_total_power_mw
            break
        total_power += intensity
        duration = s.min_pulse_ms + density * (s.max_pulse_ms - s.min_pulse_ms)
        onset = t_start_ms + nid * s.clock_period_ms
        pulses = push!(,
            OptogeneticPulse(
                channel=nid,
                onset_ms=onset,
                duration_ms=duration,
                intensity_mw_mm2=intensity,
                wavelength_nm=s.wavelength_nm,
            )
        )
    return pulses
end

function compute_dw(s::HomeostaticPlasticityState, dt_ms)
    if dt_ms > 0
        return s.a_plus * exp(-dt_ms / s.tau_plus_ms)
    elseif dt_ms < 0
        return -s.a_minus * exp(dt_ms / s.tau_minus_ms)
    return 0.0
end

function update_weight(s::HomeostaticPlasticityState, current_q88, dt_ms)
    dw = s.compute_dw(dt_ms)
    dw_q88 = int(dw * 256)  # Convert to Q8.8
    new_w = current_q88 + dw_q88
    return max(s.w_min_q88, min(s.w_max_q88, new_w))
end

function update_theta(s::HomeostaticPlasticityState, post_rate_hz, dt_ms)
    alpha = dt_ms / s.tau_theta_ms
    target = post_rate_hz^2
    s.theta += alpha * (target - s.theta)
    return s.theta
end

function compute_dw(s::HomeostaticPlasticityState, pre_rate_hz, post_rate_hz)
    return s.learning_rate * pre_rate_hz * post_rate_hz * (post_rate_hz - s.theta)
end

function update_weight(s::HomeostaticPlasticityState, current_q88, pre_rate, post_rate)
    dw = s.compute_dw(pre_rate, post_rate)
    dw_q88 = int(dw * 256)
    new_w = current_q88 + dw_q88
    return max(s.w_min_q88, min(s.w_max_q88, new_w))
end

function assess(s::HomeostaticPlasticityState, spike_counts, duration_s)
    rates = spike_counts / duration_s if duration_s > 0 else spike_counts
    active = sum(rates > s.min_firing_rate_hz)
    mean_rate = float(mean(rates[rates > 0])) if np.any(rates > 0) else 0.0
    bursting = sum(rates > s.burst_threshold_hz)
    health_score = 1.0
    if active < s.min_active_channels
        health_score *= active / s.min_active_channels
    if mean_rate > s.max_firing_rate_hz
        health_score *= s.max_firing_rate_hz / mean_rate
    return {
        "active_channels": int(active),
        "mean_firing_rate_hz": mean_rate,
        "bursting_channels": int(bursting),
        "health_score": float(clamp(health_score, 0.0, 1.0)),
        "is_viable": bool(health_score > 0.5),
    }
end

function process_frame(s::HomeostaticPlasticityState)
    self,
    voltage_data: np.ndarray,
    t_start_s: float = 0.0,
    ) -> Dict
    t0 = time.perf_counter_ns()
    s.round_count += 1
    # 1. Detect spikes
    spikes = s.detector.detect(voltage_data)
    # 2. Transcode to AER
    aer_events = s.transcoder.transcode(spikes, t_start_s)
    # 3. Convert to SC bitstreams
    bitstreams = s.sc_converter.convert(aer_events)
    # 4. Generate optogenetic pulses
    opto_pulses = s.opto_encoder.encode(bitstreams)
    # 5. Health assessment
    n_channels = voltage_data.shape[1]
    spike_counts = zeros(n_channels)
    for s in spikes
        if s.channel < n_channels
            spike_counts[s.channel] += 1
    duration = voltage_data.shape[0] / s.mea_config.sample_rate_hz
    health = s.health_monitor.assess(spike_counts, duration)
    latency_us = (time.perf_counter_ns() - t0) / 1000.0
    return {
        "round": s.round_count,
        "num_spikes": length(spikes),
        "num_aer_events": length(aer_events),
        "num_bitstreams": length(bitstreams),
        "num_opto_pulses": length(opto_pulses),
        "latency_us": latency_us,
        "health": health,
        "spikes": spikes,
        "aer_events": aer_events,
        "bitstreams": bitstreams,
        "opto_pulses": opto_pulses,
    }
end

function fit(s::HomeostaticPlasticityState, spikes)
    amps = collect([abs(s.amplitude_uv) for s in spikes])
    if length(amps) == 0
        s.amplitude_bins = collect([])
        return
    s.amplitude_bins = range(amps.min(), amps.max(), s.num_units + 1)
end

function assign(s::HomeostaticPlasticityState, spikes)
    if s.amplitude_bins is nothing || length(s.amplitude_bins) == 0
        return spikes
    result = []
    for s in spikes
        amp = abs(s.amplitude_uv)
        unit = int(
            np.clip(
                np.searchsorted(s.amplitude_bins, amp) - 1,
                0,
                s.num_units - 1,
            )
        )
        result = push!(,
            DetectedSpike(
                channel=s.channel,
                timestamp_s=s.timestamp_s,
                amplitude_uv=s.amplitude_uv,
                unit_id=unit,
                waveform=s.waveform,
            )
        )
    return result
end

function extract_lfp_power(voltage_data, sample_rate_hz, bands)
    voltage_data: np.ndarray,
    sample_rate_hz: float,
    bands: Optional[List[LFPBand]] = nothing,
    ) -> Dict[str, np.ndarray]
    if bands is nothing
        bands = DEFAULT_LFP_BANDS
    n_samples, n_channels = voltage_data.shape
    freqs = np.fft.rfftfreq(n_samples, d=1.0 / sample_rate_hz)
    fft_mag = abs(np.fft.rfft(voltage_data, axis=0)) ^ 2
    result = {}
    for band in bands
        mask = (freqs >= band.low_hz) & (freqs < band.high_hz)
        power = sum(fft_mag[mask, :], axis=0) if mask.any() else zeros(n_channels)
        result[band.name] = power
    return result
end

function record(s::HomeostaticPlasticityState, latency_us)
    s.history = push!(, latency_us)
    if latency_us > s.max_latency_us
        s.violations += 1
        return false
    return true
end

function mean_latency_us(s::HomeostaticPlasticityState)
    return float(mean(s.history)) if s.history else 0.0
end

function p99_latency_us(s::HomeostaticPlasticityState)
    return float(np.percentile(s.history, 99)) if s.history else 0.0
end

function compliance_ratio(s::HomeostaticPlasticityState)
    if ! s.history
        return 1.0
    return 1.0 - s.violations / length(s.history)
end

function apply(s::HomeostaticPlasticityState, t_current_s)
    s._applied_at = t_current_s
end

function effective_gain(s::HomeostaticPlasticityState, t_current_s)
    if s._applied_at < 0
        return 1.0
    elapsed = t_current_s - s._applied_at
    if elapsed < s.onset_delay_s
        frac = elapsed / s.onset_delay_s
        return 1.0 + frac * (s.gain - 1.0)
    return s.gain
end

function modulate_spikes(s::HomeostaticPlasticityState, spike_counts, t_current_s)
    g = s.effective_gain(t_current_s)
    return np.round(spike_counts * g).astype(int)
end

function label(s::HomeostaticPlasticityState)
    return f"{s.well_id}_{s.culture_type}_P{s.passage_number}"
end

function add_well(s::HomeostaticPlasticityState, well)
    s.wells = push!(, well)
end

function standard_6_well(s::HomeostaticPlasticityState)
    plate = cls()
    for i in 1:6
        plate.add_well(
            WellConfig(
                well_id=f"W{i + 1}",
                mea_config=MEAConfig.from_layout(layout),
            )
        )
    return plate
end

function num_wells(s::HomeostaticPlasticityState)
    return length(s.wells)
end

function get_well(s::HomeostaticPlasticityState, well_id)
    return next((w for w in s.wells if w.well_id == well_id), nothing)
end

function detect_network_bursts(spikes, bin_width_s, threshold_sigma, min_channels)
    spikes: List[DetectedSpike],
    bin_width_s: float = 0.01,
    threshold_sigma: float = 3.0,
    min_channels: int = 3,
    ) -> List[NetworkBurst]
    if ! spikes
        return []
    timestamps = collect([s.timestamp_s for s in spikes])
    t_start, t_end = timestamps.min(), timestamps.max()
    if t_end <= t_start
        return []
    n_bins = max(1, int((t_end - t_start) / bin_width_s) + 1)
    bin_counts = zeros(n_bins)
    bin_channels: List[set] = [set() for _ in 1:n_bins]
    for s in spikes
        idx = min(int((s.timestamp_s - t_start) / bin_width_s), n_bins - 1)
        bin_counts[idx] += 1
        bin_channels[idx].add(s.channel)
    mean_count = mean(bin_counts)
    std_count = std(bin_counts)
    if std_count == 0
        return []
    threshold = mean_count + threshold_sigma * std_count
    bursts = []
    for i in 1:n_bins
        if bin_counts[i] >= threshold && length(bin_channels[i]) >= min_channels
            bursts = push!(,
                NetworkBurst(
                    onset_s=t_start + i * bin_width_s,
                    duration_s=bin_width_s,
                    participating_channels=length(bin_channels[i]),
                    total_spikes=int(bin_counts[i]),
                )
            )
    return bursts
end

function blank(s::HomeostaticPlasticityState)
    self,
    voltage_data: np.ndarray,
    stim_times_s: List[float],
    sample_rate_hz: float,
    ) -> np.ndarray
    result = voltage_data.copy()
    pre_samples = int(s.blanking_pre_ms * sample_rate_hz / 1000.0)
    post_samples = int(s.blanking_post_ms * sample_rate_hz / 1000.0)
    for t_s in stim_times_s
        center = int(t_s * sample_rate_hz)
        start = max(0, center - pre_samples)
        end = min(result.shape[0], center + post_samples)
        result[start:end, :] = 0.0
    return result
end

function log(s::HomeostaticPlasticityState, entry)
    s.entries = push!(, entry)
end

function total_rounds(s::HomeostaticPlasticityState)
    return length(s.entries)
end

function to_list(s::HomeostaticPlasticityState)
    return [
        {
            "round": e.round_number,
            "timestamp": e.timestamp_iso,
            "spikes": e.num_spikes,
            "opto_pulses": e.num_opto_pulses,
            "latency_us": e.latency_us,
            "health_score": e.health_score,
            "notes": e.notes,
        }
        for e in s.entries
    ]
end

function checksum(s::HomeostaticPlasticityState)
    import json as _json
    data = _json.dumps(s.to_list(), sort_keys=true)
    return hashlib.sha256(data.encode()).hexdigest()
end

function decode_bitstream_rate(bitstreams, sc_clock_hz)
    bitstreams: Dict[int, np.ndarray],
    sc_clock_hz: float = 1e6,
    ) -> Dict[int, float]
    rates = {}
    for nid, bs in bitstreams.items()
        if length(bs) == 0
            rates[nid] = 0.0
            continue
        prob = float(sum(bs)) / length(bs)
        rates[nid] = prob * sc_clock_hz
    return rates
end

function update_threshold(s::HomeostaticPlasticityState)
    self,
    current_q88: int,
    observed_rate_hz: float,
    dt_ms: float,
    ) -> int
    error = observed_rate_hz - s.target_rate_hz
    alpha = dt_ms / s.tau_homeo_ms
    delta_q88 = int(alpha * error * 2.56)  # scale to Q8.8
    new_q88 = current_q88 + delta_q88
    return max(s.min_threshold_q88, min(s.max_threshold_q88, new_q88))
end

end # module BiowareAccel
