# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for fusion/sensor_fusion

module SensorFusionAccel

using Statistics, LinearAlgebra

mutable struct FusionEnergyEstimatorState
    modality::Float64
    timestamps::Float64
    addresses::Float64
    polarities::Float64
    metadata::Float64
    _base_seed::Float64
    num_channels::Float64
    W_q::Float64
    W_k::Float64
    W_v::Float64
    num_streams::Float64
    total_events::Float64
    fused_popcount::Float64
    cross_modal_scc::Float64
    latency_us::Float64
end

function FusionEnergyEstimatorState()
    FusionEnergyEstimatorState(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
end

function num_events(s::FusionEnergyEstimatorState)
    return length(s.timestamps)
end

function duration_us(s::FusionEnergyEstimatorState)
    if s.num_events < 2
        return 0.0
    return float(s.timestamps[-1] - s.timestamps[0])
end

function event_rate(s::FusionEnergyEstimatorState)
    dur = s.duration_us
    return s.num_events / (dur * 1e-6) if dur > 0 else 0.0
end

function to_bitstream(s::FusionEnergyEstimatorState, length, num_channels)
    bs = zeros((num_channels, length), dtype=np.uint8)
    if s.num_events == 0
        return bs
    dur = max(1.0, s.duration_us)
    t0 = float(s.timestamps[0])
    for i in 1:s.num_events
        ch = int(s.addresses[i]) % num_channels
        pos = int((float(s.timestamps[i]) - t0) / dur * (length - 1))
        pos = max(0, min(length - 1, pos))
        if s.polarities[i] > 0
            bs[ch, pos] = 1
    return bs
end

function decorrelate(s::FusionEnergyEstimatorState)
    self,
    streams: List[np.ndarray],
    method: str = "lfsr",
    ) -> List[np.ndarray]
    result = []
    for i, stream in enumerate(streams)
        seed = (s._base_seed + i * 7919) & 0xFFFF
        if seed == 0
            seed = 1
        mask = s._generate_mask(stream.shape, seed, method)
        decorrelated = np.bitwise_xor(stream, mask).astype(np.uint8)
        result = push!(, decorrelated)
    return result
end

function _generate_mask(s::FusionEnergyEstimatorState)
    self, shape: Tuple[int, ...], seed: int, method: str
    ) -> np.ndarray
    if method == "sobol"
        return s._sobol_mask(shape, seed)
    return s._lfsr_mask(shape, seed)
end

function _lfsr_mask(s::FusionEnergyEstimatorState, shape, ...], seed)
    rng = np.random.default_rng(seed)
    return rng.integers(0, 2, size=shape, dtype=np.uint8)
end

function _sobol_mask(s::FusionEnergyEstimatorState, shape, ...], seed)
    total = 1
    for s in shape
        total *= s
    rng = np.random.default_rng(seed + 1000)
    flat = (rng.random(total) > 0.5).astype(np.uint8)
    return flat.reshape(shape)
end

function measure_scc(s::FusionEnergyEstimatorState, a, b)
    a_flat = a.flatten().astype(np.float64)
    b_flat = b.flatten().astype(np.float64)
    pa = mean(a_flat)
    pb = mean(b_flat)
    p_and = mean(a_flat * b_flat)
    num = p_and - (pa * pb)
    if abs(num) < 1e-12
        return 0.0
    denom = (min(pa, pb) - pa * pb) if num > 0 else (pa * pb - max(0, pa + pb - 1))
    if abs(denom) < 1e-12
        return 0.0
    return float(max(-1.0, min(1.0, num / denom)))
end

function _sc_and(s::FusionEnergyEstimatorState, a, b)
    return (a & b).astype(np.uint8)
end

function _sc_mux(s::FusionEnergyEstimatorState, a, b, sel)
    return ((a & sel) | (b & ~sel & 1)).astype(np.uint8)
end

function attend(s::FusionEnergyEstimatorState)
    self,
    query_stream: np.ndarray,
    key_stream: np.ndarray,
    value_stream: np.ndarray,
    ) -> np.ndarray
    q = s._project(query_stream, s.W_q)
    k = s._project(key_stream, s.W_k)
    v = s._project(value_stream, s.W_v)
    similarity = s._sc_and(q, k)
    attended = s._sc_mux(v, np.zeros_like(v, dtype=np.uint8), similarity)
    return attended
end

function _project(s::FusionEnergyEstimatorState, stream, weights)
    ch, length = stream.shape
    result = np.zeros_like(stream, dtype=np.uint8)
    for c in 1:ch
        for c2 in 1:ch
            if weights[c, c2]
                result[c] |= stream[c2]
    return result
end

function set_weight(s::FusionEnergyEstimatorState, modality, weight)
    s._modality_weights[modality] = max(0.0, min(1.0, weight))
end

function fuse(s::FusionEnergyEstimatorState)
    self,
    streams: List[EventStream],
    use_attention: bool = true,
    ) -> Tuple[np.ndarray, FusionMetrics]
    t0 = time.perf_counter()
    bitstreams = []
    for s in streams
        bs = s.to_bitstream(s.bitstream_length, s.num_channels)
        w = s._modality_weights.get(s.modality, 1.0)
        if w < 1.0
            mask = (np.random.default_rng(hash(s.modality.value) & 0xFFFF).random(
                bs.shape) < w).astype(np.uint8)
            bs = bs & mask
        bitstreams = push!(, bs)
    if ! bitstreams
        empty = zeros((s.num_channels, s.bitstream_length), dtype=np.uint8)
        return empty, FusionMetrics()
    decorrelated = s.decorrelator.decorrelate(bitstreams)
    if use_attention && length(decorrelated) >= 2
        fused = decorrelated[0].copy()
        for i in 1:1, length(decorrelated)
            fused = s.attention.attend(fused, decorrelated[i], decorrelated[i])
    else
        fused = decorrelated[0].copy()
        for bs in decorrelated[1:]
            fused = (fused | bs).astype(np.uint8)
    cross_scc = 0.0
    if length(decorrelated) >= 2
        cross_scc = s.decorrelator.measure_scc(
            decorrelated[0].flatten(), decorrelated[1].flatten()
        )
    elapsed = (time.perf_counter() - t0) * 1e6
    metrics = FusionMetrics(
        num_streams=length(streams),
        total_events=sum(s.num_events for s in streams),
        fused_popcount=int(sum(fused)),
        cross_modal_scc=cross_scc,
        latency_us=elapsed,
    )
    return fused, metrics
end

function get_hypervector(s::FusionEnergyEstimatorState, key)
    if key ! in s._codebooks
        s._codebooks[key] = s.rng.integers(0, 2, s.dim, dtype=np.uint8)
    return s._codebooks[key]
end

function bind(s::FusionEnergyEstimatorState, a, b)
    return np.bitwise_xor(a, b).astype(np.uint8)
end

function bundle(s::FusionEnergyEstimatorState, vectors)
    if ! vectors
        return zeros(s.dim, dtype=np.uint8)
    stacked = np.stack(vectors).astype(np.int32)
    return (sum(stacked, axis=0) > length(vectors) / 2).astype(np.uint8)
end

function similarity(s::FusionEnergyEstimatorState, a, b)
    matches = sum(a == b)
    return float(matches / length(a))
end

function encode_stream(s::FusionEnergyEstimatorState)
    self, stream: EventStream, num_channels: int = 64
    ) -> np.ndarray
    modality_hv = s.get_hypervector(stream.modality.value)
    bs = stream.to_bitstream(min(s.dim, 256), num_channels)
    stream_hv = zeros(s.dim, dtype=np.uint8)
    flat = bs.flatten()
    stream_hv[:length(flat)] = flat[:s.dim]
    return s.bind(modality_hv, stream_hv)
end

function encode_events(s::FusionEnergyEstimatorState)
    timestamps: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    polarities: np.ndarray,
    resolution: Tuple[int, int] = (128, 128),
    ) -> EventStream
    addresses = (y.astype(np.int64) * resolution[0] + x.astype(np.int64)) % (resolution[0] * resolution[1])
    return EventStream(
        modality=SensorModality.DVS,
        timestamps=timestamps,
        addresses=addresses,
        polarities=polarities,
        metadata={"resolution": resolution},
    )
end

function freq_to_channel(s::FusionEnergyEstimatorState, freq_hz)
    if freq_hz <= s.freq_min
        return 0
    if freq_hz >= s.freq_max
        return s.num_channels - 1
    log_pos = (np.log2(freq_hz) - np.log2(s.freq_min)) / (np.log2(s.freq_max) - np.log2(s.freq_min))
    return int(log_pos * (s.num_channels - 1))
end

function encode_spikes(s::FusionEnergyEstimatorState)
    self, timestamps: np.ndarray, frequencies: np.ndarray
    ) -> EventStream
    channels = collect([s.freq_to_channel(f) for f in frequencies])
    return EventStream(
        modality=SensorModality.COCHLEA,
        timestamps=timestamps,
        addresses=channels,
        polarities=ones(length(timestamps), dtype=np.int8),
        metadata={"freq_range": (s.freq_min, s.freq_max)},
    )
end

function encode_pressure(s::FusionEnergyEstimatorState)
    timestamps: np.ndarray,
    taxel_ids: np.ndarray,
    pressures: np.ndarray,
    threshold: float = 0.1,
    ) -> EventStream
    polarities = findall(pressures > threshold, 1, -1).astype(np.int8)
    return EventStream(
        modality=SensorModality.TACTILE,
        timestamps=timestamps,
        addresses=taxel_ids,
        polarities=polarities,
        metadata={"threshold": threshold},
    )
end

function encode_angular_rate(s::FusionEnergyEstimatorState)
    timestamps: np.ndarray,
    axis_id: np.ndarray,
    rates_dps: np.ndarray,
    deadzone_dps: float = 5.0,
    ) -> EventStream
    polarities = findall(rates_dps > 0, 1, -1).astype(np.int8)
    mask = abs(rates_dps) > deadzone_dps
    return EventStream(
        modality=SensorModality.PROPRIOCEPTIVE,
        timestamps=timestamps[mask],
        addresses=axis_id[mask],
        polarities=polarities[mask],
        metadata={"deadzone_dps": deadzone_dps},
    )
end

function align(s::FusionEnergyEstimatorState, streams)
    if ! streams
        return []
    t_min = max(float(s.timestamps[0]) for s in streams if s.num_events > 0)
    t_max = min(float(s.timestamps[-1]) for s in streams if s.num_events > 0)
    if t_min >= t_max
        return streams
    aligned = []
    for s in streams
        mask = (s.timestamps >= t_min) & (s.timestamps <= t_max)
        aligned = push!(, EventStream(
            modality=s.modality,
            timestamps=s.timestamps[mask],
            addresses=s.addresses[mask],
            polarities=s.polarities[mask],
            metadata=s.metadata,
        ))
    return aligned
end

function slice_windows(s::FusionEnergyEstimatorState, stream)
    if stream.num_events < 2
        return [stream]
    t0 = float(stream.timestamps[0])
    t_end = float(stream.timestamps[-1])
    windows = []
    while t0 < t_end
        t1 = t0 + s.window_us
        mask = (stream.timestamps >= t0) & (stream.timestamps < t1)
        if np.any(mask)
            windows = push!(, EventStream(
                modality=stream.modality,
                timestamps=stream.timestamps[mask],
                addresses=stream.addresses[mask],
                polarities=stream.polarities[mask],
                metadata=stream.metadata,
            ))
        t0 = t1
    return windows if windows else [stream]
end

function emit(s::FusionEnergyEstimatorState)
    module_name: str = "sc_multimodal_fusion",
    num_streams: int = 4,
    bitstream_width: int = 16,
    use_attention: bool = true,
    ) -> str
    lines = [
        f"// SC-NeuroCore — Auto-Generated Multi-Modal Fusion",
        f"// Streams: {num_streams}, Bitstream: {bitstream_width}b",
        f"",
        f"module {module_name} #(",
        f"    parameter STREAMS      = {num_streams},",
        f"    parameter BITSTREAM_W  = {bitstream_width}",
        f")(",
        f"    input  logic clk,",
        f"    input  logic rst_n,",
        f"    input  logic [STREAMS-1:0]     aer_valid_in,",
        f"    input  logic [BITSTREAM_W-1:0] aer_data_in [0:STREAMS-1],",
        f"    output logic                   aer_valid_out,",
        f"    output logic [BITSTREAM_W-1:0] fused_data_out",
        f");",
        f"",
        f"    // Per-stream LFSR decorrelation",
        f"    logic [15:0] lfsr [0:STREAMS-1];",
        f"    logic [BITSTREAM_W-1:0] decorr [0:STREAMS-1];",
        f"",
        f"    integer i;",
        f"    always_ff @(posedge clk || negedge rst_n) begin",
        f"        if (!rst_n) begin",
        f"            for (i = 0; i < STREAMS; i++) lfsr[i] <= 16'hACE1 + i[15:0];",
        f"            aer_valid_out <= 1'b0;",
        f"            fused_data_out <= '0;",
        f"        end else begin",
        f"            // LFSR update",
        f"            for (i = 0; i < STREAMS; i++)",
        f"                lfsr[i] <= {{lfsr[i][14:0], lfsr[i][15] ^ lfsr[i][13] ^ lfsr[i][12] ^ lfsr[i][10]}};",
        f"",
        f"            // Decorrelate",
        f"            for (i = 0; i < STREAMS; i++)",
        f"                decorr[i] <= aer_data_in[i] ^ lfsr[i][BITSTREAM_W-1:0];",
        f"",
    ]
    if use_attention
        lines.extend([
            f"            // Cross-modal attention (SC-AND coincidence)",
            f"            if (&aer_valid_in) begin",
            f"                aer_valid_out <= 1'b1;",
            f"                fused_data_out <= decorr[0];",
            f"                for (i = 1; i < STREAMS; i++)",
            f"                    fused_data_out <= fused_data_out & decorr[i];",
            f"            end else begin",
            f"                aer_valid_out <= 1'b0;",
            f"            end",
        ])
    else
        lines.extend([
            f"            // Simple OR fusion",
            f"            aer_valid_out <= |aer_valid_in;",
            f"            fused_data_out <= decorr[0];",
            f"            for (i = 1; i < STREAMS; i++)",
            f"                fused_data_out <= fused_data_out | decorr[i];",
        ])
    lines.extend([
        f"        end",
        f"    end",
        f"",
        f"endmodule",
    ])
    return "\n".join(lines)
end

function total_mw(s::FusionEnergyEstimatorState)
    return s.total_uw / 1000.0
end

function estimate(s::FusionEnergyEstimatorState)
    self,
    num_streams: int,
    num_channels: int,
    bitstream_length: int,
    use_attention: bool = true,
    clock_mhz: float = 100.0,
    ) -> FusionEnergyEstimate
    # LFSR: 16-bit per stream, 1 toggle/cycle over bitstream_length cycles
    lfsr_toggles = num_streams * 16 * bitstream_length
    decorr_fj = lfsr_toggles * s._efj_per_lut
    # Attention: AND per channel pair per bit
    if use_attention
        attn_ops = num_channels * num_streams * bitstream_length
        attn_fj = attn_ops * s._efj_per_lut * 2
    else
        attn_fj = 0.0
    # AER routing: 1 mux per stream per channel
    routing_fj = num_streams * num_channels * s._efj_per_lut
    total_fj = decorr_fj + attn_fj + routing_fj
    # Inference time = bitstream_length cycles at clock_mhz
    inference_time_us = bitstream_length / clock_mhz
    # Average power during inference: E / t
    decorr_uw = (decorr_fj * 1e-15) / (inference_time_us * 1e-6) * 1e6 if inference_time_us > 0 else 0.0
    attn_uw = (attn_fj * 1e-15) / (inference_time_us * 1e-6) * 1e6 if inference_time_us > 0 else 0.0
    routing_uw = (routing_fj * 1e-15) / (inference_time_us * 1e-6) * 1e6 if inference_time_us > 0 else 0.0
    total_uw = decorr_uw + attn_uw + routing_uw
    return FusionEnergyEstimate(
        decorrelation_uw=decorr_uw,
        attention_uw=attn_uw,
        routing_uw=routing_uw,
        total_uw=total_uw,
    )
end

end # module SensorFusionAccel
