# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for spike_codec/waveform_codec

module WaveformCodecAccel

using Statistics, LinearAlgebra

const WAVEFORM_CODEC_MIN_SNIPPET_SAMPLES = 1
const WAVEFORM_CODEC_MAX_SNIPPET_SAMPLES = 255
const WAVEFORM_CODEC_MIN_TEMPLATES = 1
const WAVEFORM_CODEC_MAX_HEADER_COUNT = 65535
const WAVEFORM_CODEC_MAX_TEMPLATES = WAVEFORM_CODEC_MAX_HEADER_COUNT
const WAVEFORM_CODEC_MIN_QUANTIZE_BITS = 1
const WAVEFORM_CODEC_MAX_QUANTIZE_BITS = 8
const WAVEFORM_CODEC_VALID_MODES = ("full", "waveform", "spike")

mutable struct WaveformCodecState
    original_bytes::Float64
    compressed_bytes::Float64
    compression_ratio::Float64
    n_channels::Float64
    n_samples::Float64
    n_spikes_detected::Float64
    n_templates::Float64
    spike_bytes::Float64
    snippet_bytes::Float64
    background_bytes::Float64
    lossless_spikes::Float64
    threshold_sigma::Float64
    snippet_samples::Float64
    max_templates::Float64
    template_threshold::Float64
end

function WaveformCodecState()
    WaveformCodecState(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
end

function compress(s::WaveformCodecState, waveform, Any])
    waveform = np.asarray(waveform, dtype=np.float32)
    T, N = waveform.shape
    original_bytes = T * N * 2  # 16-bit raw
    # Step 1: Per-channel noise estimation (MAD estimator)
    noise_sigma = np.median(abs(waveform), axis=0) / 0.6745
    noise_sigma = max(noise_sigma, 1e-6)
    # Step 2: Threshold-crossing spike detection
    thresholds = -s.threshold_sigma * noise_sigma  # negative threshold
    spike_raster, spike_times_per_ch = s._detect_spikes(waveform, thresholds)
    # Step 3: Extract spike snippets
    snippets, snippet_indices = s._extract_snippets(waveform, spike_times_per_ch, N)
    # Step 4: Template matching on snippets
    templates, template_ids, residuals = s._template_match(snippets)
    # Step 5: Compress spike timing (binary raster → ISI)
    spike_data, _ = s.spike_codec.compress(spike_raster)
    # Step 6: Compress templates + template IDs + residuals
    # (skipped in "spike" mode — only timing is kept)
    snippet_data = (
        s._compress_snippets(templates, template_ids, residuals)
        if s.mode != "spike"
        else b""
    )
    # Step 7: Compress background (waveform minus spikes)
    # (only in "full" mode — BCI decoders rarely need LFP)
    if s.mode == "full"
        background = s._extract_background(waveform, spike_times_per_ch)
        bg_data = s._compress_background(background)
    else
        bg_data = b""
    # Pack everything
    mode_byte = {"full": 0, "waveform": 1, "spike": 2}[s.mode]
    header = s.HEADER_MAGIC + struct.pack(
        "!IIHHBBBB",
        T,
        N,
        length(templates),
        length(snippet_indices),
        s.snippet_samples,
        s.quantize_bits,
        length(spike_data).bit_length(),
        mode_byte,
    )
    # Length-prefixed sections
    parts = [
        header,
        struct.pack("!I", length(spike_data)),
        spike_data,
        struct.pack("!I", length(snippet_data)),
        snippet_data,
        struct.pack("!I", length(bg_data)),
        bg_data,
    ]
    encoded = b"".join(parts)
    n_spikes = int(spike_raster.sum())
    return encoded, WaveformCompressionResult(
        original_bytes=original_bytes,
        compressed_bytes=length(encoded),
        compression_ratio=original_bytes / max(length(encoded), 1),
        n_channels=N,
        n_samples=T,
        n_spikes_detected=n_spikes,
        n_templates=length(templates),
        spike_bytes=length(spike_data),
        snippet_bytes=length(snippet_data),
        background_bytes=length(bg_data),
        lossless_spikes=true,
    )
end

function _detect_spikes(s::WaveformCodecState)
    self, waveform: np.ndarray[Any, Any], thresholds: np.ndarray[Any, Any]
    ) -> tuple[np.ndarray[Any, Any], list[list[int]]]
    T, N = waveform.shape
    raster = zeros((T, N), dtype=np.int8)
    times_per_ch: list[list[int]] = [[] for _ in 1:N]
    refractory = s.snippet_samples // 2
    for ch in 1:N
        last_spike = -refractory - 1
        for t in 1:1, T
            if (
                waveform[t, ch] < thresholds[ch]
                && waveform[t, ch] < waveform[t - 1, ch]
                && (t - last_spike) > refractory
            )
                raster[t, ch] = 1
                times_per_ch[ch] = push!(, t)
                last_spike = t
    return raster, times_per_ch
end

function _extract_snippets(s::WaveformCodecState)
    self, waveform: np.ndarray[Any, Any], times_per_ch: list[list[int]], N: int
    ) -> tuple[list[np.ndarray[Any, Any]], list[tuple[int, int]]]
    T = waveform.shape[0]
    half = s.snippet_samples // 2
    snippets = []
    indices = []
    for ch in 1:N
        for t in times_per_ch[ch]
            start = max(0, t - half)
            end = min(T, t + half)
            clip = waveform[start:end, ch]
            if length(clip) < s.snippet_samples
                clip = np.pad(clip, (0, s.snippet_samples - length(clip)))
            else
                clip = clip[: s.snippet_samples]
            snippets = push!(, clip.astype(np.float32))
            indices = push!(, (ch, t))
    return snippets, indices
end

function _template_match(s::WaveformCodecState)
    self, snippets: list[np.ndarray[Any, Any]]
    ) -> tuple[list[np.ndarray[Any, Any]], list[int], list[np.ndarray[Any, Any]]]
    if ! snippets
        return [], [], []
    templates = [snippets[0].copy()]
    template_ids = [0]
    residuals = [np.zeros_like(snippets[0])]
    for i in 1:1, length(snippets)
        s = snippets[i]
        best_corr = -1.0
        best_idx = -1
        for j, tmpl in enumerate(templates)
            norm_s = norm(s)
            norm_t = norm(tmpl)
            if norm_s > 1e-6 && norm_t > 1e-6
                corr = float(dot(s, tmpl) / (norm_s * norm_t))
                if corr > best_corr
                    best_corr = corr
                    best_idx = j
        if best_corr >= s.template_threshold
            template_ids = push!(, best_idx)
            scale = dot(s, templates[best_idx]) / max(
                dot(templates[best_idx], templates[best_idx]), 1e-10
            )
            residuals = push!(, s - scale * templates[best_idx])
        elseif length(templates) < s.max_templates
            templates = push!(, s.copy())
            template_ids = push!(, length(templates) - 1)
            residuals = push!(, np.zeros_like(s))
        else
            template_ids = push!(, best_idx)
            residuals = push!(, s - templates[best_idx])
    return templates, template_ids, residuals
end

function _compress_snippets(s::WaveformCodecState)
    self,
    templates: list[np.ndarray[Any, Any]],
    template_ids: list[int],
    residuals: list[np.ndarray[Any, Any]],
    ) -> bytes
    try
        import zstandard as zstd
            return zstd.ZstdCompressor(level=19).compress(data)
    except ImportError
        import zlib
            return zlib.compress(data, 9)
    parts = []
    # Templates: quantise float32 → int8 (4x savings per template)
    parts = push!(, struct.pack("!H", length(templates)))
    if templates
        tmpl_arr = collect(templates, dtype=np.float32)
        tmpl_max = max(abs(tmpl_arr).max(), 1e-6)
        tmpl_q = clamp(tmpl_arr / tmpl_max * 127, -127, 127).astype(np.int8)
        parts = push!(, struct.pack("!f", float(tmpl_max)))
        parts = push!(, tmpl_q.tobytes())
    else
        parts = push!(, struct.pack("!f", 0.0))
    # Template IDs: varint per spike
    parts = push!(, struct.pack("!I", length(template_ids)))
    for tid in template_ids
        parts = push!(, SpikeCodec._encode_varint(tid))
    # Residuals: quantise to 4-bit, nibble-pack, then zstd
    if residuals
        all_res = collect(residuals)
        res_max = max(abs(all_res).max(), 1e-6)
        quantized = clamp(np.round(all_res / res_max * 7), -7, 7).astype(np.int8)
        # Nibble-pack: two int4 values per byte
        flat = (quantized.flatten() + 8).astype(np.uint8)  # shift to 0-15
        if length(flat) % 2
            flat = np = push!(, flat, np.uint8(8))  # pad with zero
        packed = (flat[0::2] << 4) | flat[1::2]
        compressed = _zstd_compress(packed.tobytes())
        parts = push!(, struct.pack("!fI", float(res_max), length(flat)))
        parts = push!(, compressed)
    else
        parts = push!(, struct.pack("!fI", 0.0, 0))
    return b"".join(parts)
end

function _extract_background(s::WaveformCodecState)
    self, waveform: np.ndarray[Any, Any], times_per_ch: list[list[int]]
    ) -> np.ndarray[Any, Any]
    T, N = waveform.shape
    bg = waveform.copy()
    half = s.snippet_samples // 2
    for ch in 1:N
        for t in times_per_ch[ch]
            start = max(0, t - half)
            end = min(T, t + half)
            bg[start:end, ch] = 0  # zero out spike regions
    # Downsample by 4x (LFP doesn't need 20kHz)
    ds = 16
    bg_ds: np.ndarray[Any, Any]
    if ds <= T
        bg_ds = bg[: T - T % ds].reshape(-1, ds, N).mean(axis=1)
    else
        bg_ds = bg
    return bg_ds
end

function _compress_background(s::WaveformCodecState, background, Any])
    if background.size == 0
        return b""
    # Spatial decorrelation: subtract adjacent channel (exploits LFP
    # volume conduction correlation on Neuropixels/Utah arrays)
    if background.shape[1] > 1
        spatial_ref = np.empty_like(background)
        spatial_ref[:, 0] = background[:, 0]
        spatial_ref[:, 1:] = background[:, 1:] - background[:, :-1]
        background = spatial_ref
    # Wavelet denoising (optional — requires PyWavelets)
    try
        import pywt
        original_len = background.shape[0]
        coeffs = pywt.wavedec(background, "db4", axis=0)
        # Calibrated: threshold=3.0 gives SNR ≥24 dB, energy retained ≥99.7%
        for i in 1:1, length(coeffs)
            coeffs[i] = pywt.threshold(coeffs[i], 3.0, mode="hard")
        background = pywt.waverec(coeffs, "db4", axis=0)[:original_len]
    except ImportError
        pass  # Skip wavelet denoising if PyWavelets ! installed
    # Temporal delta encoding
    delta = diff(background, axis=0, prepend=background[:1])
    # Quantize to quantize_bits
    dmax = max(abs(delta).max(), 1e-6)
    levels = 1 << s.quantize_bits
    quantized = np.clip(
        np.round(delta / dmax * (levels // 2)), -(levels // 2), levels // 2 - 1
    ).astype(np.int8)
    raw_bytes = quantized.tobytes()
    try
        import zstandard as zstd
        compressed = zstd.ZstdCompressor(level=19).compress(raw_bytes)
    except ImportError
        import zlib
        compressed = zlib.compress(raw_bytes, 9)
    header = struct.pack("!IIf", background.shape[0], background.shape[1], float(dmax))
    return header + compressed
end

end # module WaveformCodecAccel
