# Neural Data Compression Library

Two compression layers: raw waveform (10-bit ADC) and spike raster (binary).

## The Problem

| System | Channels | Raw Rate | Uplink | Gap | Solution |
|--------|----------|----------|--------|-----|----------|
| Neuralink N1 | 1,024 | 205 Mbps | 10-20 Mbps | 10-20x | WaveformCodec (see measured results) |
| Neuralink next-gen | 3,072 | 614 Mbps | 10-20 Mbps | 30-60x | WaveformCodec + better bg |
| Neuropixels 2.0 | 384 | 77 Mbps | storage | archival | WaveformCodec `lossless` |
| Closed-loop BCI | 256-1024 | 51-205 Mbps | on-chip | <1ms | StreamingCodec |

## WaveformCodec: Raw Electrode Compression (NEW)

End-to-end pipeline from raw 10-bit ADC to compressed telemetry bytes:

```
Raw ADC (T, N, 10-bit)
    |
    +-- Spike Detection (MAD noise est. + threshold crossing)
    |       -> spike_times, spike_snippets
    |
    +-- Spike Timing -> ISI + Huffman (LOSSLESS)
    |       -> ~0.3% of compressed output
    |
    +-- Spike Snippets -> Template Library + Quantized Residuals
    |       -> ~7% of compressed output (16 templates learned)
    |
    +-- Background LFP -> 4x downsample + delta + quantize + zlib
            -> ~92% of compressed output
```

```python
from sc_neurocore.spike_codec import WaveformCodec

codec = WaveformCodec(
    threshold_sigma=4.5,   # finite positive spike threshold
    snippet_samples=48,    # waveform clip size, 1-255 samples
    max_templates=16,      # template library size, 1-65535 entries
    quantize_bits=4,       # background quantization, 1-8 bits
)

# Compress raw electrode data
data, result = codec.compress(raw_waveform)  # (T, N) int16/float
print(f"{result.compression_ratio:.1f}x, {result.n_spikes_detected} spikes detected")
```

`raw_waveform` must be a finite, non-empty two-dimensional array with time on
axis 0 and channels on axis 1. NaN/Inf samples, vectors, empty recordings, and
header-unsafe constructor ranges fail before compression state is derived.

### Modes and decoding

| Mode | Kept | Exact? |
|------|------|--------|
| `spike` | spike times | spike times exact; no waveform |
| `waveform` | spike times + template-coded snippets | lossy |
| `full` (default) | + downsampled background | lossy |
| `lossless` | every int16 sample | exact |

```python
codec = WaveformCodec(mode="lossless")
data, result = codec.compress(raw_int16)          # integer samples in the int16 range
decoded = WaveformCodec.decompress(data)          # any mode
assert decoded.lossless_waveform and (decoded.waveform == raw_int16).all()
```

`lossless` maps each channel onto its ADC levels (an affine lattice, or the
table of values actually used when a 10-bit ADC is scaled to 16 bits by a
non-integer factor), predicts with the best fixed polynomial of order 0-3 and
codes residuals with block-adaptive Rice codes. Streams (format `WFC2`) end
with a CRC-32; a corrupted stream raises `ValueError` instead of decoding.
`decompress(..., max_samples=...)` (default `2**28`) refuses headers declaring
more samples before allocating.

### Rust kernels

Noise thresholds, spike detection, template matching, background extraction
and quantisation, and the lossless coder run in Rust (`sc_neurocore_engine`)
when the installed engine provides them; `backend="auto"` (default) selects
them, `backend="python"` forces the NumPy reference and `backend="rust"`
refuses to run without them. Both backends write byte-identical streams: the
float32 operations are the same IEEE operations, rounding is half to even, and
dot products are correctly rounded sums (`math.fsum` in Python, the same
algorithm in Rust). The zstd/zlib byte compressors and the optional wavelet
denoising stay in Python.

### Measured on the Neuralink compression-challenge data

The public challenge data (`data.zip`, 743 single-electrode WAV files, about
62.6 minutes at 19,531 Hz, 10-bit levels in 16-bit samples, 146,800,526
bytes) was retrieved from the Internet Archive copy of the original Neuralink
URL; its SHA-1 matches the archive's record of the original. The challenge
asks for lossless compression; its ratio is original file bytes divided by
compressed bytes. Measured 2026-10-22 on one workstation, one file at a time:

| Codec | Ratio | Exact |
|-------|-------|-------|
| `WaveformCodec(mode="lossless")` (WAV header stored verbatim) | **3.33x** | all 743 files byte-identical after decoding |
| `xz -9e` | 2.75x | yes |
| `zstd -19` | 2.48x | yes |
| `WaveformCodec(mode="full")`, median-centred input | 48.3x | no: RMS error 551 ADC units against a signal RMS of 3,234 |
| `WaveformCodec(mode="spike")`, median-centred input | 3,118x | spike times only (7,729 spikes detected in 62.6 min) |

Time for the whole 62.6 minutes of recordings on the same workstation
(Python 3.12, one core), both backends producing identical bytes:

| Step | NumPy backend | Rust backend |
|------|---------------|--------------|
| `lossless` encode | 86.4 s | 10.1 s |
| `lossless` decode | 6.2 s | 1.6 s |
| `full` encode | 41.4 s | 1.2 s |
| `spike` encode | 10.5 s | 0.38 s |

The lossy ratios are not comparable with the challenge, which requires every
sample back. The spike-mode ratio depends on the detector: without removing
each file's DC offset the detector finds only 1,864 spikes in the same data.

### Measured Results (synthetic 1024-channel, 1 second at 20 kHz)

`examples/demo_waveform_codec.py --channels 1024 --duration 1.0`, `full` mode,
format `WFC2`, measured 2026-10-22 (the previous format `WFCX` gave 89.7x on
the same synthetic data; the earlier 24x figure on this page was stale).
Encoding one second of 1024 channels took 1.11 s with the Rust backend and
7.9 s with the NumPy backend; in the Rust run about 0.5 s is zstd level 19
in Python:

| Metric | Value |
|--------|-------|
| Raw data | 40,960,000 bytes (328 Mbit) |
| Compressed | 442,002 bytes (3.5 Mbit) |
| **Compression ratio** | **92.7x** (lossy) |
| Spikes detected | 1,931 |
| Templates learned | 16 |
| Spike timing | exact |
| Bluetooth capacity | 15 Mbit/s |
| **Fits in uplink** | **YES (24% used)** |

The demo's channel-count table scales the measured 1024-channel ratio
linearly; it is an extrapolation, not a measurement per channel count. On
real recordings the ratio is far lower (see the Neuralink measurement above).

### Run the Demo

```bash
python examples/demo_waveform_codec.py --channels 1024 --duration 1.0
```

## Six Codecs, One API

```python
from sc_neurocore.spike_codec import get_codec, recommend_codec, list_codecs

# Auto-select based on your system
name = recommend_codec(n_channels=1024, firing_rate=2.0, latency_ms=5.0)
codec = get_codec(name)

# Or pick directly
codec = get_codec("predictive", alpha=0.005)

# All codecs: compress(spikes) → (bytes, result), decompress(bytes, T, N) → spikes
data, result = codec.compress(spikes)
recovered = codec.decompress(data, T, N)
```

| Codec | Best For | Strategy | Measured Compression |
|-------|----------|----------|---------------------|
| `isi` (auto entropy) | General purpose | ISI + varint/Huffman | 401x at 0.1%, 8.8x at 30% |
| `predictive` (context) | Structured data | Markov context + XOR | 25.5x on bursty (3x over ISI) |
| `predictive` (lfsr) | Hardware BCI | Q8.8 LFSR, bit-true Verilog | Same ratio, ASIC-deployable |
| `delta` | Neural probes | Inter-channel XOR residuals | 8.2x on correlated (70% over ISI) |
| `streaming` | Real-time BCI | Fixed-latency bitmask frames | Bounded worst-case |
| `aer` (adaptive) | Neuromorphic | Event list, auto-invert >50% | Format-compatible with Loihi/SpiNNaker |

### Competitive Benchmarks (measured, all lossless)

ISI codec with auto entropy selection beats zlib-9 at every firing rate:

| Firing Rate | ISI (auto) | zlib-9 | lzma | Advantage |
|-------------|-----------|--------|------|-----------|
| 0.1% | **401x** | 359x | 194x | +12% over zlib |
| 1% | **78x** | 65x | 48x | +20% over zlib |
| 5% | **24x** | 19x | 20x | +28% over zlib |
| 10% | **16x** | 12x | 13x | +30% over zlib |
| 30% | **8.8x** | 7.0x | 7.8x | +24% over zlib |

Context predictor on structured data (periodic bursts):

| Predictor | Ratio | Accuracy |
|-----------|-------|----------|
| ISI (no prediction) | 8.6x | — |
| EMA | 8.5x | 90% |
| **Context (Markov)** | **25.5x** | 97.8% |

Realistic SpikeInterface benchmarks (physiological spike trains):

| Scenario | Best Codec | Ratio |
|----------|-----------|-------|
| Neuropixels 10 units 1-5 Hz | ISI | 457x |
| BCI-scale 50 units 0.5-3 Hz | ISI | 756x |
| High-density 100 units 1-10 Hz | ISI | 317x |

These are spike-raster ratios (binary events already detected). They are not
comparable with Neuralink's compression challenge, which asks for lossless
compression of the raw electrode samples (see the measured results above).

## ISI Codec (Baseline)

Inter-spike interval encoding with LEB128 variable-length integers. Per-neuron
spike times → differences → varint bytes. Exploits sparsity: cortical neurons
fire at 0.5-5 Hz, so >99.9% of time bins are zeros.

```python
from sc_neurocore.spike_codec import SpikeCodec

codec = SpikeCodec(mode="lossless")  # or "lossy" with timing_precision
data, result = codec.compress(spikes)
print(result.summary())
```

## Predictive Codec (BCI Implants)

Only transmit surprises. Two predictor modes:

- **`ema`** (default): float EMA rate tracking + threshold comparison. Simple, fast.
- **`lfsr`**: Q8.8 fixed-point rate + LFSR comparator. **Bit-true with `sc_bitstream_encoder.v`** — the prediction logic maps directly to Verilog RTL. No float arithmetic, no multipliers. Same LFSR polynomial as the hardware (x^16 + x^14 + x^13 + x^11 + 1).

```
Encoder:                          Decoder:
  predict → XOR → ISI encode       ISI decode → XOR → recover
      ↑                                              ↑
      └── update(actual)                 update(recovered) ──┘
```

Encoder and decoder run identical predictors. Deterministic, no state sync.

```python
from sc_neurocore.spike_codec import PredictiveSpikeCodec

# Float EMA mode (default)
codec = PredictiveSpikeCodec(alpha=0.005, threshold=0.5)
data, result = codec.compress(spikes)
print(f"{result.compression_ratio:.1f}x, accuracy: {result.prediction_accuracy:.1%}")

# SC-native LFSR mode (bit-true with Verilog RTL)
codec_hw = PredictiveSpikeCodec(predictor="lfsr", alpha_q8=1, seed=0xACE1)
data_hw, result_hw = codec_hw.compress(spikes)
print(f"LFSR: {result_hw.compression_ratio:.1f}x")
```

### Hardware Mapping

**LFSR mode maps 1:1 to existing Verilog RTL:**

| Operation | Verilog Module | Gates (1024ch) |
|-----------|---------------|----------------|
| LFSR pseudo-random | `sc_bitstream_encoder.v` | ~2K |
| Q8.8 rate update | shift-add accumulator | ~10K |
| Comparator (LFSR < rate) | comparator bank | ~1K |
| XOR (actual vs predicted) | XOR gate array | ~1K |
| ISI encoder | counter + shift register | ~30K |
| **Total** | | **~44K** (excl. SRAM) |

The LFSR predictor uses the same polynomial (x^16 + x^14 + x^13 + x^11 + 1)
and step semantics as `sc_bitstream_encoder.v`. Python prediction = Verilog
prediction, bit-for-bit. Formally verified through the HDL SymbiYosys
inventory: 90 proof jobs and 447 formal statements.

## Delta Codec (Neural Probes)

Exploits spatial correlation on probe arrays. Groups channels, picks reference
(highest spike count), XOR-encodes others as delta residuals.

```python
from sc_neurocore.spike_codec import DeltaSpikeCodec

# Neuropixels: 384 channels, nearby electrodes correlated
codec = DeltaSpikeCodec(group_size=8)
data, result = codec.compress(spikes)
print(f"{result.compression_ratio:.1f}x, delta sparsity: {result.mean_delta_sparsity:.1%}")
```

## Streaming Codec (Real-Time)

Fixed-size time windows, each independently decodable. Bounded worst-case
latency = window_size / sample_rate.

```python
from sc_neurocore.spike_codec import StreamingSpikeCodec

# 1ms windows at 20kHz = 20 samples per frame
codec = StreamingSpikeCodec(window_size=20)
data, result = codec.compress(spikes)

# Frame-level API for real-time use
frame = codec.compress_frame(window)  # single window
recovered = codec.decompress_frame(frame)
```

## AER Codec (Neuromorphic)

Address-Event Representation: compact (timestamp_delta, neuron_id) event stream.
Compatible with `comm/aer_udp.py` protocol. Delta-encodes timestamps for
compression. O(n_spikes) bytes.

```python
from sc_neurocore.spike_codec import AERSpikeCodec

codec = AERSpikeCodec()
data, result = codec.compress(spikes)
print(f"{result.compression_ratio:.1f}x, {result.n_events} events, "
      f"{result.bytes_per_event:.1f} bytes/event")
```

## Codec Selection Guide

```python
from sc_neurocore.spike_codec import recommend_codec

# Auto-recommend based on constraints
name = recommend_codec(
    n_channels=1024,
    firing_rate=2.0,        # Hz per neuron
    latency_ms=5.0,         # max acceptable latency
    correlated=False,       # nearby channels correlated?
    neuromorphic=False,     # target is neuromorphic hardware?
)
```

Decision logic:

1. Neuromorphic target → `aer`
2. Latency ≤ 1ms → `streaming`
3. Correlated channels, N ≥ 16 → `delta`
4. High channel count (N ≥ 64) → `predictive`
5. Default → `isi`

## API Reference

::: sc_neurocore.spike_codec.registry
    options:
      show_source: true
      members:
        - get_codec
        - list_codecs
        - recommend_codec
