# Neural Data Compression Library

Two compression layers: raw waveform (10-bit ADC) and spike raster (binary).

## The Problem

| System | Channels | Raw Rate | Uplink | Gap | Solution |
|--------|----------|----------|--------|-----|----------|
| Neuralink N1 | 1,024 | 205 Mbps | 10-20 Mbps | 10-20x | **WaveformCodec (25x)** |
| Neuralink next-gen | 3,072 | 614 Mbps | 10-20 Mbps | 30-60x | WaveformCodec + better bg |
| Neuropixels 2.0 | 384 | 77 Mbps | storage | archival | WaveformCodec (25x) |
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

### Measured Results (synthetic 1024-channel, 1 second at 20 kHz)

| Metric | Value |
|--------|-------|
| Raw data | 40,960,000 bytes (328 Mbit) |
| Compressed | 1,703,435 bytes (13.6 Mbit) |
| **Compression ratio** | **24x** |
| Spikes detected | 3,087 |
| Templates learned | 16 |
| Spike timing | LOSSLESS |
| Bluetooth capacity | 15 Mbit/s |
| **Fits in uplink** | **YES (9% used)** |

### Scaling

| Channels | Raw Mbit/s | Compressed Mbit/s | Fits Bluetooth |
|----------|-----------|-------------------|---------------|
| 128 | 26 | 1.0 | YES |
| 256 | 51 | 2.0 | YES |
| 384 | 77 | 3.0 | YES |
| 1024 | 205 | 8.0 | YES |
| 3072 | 614 | 23.9 | NO |

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

All above Neuralink's 200x target.

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
inventory: 18 proof jobs and 130 formal statements.

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
