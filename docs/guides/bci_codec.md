# Spike Codec Library

Spike train compression for neural recording, BCI telemetry, and neuromorphic routing.

## The Problem

| System | Channels | Raw Rate | Uplink | Gap |
|--------|----------|----------|--------|-----|
| Neuralink N1 (2026) | 1,024 | 200 Mbps | 10-20 Mbps | 10-20x |
| Neuralink next-gen | 3,000-10,000 | 600-2000 Mbps | 10-20 Mbps | 60-200x |
| Neuropixels 2.0 | 384 | 30 Mbps | storage | archival |
| Loihi 2 inter-chip | variable | event-based | NoC | routing overhead |
| Closed-loop BCI | 256-1024 | 200 Mbps | on-chip | <1ms latency |

## Five Codecs, One API

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

| Codec | Best For | Strategy | Compression |
|-------|----------|----------|-------------|
| `isi` | General purpose | ISI + LEB128 varint | 50-200x (sparse) |
| `predictive` | BCI implants | EMA predictor + XOR errors | 10-15x + structure |
| `delta` | Neural probes | Inter-channel XOR residuals | 2-5x over ISI (correlated) |
| `streaming` | Real-time BCI | Fixed-latency bitmask frames | bounded worst-case |
| `aer` | Neuromorphic | Event list (timestamp, neuron_id) | 40x+ (sparse) |

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

Only transmit surprises. EMA predictor learns per-channel firing rates.
XOR actual vs predicted → compress only error bits.

```
Encoder:                          Decoder:
  predict → XOR → ISI encode       ISI decode → XOR → recover
      ↑                                              ↑
      └── update(actual)                 update(recovered) ──┘
```

Encoder and decoder run identical predictors. Deterministic, no state sync.

```python
from sc_neurocore.spike_codec import PredictiveSpikeCodec

codec = PredictiveSpikeCodec(alpha=0.005, threshold=0.5)
data, result = codec.compress(spikes)
print(f"{result.compression_ratio:.1f}x, accuracy: {result.prediction_accuracy:.1%}")
```

### Hardware Mapping

- EMA update: one MAC per channel per timestep (or shift-add in fixed-point)
- Threshold compare: one comparator per channel
- XOR: one gate per channel per timestep
- ISI encoder: counter + LEB128 shift register per channel
- Verilog building blocks in `hdl/`: `sc_aer_encoder.v`, `sc_lif_neuron.v`

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
