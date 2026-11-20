<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SC-NeuroCore — Spike codec tutorial -->

# Tutorial 70: Spike Codec Library

Compress spike trains for BCI telemetry, neural probes, neuromorphic routing,
and independent streaming frames. Five raster codecs share the legacy factory;
the waveform family and two stateful stream APIs have separate contracts.
Use the [typed catalogue](../api/spike_codec.md#discover-and-inspect) to discover
all six families, their variants and actual native-component availability.
The [wire reference](../reference/spike_codec_wire_formats.md) records the
metadata, state and compatibility requirements for each interface.

## The Bandwidth Problem

| System | Raw Data | Available Uplink | Required Compression |
|--------|----------|------------------|---------------------|
| Neuralink N1 (1024ch, 20kHz) | 200 Mbps | 10-20 Mbps | 10-20x |
| Neuralink next-gen (3000-10000ch) | 600-2000 Mbps | 10-20 Mbps | 60-200x |
| Neuropixels 2.0 (384ch) | 30 Mbps | disk storage | archival ratio |
| Closed-loop BCI | 200 Mbps | on-chip | <1ms latency |

## Quick Start — Select a Raster Configuration

`recommend_codec` applies a deterministic configuration heuristic. It does not
measure the data, backend speed or end-to-end latency. `firing_rate` is retained
for compatibility and does not change its selection.

```python
import numpy as np
from sc_neurocore.spike_codec import get_codec, recommend_codec

# Synthetic data: 1024 channels, 1000 timesteps, 0.5% firing
rng = np.random.RandomState(42)
spikes = (rng.random((1000, 1024)) < 0.005).astype(np.int8)

# Apply the raster configuration heuristic
name = recommend_codec(
    n_channels=1024,
    firing_rate=2.0,      # Hz per neuron
    latency_ms=5.0,       # max acceptable latency
    correlated=False,     # nearby channels correlated?
    neuromorphic=False,   # target is neuromorphic hardware?
)
print(f"Recommended: {name}")  # "predictive"

codec = get_codec(name, alpha=0.005)
data, result = codec.compress(spikes)
recovered = codec.decompress(data, 1000, 1024)
assert np.array_equal(recovered, spikes)  # lossless
print(f"{result.compression_ratio:.1f}x compression")
```

## Codec 1: ISI (Baseline)

Inter-spike interval encoding. Entropy defaults to `auto`, which selects
canonical Huffman above 3% event density and LEB128 varints otherwise.
Explicit `entropy="varint"` and `entropy="huffman"` are also available.
The default `backend="auto"` uses the Rust engine when installed and otherwise
uses Python. Both implementations read each other's packets.

```python
from sc_neurocore.spike_codec import SpikeCodec

codec = SpikeCodec(mode="lossless")
data, result = codec.compress(spikes)
print(result.summary())
# "SpikeCodec (lossless): 459.8x compression, ..."

# Lossy mode: quantize timing
codec_lossy = SpikeCodec(mode="lossy", timing_precision=5)
data_lossy, result_lossy = codec_lossy.compress(spikes)
bins = codec_lossy.decompress(data_lossy, spikes.shape[0] // 5, spikes.shape[1])
```

Each output bin covers five original samples. The encoder merges events in
the same bin and drops the incomplete final block. Keep the timing precision
with the packet so consumers can recover the original time scale.

Compare the complete packet size on your own sparse raster before choosing
an entropy mode; compression ratios depend on the recording and framing.

## Codec 2: Predictive (BCI Implants)

The default EMA predictor estimates each channel's firing rate. The codec
XORs predictions with actual spikes and compresses the residual. Bursts or
other recurring patterns can make the residual sparser than the input.

The catalogue also exposes fixed-point `lfsr`, binary `context` and
autoregressive `world_model` predictors. All four work in the base wheel.
[World-model packets](../reference/spike_codec_wire_formats.md#predictive-xor-containers)
require matching NumPy, BLAS and math-library builds, numerical thread
settings and CPU floating-point behaviour on both endpoints. Use EMA,
LFSR or context when the encoder and decoder profiles differ.

Select `backend="python"` for reference execution, or `backend="rust"` with
a matching engine wheel. The Rust world model uses NumPy for probability
evaluation and a native weight update. See the
[paired benchmarks](../benchmarks/BENCHMARKS.md#predictive-codec-python-and-rust-timing)
for the measured workloads and hardware.

```python
from sc_neurocore.spike_codec import PredictiveSpikeCodec

codec = PredictiveSpikeCodec(
    alpha=0.005,      # EMA smoothing (0.001-0.01 for 20kHz)
    threshold=0.5,    # predict spike if rate > threshold
    backend="auto",
)
data, result = codec.compress(spikes)
print(f"{result.compression_ratio:.1f}x, "
      f"prediction accuracy: {result.prediction_accuracy:.1%}")

# Decoder runs identical predictor — lossless
recovered = codec.decompress(data, 1000, 1024)
```

Architecture:
```
Encoder: actual → predict → XOR → ISI encode → bytes
                    ↑
                    └── update(actual)

Decoder: bytes → ISI decode → XOR → recover → spikes
                               ↑
                    predict ────┘
                      ↑
                      └── update(recovered)
```

Compare the complete packet size and latency on your recordings before
choosing a predictor.

## Codec 3: Delta (Neural Probes)

Groups channels spatially. Within each group, one reference channel is
transmitted raw; others are XOR'd against the reference. Correlated
channels (co-firing neurons on probe arrays) produce sparser deltas.

```python
from sc_neurocore.spike_codec import DeltaSpikeCodec

# Neuropixels: 384 channels, nearby electrodes correlated
codec = DeltaSpikeCodec(group_size=8)
data, result = codec.compress(probe_spikes)
print(f"{result.compression_ratio:.1f}x, "
      f"delta sparsity: {result.mean_delta_sparsity:.1%}, "
      f"{result.n_groups} groups of {result.group_size}")
```

The default backend selects the complete Rust codec if available. Use
`backend="python"` to run the reference implementation. Either backend
can decode the packet using the original time and channel counts.

Use for spatially correlated rasters and compare its complete packet size with
ISI on the same data. Group size must be in 1–256, including after mutation.
[Measured packet sizes and timing](../benchmarks/BENCHMARKS.md#inter-channel-delta-codec-in-python-and-rust).

## Codec 4: Streaming (Real-Time BCI)

Fixed-size time windows, each independently decodable. No inter-frame
dependencies. `window_size / sample_rate` is the acquisition duration of a
frame; encoding, transport and decoding add latency and require measurement.

```python
from sc_neurocore.spike_codec import StreamingSpikeCodec

# 1ms windows at 20kHz = 20 samples per frame
codec = StreamingSpikeCodec(window_size=20)

# Batch compress
data, result = codec.compress(spikes)
print(f"{result.n_frames} frames, max {result.max_frame_bytes} bytes/frame")

# Real-time frame-level API
frame_bytes = codec.compress_frame(window)   # single (20, N) window
recovered_window = codec.decompress_frame(frame_bytes)
```

Use the pull API when a source can supply chunks on demand. The encoder
buffers a single window across chunks. The last frame preserves the exact
remaining rows.

```python
chunks = (spikes[start:start + 13] for start in range(0, len(spikes), 13))
frames = codec.iter_compress(chunks)
for raster_window in codec.iter_decompress(frames):
    print(raster_window.shape[0])
```

Close an iterator to cancel it. Start a new iterator to reset the buffered
state, and resume decoding at a known frame boundary after an error.
Both backends produce the same SSCF and standalone frame bytes.

## Codec 5: AER (Neuromorphic Routing)

Address-Event Representation stores sample deltas and variable-width
neuron addresses. The codec uses AERI silence events for dense inputs.
Use `send_aer_packet` and `receive_aer_packet` from `sc_neurocore.comm`
to translate between raster packets and absolute UDP timestamps.

```python
from sc_neurocore.spike_codec import AERSpikeCodec

codec = AERSpikeCodec(timestamp_bits=16)
data, result = codec.compress(spikes)
print(f"{result.n_events} events, "
      f"{result.bytes_per_event:.1f} bytes/event, "
      f"{result.compression_ratio:.1f}x")
```

This interface encodes binary rasters. Physical-device and camera
interoperability need their own adapters and execution evidence.

## Comparing the Five Raster Codecs

```python
from sc_neurocore.spike_codec import get_codec, list_codecs

for name in list_codecs():
    codec = get_codec(name)
    data, result = codec.compress(spikes)
    print(f"{name:>12}: {result.compression_ratio:>8.1f}x  "
          f"({len(data):>8,} bytes)")
```

Run `examples/codec_benchmark.py` for a full comparison across four
target systems (BCI, probe, neuromorphic, real-time).

## Hardware Mapping

The predictive algorithm uses these operations. This is an illustrative
mapping, not a synthesised whole-codec implementation or a gate-count result:

| Operation | Candidate building block |
|-----------|--------------------------|
| EMA update | Shift-add accumulator |
| Threshold compare | Comparator bank |
| XOR (actual vs predicted) | XOR gate array |
| ISI encoder | Counter and shift register |

Verilog building blocks in `hdl/`: `sc_aer_encoder.v`, `sc_lif_neuron.v`,
`sc_bitstream_encoder.v`.

The clocked `sc_aer_encoder.v` serves each lowest-index pending address once,
with a service-cycle timestamp. Pending spike bits coalesce, and clearing the
served bit takes precedence over a simultaneous spike on that bit. A downstream
queue must accept the encoder's one-event-per-clock cadence.

## Next Steps

- [Spike Codec Library Guide](../guides/bci_codec.md) — architecture details, parameter tuning
- [API Reference](../api/spike_codec.md) — full class documentation
- [AER Communication](50_aer_communication.md) — AER-over-UDP protocol
- [Predictive Coding](26_predictive_coding.md) — SC predictive coding layer (Conjecture C9)
