<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SC-NeuroCore — Spike codec library -->

# Spike Codec Library

The catalogue covers eight families and 18 variants for binary spike rasters,
numeric electrode waveforms, stateful lossless streams and detected events
with band power. Each family defines its own decoder contract.

The [recording codec commands](../guides/spike_codec_cli.md) use these
interfaces for file discovery, encoding, decoding, verification and timing.

[Wire formats and compatibility](../reference/spike_codec_wire_formats.md)
documents bytes, state and external metadata. The catalogue describes existing
implementations; import availability does not certify performance, complete
native containers or integration with a physical device.

## Discover and inspect

```python
from dataclasses import asdict
import json
from sc_neurocore.spike_codec import (
    describe_codec, list_codec_descriptors, probe_codec,
)

for item in list_codec_descriptors():
    print(item.name, item.category, item.input_kind)
print(json.dumps(asdict(describe_codec("waveform")), indent=2))
print(probe_codec("waveform", "lossless"))
```

Descriptions are immutable and static: listing/lookup imports neither the
optional native engine nor the legacy raster factory. The explicit probe
imports public classes, selected optional
modules and actual engine callables. It reports Python availability, native
component scope, backend selector values and diagnostics. It does not
construct a codec or change an existing instance. Restart after installing
an engine: the predictive module caches native imports. Unknown names or
variants raise `ValueError`. Generated language source files do not establish
an available codec backend.

Host import policies can make an entry point unavailable. An `OSError`,
`RuntimeError` or import refusal from a present engine appears in diagnostics
and excludes `auto` from the advertised choices for every family. An absent
engine permits Python fallback. The probe reports an unavailable Python class
separately. The [installed codec comparison](../benchmarks/BENCHMARKS.md#installed-codec-catalogue)
checks actual wheel imports and packet compatibility.

| Name | Input / decoded output | Choices | Persistent state | Backend selector |
|---|---|---|---|---|
| isi | Binary raster / raster | mode: lossless, lossy; entropy: auto, varint, huffman | None | auto, python, rust when available |
| predictive | Binary raster / raster | ema, lfsr, context, world_model | Reset per batch | auto, python, rust when available |
| delta | Binary raster / raster | lossless, lossy | None | auto, python, rust when available |
| streaming | Binary raster / raster | Independent frames | None | auto, python, rust when available |
| aer | Binary raster / raster | Adaptive event/silence coding | None | auto, python, rust when available |
| waveform | Numeric waveform / WaveformDecodeResult | full, waveform, spike, lossless, lossless_rice | Reset per recording | auto, python, rust when available |
| lossless_stream | Integer waveform / exact int16 waveform | Lossless | Yes | auto, python, rust when available, optional GPU encoder |
| spike_events | Integer waveform / events and power | timing: bin, sample | Yes | auto, python, rust when available |

Arrays use time rows and channel columns. Raster values are binary.
Lossless waveform modes need integer-valued samples in the int16 range;
lossy waveform modes accept finite numeric samples. Timestamps and window
counts are sample indices. The event encoder also takes sample_rate_hz
to configure filters and its default bin duration. Preserve sampling rate,
channel identity and acquisition metadata outside these packets.

::: sc_neurocore.spike_codec.catalogue
    options:
      show_root_heading: true
      members:
        - list_codec_descriptors
        - describe_codec
        - probe_codec

::: sc_neurocore.spike_codec.catalogue_types
    options:
      show_root_heading: true
      members:
        - CodecDescriptor
        - CodecVariant
        - CodecWireFormat
        - CodecAvailability

## Legacy raster factory

get_codec and list_codecs retain five names: isi, predictive, delta,
streaming and aer. They use compress(raster) and decompress(packet, T, N).
AER and whole raster-stream packets can infer dimensions from their headers.
Waveforms and stateful streams use their separate entry points.
recommend_codec is a deterministic raster configuration heuristic, not a
measured performance ranking or latency guarantee; see its API priority rules.


::: sc_neurocore.spike_codec.registry
    options:
      show_root_heading: true
      members:
        - get_codec
        - list_codecs
        - recommend_codec

## ISI Codec (Baseline)

Inter-spike interval encoding with LEB128 or canonical Huffman coding.
Python and Rust produce the same packets for the same raster and settings.
Use the instance's `backend` property to inspect the selected implementation.
Entropy defaults to auto: Huffman above 3% spike density and varint
otherwise. Ratios depend on input and framing; measure your data.
Encoding refuses varint rasters with 16,777,216 or more time steps, including
when auto selects varint. Explicit Huffman supports longer rasters. Re-encode
4.0.0 varint data with 16,777,216 through 33,554,431 time steps from its
original source. That range collides with the Huffman marker. Both readers
accept unambiguous legacy varint packets with at least 33,554,432 time steps.
See the [Python and Rust timings](../benchmarks/BENCHMARKS.md#isi-codec-python-and-rust-timing)
for the synthetic and recorded workloads.

::: sc_neurocore.spike_codec.codec
    options:
      show_root_heading: true
      members:
        - SpikeCodec
        - CompressionResult

## Predictive Codec

EMA, LFSR, context and world-model predictors encode XOR residuals.
Predictor state resets per batch. Lossless reconstruction requires
`base_mode="lossless"`. With timing precision greater than one, lossy mode
merges each complete bin before prediction. Both backends decode the same
bin raster. Predictor state resets on that grid for each packet.

Select `backend="python"` to use the reference path. Explicit Rust requires
the selected predictor's native components and Rust ISI. The base wheel
contains the world-model codec predictor using its standard NumPy dependency.
The Rust path still uses NumPy for probability evaluation and uses Rust
for the weight update. The experimental world-model package is separate.
`predictor_backend` reports the selected encoding stage. `base_codec.backend`
reports the residual ISI coder. `backend` reports the common selection or
`mixed` when those two stages differ.
World-model packets require a matching numerical profile for recovery. See the
[wire-format conditions](../reference/spike_codec_wire_formats.md#predictive-xor-containers).
See the [predictive codec benchmarks](../benchmarks/BENCHMARKS.md#predictive-codec-python-and-rust-timing)
for paired Python and Rust encode, decode and round-trip measurements.

::: sc_neurocore.spike_codec.predictive_codec
    options:
      show_root_heading: true
      members:
        - PredictiveSpikeCodec
        - PredictiveCompressionResult

## Delta Codec (Neural Probes)

Inter-channel XOR residuals. Groups channels spatially, picks reference
per group, encodes others as delta. Reference indices are group-local uint8
values. Construction and compression require group_size in 1–256, including
after configuration mutation. Wider groups written by 4.0.0 could wrap the
index; re-encode affected data from its original source. Decoding validates
the group count, final group width and local references before reading the
embedded ISI packet.

Select `backend="python"` for the reference implementation or
`backend="rust"` for native group selection, XOR and packet coding.
The default `"auto"` uses the complete native path when the installed
engine exports it. Both backends read and write the same DSCX format.

::: sc_neurocore.spike_codec.delta_codec
    options:
      show_root_heading: true
      members:
        - DeltaSpikeCodec
        - DeltaCompressionResult

## Streaming Codec (Real-Time)

The Python and Rust backends encode the same independently decodable frames.
Use `backend="auto"` to select the installed Rust codec with Python fallback,
or select a backend explicitly. The default window spans 20 samples.

Use `compress` and `decompress` for an SSCF container. The last frame is
padded, and the decoder returns the original row count or a requested row
prefix. Use `iter_compress` and `iter_decompress` for a pull stream of
standalone frames with an unpadded final frame. The consumer controls
backpressure by advancing the iterator. Close it to cancel, then create a
new iterator at a known frame boundary to restart.
[Measured first-frame latency and sustained throughput](../benchmarks/BENCHMARKS.md#framed-spike-rasters-in-python-and-rust).

::: sc_neurocore.spike_codec.streaming_codec
    options:
      show_root_heading: true
      members:
        - StreamingSpikeCodec
        - StreamingCompressionResult

## AER Codec (Neuromorphic)

Address-Event Representation stores ordered spike positions with
delta-coded sample indices. Above 50% density it stores silence
positions. Python and Rust encode the same bytes and cross-decode
AERX and AERI. Use `backend="auto"` for native selection with Python
fallback, or select either backend explicitly.

Use the UDP adapter to translate those containers into absolute-timestamp
events and reconstruct the raster on the receiver. Use the announcement
callback to pass the shape, clock and sequence window to the receiver
through the application's control channel.
[Wire formats and UDP translation](../reference/spike_codec_wire_formats.md#aer-eventsilence-containers).

::: sc_neurocore.spike_codec.aer_codec
    options:
      show_root_heading: true
      members:
        - AERSpikeCodec
        - AERCompressionResult

## Waveform Codec (Raw Electrode)

End-to-end raw waveform compression: spike detection + template matching +
background LFP compression, plus exact lossless modes. Five modes:
`spike` (spike timing only), `waveform` (timing + snippet templates),
`full` (timing + snippets + background, lossy), `lossless` (every int16
sample exactly, with the streaming lossless coder below) and `lossless_rice`
(every int16 sample exactly, with the earlier block Rice coder). `WaveformCodec.decompress(data)` decodes every mode and returns
a `WaveformDecodeResult`. Streams use format `WFC2` with a CRC-32 trailer;
a corrupted stream raises `ValueError`, and `max_samples` (default `2**28`)
bounds what a header may declare. Legacy `WFCX` streams decode their spike
raster only. The heavy kernels run in Rust when `sc_neurocore_engine`
provides them (`backend="auto"`, `"rust"` or `"python"`); both backends write
byte-identical streams. Measured ratios on the public Neuralink compression-challenge
recordings are in the [BCI codec guide](../guides/bci_codec.md). The codec validates inputs
before telemetry sections are built: raw samples must be a finite, non-empty
`(time, channel)` matrix, `snippet_samples` must fit the one-byte header
(1-255), `max_templates` must fit the two-byte template count (1-65535),
`template_threshold` must be in `[0, 1]`, and `quantize_bits` must be in
`[1, 8]`.

::: sc_neurocore.spike_codec.waveform_codec
    options:
      show_root_heading: true
      members:
        - WaveformCodec
        - WaveformCompressionResult
        - WaveformDecodeResult

## Streaming Lossless Coder (Live Electrode Streams)

Stateful lossless coding of a multi-channel int16 stream, window by window:
ADC levels with dense numbering (missing codes leave it), a sign-sign LMS
predictor and an adaptive binary range coder. State persists across windows;
each packet carries a four-byte length per group plus range-coder bytes.
Supply matching channels, ADC width and groups to the decoder, preserve packet
order and pass the exact sample count per packet. Create a new pair to reset.
Every valid packet decodes to exactly the
encoded samples. Rust (`backend="auto"`/`"rust"`) and the pure-Python reference
(`"python"`) write identical packets. Details and measurements are in the
[BCI codec guide](../guides/bci_codec.md).

::: sc_neurocore.spike_codec.waveform_stream
    options:
      show_root_heading: true
      members:
        - LosslessStreamEncoder
        - LosslessStreamDecoder
        - detect_adc_bits
        - level_of
        - nominal_value

## Spike-Event Layer (Bluetooth-Class Links)

Import `SpikeEventEncoder`, `SpikeEventDecoder` and `SpikeEventWindow` from
`sc_neurocore.spike_codec`. A pair uses the same channels, bin size, timing
mode, band-power interval and groups. Each packet carries completed bins and
periods. Unfinished intervals continue across calls as long as both ends
receive the same window sizes in order. Call `reset()` on both ends at the
start of an independent recording. The `backend` attribute reports the
effective implementation.


Threshold-crossing events in 1 ms bins and spike-band power, computed from raw
int16 samples with integer filters and a streaming noise estimate, coded with
the range coder of the lossless coder. Rust and the pure-Python reference write
identical packets. Validation and bit rates are in the
[BCI codec guide](../guides/bci_codec.md).

::: sc_neurocore.spike_codec.spike_events
    options:
      show_root_heading: true
      members:
        - SpikeEventEncoder
        - SpikeEventDecoder
        - SpikeEventWindow
        - highpass_coefficients
        - lowpass_coefficients
        - threshold_factor

## Complete waveform containers

The catalogue reports native scope `container` for all five waveform
modes. `WaveformCodec(backend="rust")` writes complete WFC2 packets and
`WaveformCodec.decompress(packet, backend="rust")` reads them. Select
`backend="python"` for the reference. The decoder checks the total number
of values against `max_samples` before allocating its output.
`WaveformCodec.backend` retains the requested setting. Inspect
`effective_backend` and `native_scope` to distinguish a full container, an
older engine providing signal kernels, and a Python-only instance.
[Wire formats](../reference/spike_codec_wire_formats.md#wfc2-waveform-container-and-legacy-wfcx)
describe WFC2 and the legacy spike-only WFCX reader.

## Stateful lossless example

```python
import numpy as np
from sc_neurocore.spike_codec import LosslessStreamEncoder, LosslessStreamDecoder

x = (np.arange(257 * 2).reshape(257, 2) % 47).astype(np.int16)
encoder = LosslessStreamEncoder(2, backend="python")
decoder = LosslessStreamDecoder(2, backend="python")
restored = []
for first in range(0, len(x), 20):
    window = x[first:first + 20]
    packet = encoder.encode(window)
    restored.append(decoder.decode(packet, samples=len(window)))
np.testing.assert_array_equal(np.concatenate(restored), x)
```

Use `backend="rust"` for CPU encoding and decoding with the installed engine.
The `backend` property reports the backend selected by `"auto"`. A
GPU-feature engine also supports `LosslessStreamEncoder(..., groups=channels,
backend="gpu", gpu_max_samples=window_rows)`. Read `encoder.device` for the
adapter name and graphics API. Pass the same channel count, ADC width and
group count to a CPU decoder. The catalogue reports an importable GPU
encoder in `probe_codec("lossless_stream").encoder_only_backends`. Device
initialization happens when the encoder is created.

Raw lossless and event packets carry their group lengths and range-coded
bodies. Supply the stream configuration, packet boundaries and row counts
out of band. Bad framing leaves decoder state unchanged. Invalid coded
content stops a lossless waveform decoder until `reset()`. Reset both
waveform endpoints together to start an independent stream. Event
streams preserve coded events/power, not the original waveform. Incomplete
event bins/power periods continue in the next packet; no flush API exists
for the final partial bin. See the wire guide before persisting raw packets.
