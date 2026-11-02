<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SC-NeuroCore — Spike codec library -->

# Spike Codec Library

Six existing codec families cover binary spike rasters and numeric electrode
waveforms. Two additional APIs handle stateful lossless waveforms and detected
events/band power. These interfaces have different decoder contracts.

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

Host import policies can make an entry point unavailable. A native loader's
`OSError` or `RuntimeError` appears in diagnostics and excludes `auto` from
the waveform/stream backend choices; an absent engine permits documented
Python fallback. The probe reports an unavailable Python class separately.

| Name | Input / decoded output | Choices | Persistent state | Backend selector |
|---|---|---|---|---|
| isi | Binary raster / raster | mode: lossless, lossy; entropy: auto, varint, huffman | None | None |
| predictive | Binary raster / raster | ema, lfsr, context, world_model | Reset per batch | None; EMA/LFSR may use Rust kernels |
| delta | Binary raster / raster | lossless, lossy | None | None |
| streaming | Binary raster / raster | Independent frames | None | None |
| aer | Binary raster / raster | Adaptive event/silence coding | None | None |
| waveform | Numeric waveform / WaveformDecodeResult | full, waveform, spike, lossless, lossless_rice | Reset per recording | auto, python, rust when available |
| lossless_stream | Integer waveform / exact int16 waveform | Lossless | Yes | auto, python, rust when available |
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

Inter-spike interval encoding with LEB128 variable-length integers.
Entropy defaults to auto: Huffman above 3% spike density and varint
otherwise. Ratios depend on input and framing; measure your data.
Encoding refuses varint rasters with 16,777,216 or more time steps, including
when auto selects varint. Explicit Huffman supports longer rasters. Re-encode
affected 4.0.0 varint data from its original source because legacy packets
can be mistaken for Huffman.

::: sc_neurocore.spike_codec.codec
    options:
      show_root_heading: true
      members:
        - SpikeCodec
        - CompressionResult

## Predictive Codec (BCI Implants)

EMA, LFSR, context and optional world-model predictors encode XOR residuals.
Predictor state resets per batch. Lossless reconstruction requires
base_mode="lossless". Rust EMA/LFSR functions accelerate predictor loops;
the container and ISI entropy path remain Python. This class has no backend
keyword. The world-model module is excluded from the base wheel; probe it
per variant.

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
index; re-encode affected data from its original source.

::: sc_neurocore.spike_codec.delta_codec
    options:
      show_root_heading: true
      members:
        - DeltaSpikeCodec
        - DeltaCompressionResult

## Streaming Codec (Real-Time)

Independently decodable frames. window_size is a sample count (default 20),
not measured processing or transport latency. The whole-raster container
pads its last frame and trims to the original decoded length.

::: sc_neurocore.spike_codec.streaming_codec
    options:
      show_root_heading: true
      members:
        - StreamingSpikeCodec
        - StreamingCompressionResult

## AER Codec (Neuromorphic)

Address-Event Representation: compact event stream with delta-coded
timestamps. Size depends on address width and timestamp escapes. Above 50%
density the encoder stores silence events. This catalogue does not establish
compatibility with comm.aer_udp or physical hardware.

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

Rust accelerates actual predictors, waveform kernels/channel coders and the
two packet pipelines; native_scope distinguishes these. It does not imply a
standalone Rust implementation of every enclosing Python format. An optional
native Vulkan encoder exists for lossless waveform streaming with CPU decode;
there is no Python backend="gpu" or ROCm selector. Python is the reference
language and Rust the native implementation language. Go, Julia and Mojo
codec mirrors are not selectable supported backends.

Raw lossless/event packets have no checksum, dimensions or reset marker.
Bad framing leaves decoder state unchanged, but invalid content encountered
during decoding breaks the stream; start a fresh pair afterward. Event
streams preserve coded events/power, not the original waveform. Incomplete
event bins/power periods continue in the next packet; no flush API exists
for the final partial bin. See the wire guide before persisting raw packets.
