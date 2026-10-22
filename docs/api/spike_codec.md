# Spike Codec Library

Six codecs for neural data compression — BCI telemetry, neural probes,
neuromorphic routing, real-time streaming, and general-purpose archival.

All codecs share `compress(spikes) → (bytes, result)` and
`decompress(bytes, T, N) → spikes`.

## Registry

::: sc_neurocore.spike_codec.registry
    options:
      show_root_heading: true
      members:
        - get_codec
        - list_codecs
        - recommend_codec

## ISI Codec (Baseline)

Inter-spike interval encoding with LEB128 variable-length integers.
50-200x compression on typical cortical firing rates.

::: sc_neurocore.spike_codec.codec
    options:
      show_root_heading: true
      members:
        - SpikeCodec
        - CompressionResult

## Predictive Codec (BCI Implants)

EMA predictor + XOR error coding. Only transmit surprises.
Encoder and decoder share identical deterministic predictor state.

::: sc_neurocore.spike_codec.predictive_codec
    options:
      show_root_heading: true
      members:
        - PredictiveSpikeCodec
        - PredictiveCompressionResult

## Delta Codec (Neural Probes)

Inter-channel XOR residuals. Groups channels spatially, picks reference
per group, encodes others as delta. Best for correlated probe arrays.

::: sc_neurocore.spike_codec.delta_codec
    options:
      show_root_heading: true
      members:
        - DeltaSpikeCodec
        - DeltaCompressionResult

## Streaming Codec (Real-Time)

Fixed-latency, independently decodable frames. Each time window is
a self-contained frame with bounded worst-case latency.

::: sc_neurocore.spike_codec.streaming_codec
    options:
      show_root_heading: true
      members:
        - StreamingSpikeCodec
        - StreamingCompressionResult

## AER Codec (Neuromorphic)

Address-Event Representation: compact event stream with delta-coded
timestamps. Compatible with `comm.aer_udp` protocol. O(n_spikes) bytes.

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
each packet adds about 9 bytes per channel group, so 1 ms packets cost little
only when many channels share a group. Every packet decodes to exactly the
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
