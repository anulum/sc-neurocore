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
background LFP compression. 24x on 1024-channel Neuralink-scale data.
Spike timing is lossless. Fits in Bluetooth uplink. The codec validates inputs
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
