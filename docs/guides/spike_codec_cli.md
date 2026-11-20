<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SC-NeuroCore — Recording codec commands -->

# Recording codec commands

The installed `sc-neurocore codec` command works with the same catalogue and
packets as the Python API. It reports JSON on standard output and sends
processing errors as JSON to standard error with a nonzero status. Use
`--help` on any action to list its arguments.

```bash
sc-neurocore codec discover
sc-neurocore codec inspect waveform --variant lossless
```

The base wheel supports the Python reference. With the matching engine wheel
installed, `auto` selects the available native implementation. Pass
`--backend python` or `--backend rust` to select explicitly. A missing Rust
implementation is an error on an explicit request. Encoding and decoding
report the effective backend and native scope. The lossless stream can also
select `gpu` when an engine supplies that encoder and the hardware accepts
the configuration. For GPU encoding, set `groups` to the channel count.
Use a CPU backend for decoding its packets.

## Encode, verify and decode

A numeric NPY input has time rows and channel columns. A one-dimensional
array is a single channel. Raster codecs take binary values. For example,
encode an existing raster and decode it back to NPY:

```bash
sc-neurocore codec encode raster.npy raster.sc-codec --codec isi --variant lossless
sc-neurocore codec inspect --file raster.sc-codec
sc-neurocore codec verify raster.sc-codec
sc-neurocore codec decode raster.sc-codec restored.npy
```

`inspect --file` reads the header. `verify` reads the entire file, reconstructs
each packet and checks the checksum and end of input. `decode` performs the
same checks before publishing the output.

For an event CSV, write the header `sample,channel`, followed by sorted
zero-based integer pairs. Every row represents a spike. Supply the total
recording geometry, including silent channels and trailing time rows:

```bash
sc-neurocore codec encode events.csv events.sc-codec --codec aer --rows 10000 --channels 32
sc-neurocore codec decode events.sc-codec raster.npy --field raster
```

Legacy BCI `SpikeCodec.encode` bytes can be migrated by selecting
`--legacy-rle` and supplying the time-by-channel shape. The old format stores
only the flattened length. Use one row for a single BCI channel mask.
The importer checks that every run matches the supplied shape and writes
a framed recording without changing the source.

```bash
sc-neurocore codec encode telemetry.bin recording.sc-codec --codec isi --legacy-rle --rows 1 --channels 1024
sc-neurocore codec decode recording.sc-codec restored.npy --field raster
```

Python applications can use `sc_neurocore.cli.codec.recording` to encode
and verify the same files. The recording functions support per-window
progress and read-only preview callbacks. A preview becomes verified
when the complete decode operation succeeds.

Waveform codecs also accept uncompressed signed 16-bit PCM WAV. Use
`--sample-rate` for NPY input to the spike-event codec. WAV supplies its
own sample rate. Encode a continuous lossless recording or extract events
and band power:

```bash
sc-neurocore codec encode signal.wav lossless.sc-codec --codec lossless_stream
sc-neurocore codec decode lossless.sc-codec samples.npy --field waveform
sc-neurocore codec encode signal.wav events.sc-codec --codec spike_events --variant sample
sc-neurocore codec decode events.sc-codec detections.npz
sc-neurocore codec decode events.sc-codec power.npy --field power
```

| Result | Fields |
| --- | --- |
| Raster codecs | `raster` |
| WFC2 full or waveform mode | `waveform`, `spikes` |
| WFC2 spike mode | `spikes` |
| WFC2 lossless modes or the lossless stream | `waveform` |
| Spike events | `events`, `power`, and `offsets` in sample timing |

A single field can go to NPY. NPZ stores the selected fields as named
arrays. An event row represents one complete bin. Power rows represent
complete power periods. Offsets are sample indices within each bin, with
`-1` for channels without an event. Incomplete bins and power periods at
the end of a file do not emit a row.

Use `--force` to replace an existing destination after success. The
command rejects using the same file as both input and output. Decoded
fields and packet streams are written to temporary files beside the
destination. A failed encode, decode or integrity check leaves a previously
published destination intact.

## Bound memory and select codec options

The input reader maps NPY and streams WAV and CSV. A codec frame holds at
most 1,048,576 values and 65,535 time rows. The default frame uses up to
4,096 rows, reduced for wide recordings. Use `--frame-rows` to select a
different compatible size. Stateful lossless waveform and event encoders
carry their state across file frames. The other families start a new packet
per frame. For detection, prediction and lossy binning, this means frame
boundaries are part of the result. Pick the same frame size when comparing
two files or backends. Each lossy raster frame retains the original output
shape, with bin events indexed at the start of the frame.

The command accepts repeated `--option NAME=VALUE` arguments. Options
not given use the public API defaults.

| Family | File options |
| --- | --- |
| ISI | `entropy`, `timing_precision` |
| Delta | `group_size`, `timing_precision` |
| Predictive | `timing_precision` (values above one select lossy binning) |
| Streaming raster | `window_size` |
| AER | None |
| WFC2 | `adc_bits` in lossless modes |
| Lossless waveform stream | `adc_bits`, `groups` |
| Spike events | `bin_samples`, `power_bins`, `groups` |

## Measure both backends

```bash
sc-neurocore codec benchmark signal.wav --codec waveform --variant lossless --repetitions 5
```

The default comparison requires both installed CPU implementations. It
alternates their order and checks that packet and decoded array hashes match.
Use `--backend python` to measure a base-only install. Each JSON run records
the effective encoder and decoder, input geometry, payload bytes and separate
encoding and decoding times. Those timings exclude file I/O and hashing.

The [recording file format](../reference/spike_codec_wire_formats.md#recording-files)
describes the framing around the existing [public codec API](../api/spike_codec.md).
