<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SC-NeuroCore — Codec wire formats and compatibility -->

# Codec wire formats and compatibility

This reference describes the current Python formats and the real Rust
components used by those interfaces. The [catalogue API](../api/spike_codec.md)
is the machine-readable discovery surface. Its descriptions do not select a
backend or certify decoder security, hardware support or performance.

All integer fields below are unsigned and big-endian unless stated otherwise.
T means time rows and N means channels. Raster samples are binary; timestamps
are zero-based sample indices. Sampling rate, channel identifiers and original
file metadata are external. These representations have different framing
contracts and cannot be substituted merely because both contain events.

## Current API and native scope

| Interface | Encode / decode | Required external decode arguments | Rust component scope |
|---|---|---|---|
| ISI | SpikeCodec.compress / decompress | T, N | None |
| Predictive | PredictiveSpikeCodec.compress / decompress | T, N | EMA/LFSR predict-XOR/recovery kernels |
| Delta | DeltaSpikeCodec.compress / decompress | T, N | None |
| Raster frames | StreamingSpikeCodec.compress / decompress | None for whole container | None |
| AER | AERSpikeCodec.compress / decompress | None by default | None |
| Waveform | WaveformCodec.compress / classmethod decompress | None; decoder backend selected separately | Signal kernels, channel coder or stream packet coder; Python WFC2 container |
| Lossless stream | LosslessStreamEncoder.encode / LosslessStreamDecoder.decode | Matching channels, ADC bits, groups, samples per packet and preceding state | Complete packet encoder/decoder |
| Event stream | SpikeEventEncoder.encode / SpikeEventDecoder.decode | Matching channels, bin samples, timing, power bins, groups, samples per packet and preceding state | Complete event/power packet pipeline |

Only waveform and the two raw stream APIs expose backend=auto/python/rust.
Predictive caches all four EMA/LFSR native imports together; there is no
backend keyword. Context and optional world_model use Python. world_model's
module is excluded from the base wheel. Optional zstandard and PyWavelets
can affect lossy waveform encoding; preserve their versions for reproduction.
A zstandard-encoded payload requires zstandard to decode; zlib fallback
payloads do not.

Native waveform selection currently requires all eight waveform kernel
callables, including for lossless modes. The lossless mode also needs the
lossless stream classes. probe_codec checks these actual requirements.
Its result is import availability, not an encoder instance's effective
backend. It does not validate arbitrary numerical configurations. Native
OSError/RuntimeError loader failures appear as diagnostics and exclude auto
from available choices; codec-operation failures are never caught by the
catalogue. Missing imports allow the owning classes' Python fallback.

Static description lookup does not import the native engine or the legacy
raster factory. An embedding application's import policy can make a public
class unavailable as well as deny native loading. The probe reports both
failures; it cannot promise a Python fallback when the class itself cannot
be imported. Explicit Python waveform coding remains available when only
native extension loading is denied.

## ISI: varint and Huffman

Varint has **no magic or entropy flag**:

| Field | Bytes |
|---|---|
| T, N | uint32 each |
| For each channel: event count | Unsigned LEB128 |
| First event time, if nonempty | Unsigned LEB128 |
| Each subsequent inter-spike interval | Unsigned LEB128 |

LEB128 stores seven low bits per byte and sets bit7 when another byte follows.
The public decoder still requires T and N, despite their presence in the
header; it uses the supplied dimensions.

Huffman begins byte01, T:uint32, N:uint32, then count-section length:uint32,
N LEB128 event counts, first-time-section length:uint32, first event times
for nonempty channels as LEB128, and a single Huffman body for all remaining
intervals in channel order. A Huffman body is:

- Entry count:uint16. Zero means an empty body and has no following bit count.
- Each entry: symbol:int32 signed big-endian and code length:uint8.
- Bit count:uint32 and ceil(bit_count/8) bytes, most-significant bit first.

Codes are canonical, ordered by length then symbol. Auto entropy selects
Huffman above 3% spike density; otherwise varint. This policy is per packet,
not a CPU/GPU selector.

Compatibility limit: the current reader treats leading01 as Huffman. A raw
varint T whose high byte is01 therefore collides with that discriminator
(T from 2**24 through 2**25-1). Current encoding refuses every varint raster
with T >= 2**24 after entropy selection, including auto-selected varint.
Explicit Huffman remains available for longer rasters. Unsafe varint packets
written by 4.0.0 cannot reliably be distinguished from Huffman; re-encode
affected data from its original source. There is no checksum or general
corruption-detection guarantee.

Lossy mode with timing_precision>1 ORs each complete block into one bin,
drops a final incomplete block and encodes those bin indices. The packet's
T remains the original input length, and public decode writes bin indices
into the caller's requested shape. It does not expand times back to original
sample positions. This is not a uniform waveform timing-error guarantee.

## Predictive XOR containers

Each header precedes an ordinary ISI residual packet:

| Predictor | Magic | Header after magic |
|---|---|---|
| ema | PSCX | alpha:float64, threshold:float64 |
| lfsr | PSCL | alpha_q8:uint16, seed:uint16 |
| context | PSCC | history length:uint8 |
| world_model | PSCW | history length:uint8, alpha:float64, seed:uint16 |

IEEE floating-point fields are big-endian. Decoder configuration is read from
the predictor header; public T and N are still required. Predictors restart
for every batch compress/decompress call. Seed/history and predictor arithmetic
must match across implementations. Lossless residual coding reconstructs the
input raster. Lossy residual quantisation may also change subsequent predictor
feedback; it does not establish a bounded original timestamp displacement.

## Inter-channel delta

DSCX is followed by group_size:uint16, group_count:uint16, one uint8
reference index per group and the base ISI residual packet. Reference indices
are **local to each group**. The encoder selects the most active channel
(first maximum on ties) and XORs each other channel against it. Decoder T,N
remain external.

Encoding requires group_size in 1..256, both at construction and immediately
before compression after any configuration mutation. Wider groups written
by 4.0.0 could wrap a selected reference index on serialisation; re-encode
affected data from its original source. The uint16 group_size field does not
establish support beyond this limit. No checksum is present.

## Independent raster frames

The whole-raster SSCF container has this 16-byte header:

| Field | Bytes |
|---|---|
| SSCF | 4 |
| window_size | uint16 |
| T | uint32 |
| N | uint16 |
| frame_count | uint32 |

Each frame contains N:uint16, W:uint16, ceil(N/8) silent-channel bitmap
bytes (channel0 in bit0), then ceil(W/8) bytes for each active channel, in
channel order. Each active mask is **little-endian**, sample0 in bit0.
A silent bit of1 omits that channel's mask.

Whole-raster encoding pads the last frame to window_size; decoding trims to
original T. compress_frame/decompress_frame exchange a standalone frame
without SSCF. Frames are independently decodable and have no persistent
entropy state or checksum. window_size counts samples and does not promise
a measured end-to-end latency.

## AER event/silence containers

The 17-byte header is AERX (events) or AERI (silences), T:uint32, N:uint32,
event_count:uint32 and neuron_bytes:uint8. AERI is selected above 50% raster
density. Events are ordered by time, then channel.

Each item contains dt:uint16 and neuron_id encoded in neuron_bytes
big-endian bytes. The all-FF address is reserved as an escape: add that
item's actual dt to the current time and keep reading without consuming an
event. Ordinary items add dt, set their raster position and consume one event.
The encoder widens addresses when needed to keep all-FF separate from valid
IDs. Encoder timestamp_bits affects escape splitting; actual stored deltas
and neuron_bytes suffice for decoding. The default timestamp_bits is 16;
the physical delta field remains uint16.

AERX/AERI are raster codec packets. Compatibility with the separate
comm.aer_udp protocol or any physical neuromorphic device requires an
explicit, exercised adapter. Names alone do not establish interoperability.
No embedded checksum is present.

## WFC2 waveform container and legacy WFCX

WFC2 writes a 20-byte header:

| Field | Bytes |
|---|---|
| WFC2 | 4 |
| T, N | uint32 each |
| template count, snippet count | uint16 each |
| snippet_samples, quantize_bits | uint8 each |
| ISI byte-count bit length | uint8 |
| mode | uint8 |

Mode bytes are full=0, waveform=1, spike=2, lossless_rice=3, lossless=4.
Every WFC2 packet ends with CRC-32:uint32 over all preceding packet bytes.
T*N must not exceed decoder max_samples (default 2**28) unless explicitly
overridden. This bound counts values, not bytes. Checksum failure raises
ValueError; a checksum is not authentication.

Modes0..2 contain three sections, each prefixed by its uint32 byte length:

1. Spike raster: an ISI packet.
2. Snippets: template count:uint16, common scale:float32, quantised int8
   templates, ID count:uint32, LEB128 template IDs, one float32 scale per ID,
   residual scale:float32, padded residual-value count:uint32, and compressed
   nibble-packed residuals (high nibble first, value shifted by 8). Residual padding uses shifted zero.
3. Background: rows:uint32, columns:uint32, delta scale:float32, first row
   as big-endian float32 and compressed row-major int8 delta codes. Background
   is downsampled by 16 in full mode and reconstructed to the original shape.

Unused sections are empty. Snippet/background compressed payloads use zstd
when available, otherwise zlib; readers identify zstd by its frame magic.
The mode determines the decoded result: spike returns the detected raster
and no waveform; waveform adds lossy snippets on zeros; full adds lossy
background reconstruction. Exact detected spike preservation is distinct
from preserving the original continuous waveform.

Lossless modes have one uint32-length section. lossless contains
ADC_bits:uint8 followed by one lossless stream packet, groups=1, fresh state,
N channels and T samples. Container overhead is 29 bytes beyond the raw stream
packet. lossless_rice contains N uint32-length channel sections. Each channel
selects an affine mapping (kind0:uint8, offset:int32, step:uint32) or sorted
value-table mapping (kind1:uint8, count:uint32, zigzag LEB128 table deltas).
It then stores predictor order:uint8, zigzag LEB128 initial finite differences
and block Rice residual data. The implementation in waveform_codec.py defines
adaptive Rice parameters, escaped residuals and level reconstruction; the
whole container remains Python even when Rust encodes channel sections.

WFCX is legacy **read-only, spike mode only**. It has no CRC and lacks
per-snippet scales/background first-row information required for reliable
waveform reconstruction. Current writers always emit WFC2; other WFCX modes
are refused. Neither format preserves an original WAV header. Preserve file
metadata separately when byte-for-byte WAV restoration is required.

## Raw stateful lossless and event packets

Both formats concatenate group bodies as uint32 byte length followed by
range-coded bytes. They contain no format magic, version, dimensions, packet
sample count, reset marker or checksum. Keep a versioned application envelope
with the exact stream configuration, packet boundaries and sample counts;
concatenating packet bodies alone loses necessary metadata.

Lossless decoding needs channels, adc_bits (1..16) and groups plus samples
for each packet. Groups are contiguous channel ranges. Adaptive level
numbering, predictors and entropy state continue across windows. A short
final window is encoded with its real length; no padding sample is invented.

Event encoding additionally applies integer filters, warmup/noise adaptation,
threshold crossing and refractory state. The decoder needs channels,
bin_samples, timing (bin or sample), power_bins and groups. Use the encoder's
resolved settings. Results carry absolute first_bin/first_period indices,
uint8 event matrices, optional int16 in-bin offsets (-1 for absent events)
and int32 power matrices. Original waveform reconstruction is unavailable.
An unfinished bin/power period continues into the next packet; there is no
public flush operation for the last partial bin.

Create a new encoder/decoder pair to reset either stream. Preserve order and
all preceding packets. Framing errors leave decoder state unchanged; invalid
content encountered during decoding permanently breaks it. Later decode
calls then refuse; begin a new stream. Raw packets cannot be assumed to
detect every content corruption. Security and recovery require the owning
application's explicit contract rather than an inferred container guarantee.

## Compatibility and implementation languages

Python is the reference interface. Rust is the native implementation language
for the components named above. Generated Go, Julia and Mojo codec files are
not supported selectable implementations. No standalone whole-container Rust
support is implied by a kernel or packet-coder capability.

The lossless stream also has an optional native Vulkan WGSL encoder, with
CPU decode. It has no Python GPU selector, GPU decoder or ROCm codec path.
Physical-device qualification is specific to the tested geometry and driver;
a device or source file existing is not proof of another configuration.

Preserve old packets and their external metadata when changing consumers.
The catalogue does not rewrite any format, add an envelope or silently migrate
data. Cross-decode through the actual public interfaces before deploying a
new native/backend/package combination. Performance comparisons must include
the same data, framing, configuration, software versions and identified HW.
