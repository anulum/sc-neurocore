# Rust Engine API (sc_neurocore_engine)

The high-performance Rust engine provides SIMD-accelerated stochastic computing
with SIMD-accelerated bitstream operations, IR compilation, and HDC support.

**[Browse the full Rust API documentation →](https://anulum.github.io/sc-neurocore/rust-api/sc_neurocore_engine/)**

## Key Modules

| Module | Description |
|--------|-------------|
| `bitstream` | Packed bitstream types and SIMD operations (AND, popcount, rotate) |
| `encoder` | LFSR-based stochastic encoders with decorrelated seeds |
| `neuron` | Fixed-point LIF neuron with Q8.8 arithmetic |
| `layer` | Dense layer pipeline with vectorised forward pass |
| `ir` | Intermediate representation for graph compilation |
| `graph` | Computational graph builder and verifier |
| `attention` | Stochastic attention mechanism |
| `grad` | Surrogate gradient training support |
| `scpn` | SCPN layer primitives (Petri net places/transitions) |
| `simd` | Platform-adaptive SIMD kernels (AVX2, SSE4.1, NEON, portable) |
| `analysis` | **22 spike train analysis modules** (see [Rust Analysis Engine](rust-analysis-engine.md)) |
| `neurons` | **100+ neuron models** — biophysical, maps, hardware, interneurons |

**[Criterion Benchmarks →](rust-benchmarks.md)** — all measured latencies for engine, neurons, and analysis.

## Whole waveform containers

`codecs::waveform_container` encodes and decodes complete WFC2 packets
independently of Python. Pass a `WaveformConfig` and contiguous row-major
samples to `encode_lossy` for `spike`, `waveform` or `full`, or to
`encode_lossless` for `lossless` or `lossless_rice`. Call `decode` with a
limit on the total number of sample values. Its result reports the mode,
shape and reconstructed raster and waveform.

The default Cargo features include `waveform-zstd`. The default writer uses
zlib and the Rust configuration can select Zstandard. Select
`denoise_background` for symmetric db4 denoising in `full` mode.

The standalone example reads interleaved signed little-endian int16 input.

```bash
cargo run --release --example waveform_container -- encode lossless 9760 64 eeg.pcm eeg.wfc2
cargo run --release --example waveform_container -- decode eeg.wfc2 restored.pcm
```

The decoder prints mode, time rows, channels and sample type. For `spike`
it writes one byte per raster cell. `waveform` and `full` write little-endian
float32 samples, and both lossless modes write little-endian int16 samples.
Python reads the same packets through `WaveformCodec.decompress`.

The compiled module exports the same operations as `waveform_container_encode`
and `waveform_container_decode`; `WAVEFORM_CONTAINER_ZSTD` reports whether the
wheel was built with `waveform-zstd`. The encoder takes a C-contiguous int16
matrix for the lossless modes or a float32 matrix otherwise, with the
configuration options. It returns the packet, the spike and template counts
and the byte sizes of the spike, snippet and background sections. The decoder
takes the packet and the sample limit. It returns the mode, the int8 raster and
the samples, each of the last two when the mode carries it. Both release the
GIL while coding.

## Lossless waveform streaming

Use `codecs::waveform_stream::{StreamEncoder, StreamDecoder}` on row-major
`i16` samples. Both endpoints use the same channel count, ADC width and
contiguous group count. Pass the number of rows separately for each packet.

```rust
use sc_neurocore_engine::codecs::waveform_stream::{StreamDecoder, StreamEncoder};

fn round_trip() -> Result<(), String> {
    let mut encoder = StreamEncoder::new(2, 16, 1)?;
    let mut decoder = StreamDecoder::new(2, 16, 1)?;
    let window = [40, -2, 42, -1, 43, -3];
    let packet = encoder.encode(&window, 3)?;
    assert_eq!(decoder.decode(&packet, 3)?, window);
    encoder.reset();
    decoder.reset();
    Ok(())
}
```

With the `gpu` feature, `codecs::waveform_stream_gpu::GpuStreamEncoder`
encodes one group per channel on a selected physical adapter. Read
`adapter_name` and `device_api()` after constructing it. Pass the packets
to a CPU `StreamDecoder` with `groups == channels`. The matching PyO3
classes in `sc_neurocore_engine.sc_neurocore_engine` are
`WaveformLosslessEncoder`, `WaveformLosslessDecoder` and the optional
`WaveformLosslessGpuEncoder`. Each `reset()` starts an independent stream; the
decoder's also clears an earlier decoding error. The [BCI guide](../guides/bci_codec.md#streaming-lossless-coder)
documents Python backend selection and device qualification.

## Fallible dense storage

`layer::DenseLayer::try_new` and the `try_get_weights`, `try_set_weights`,
`try_refresh_packed_weights`, `try_forward`, `try_forward_fast`,
`try_forward_fused`, `try_forward_batch`, `try_forward_batch_into`,
`try_forward_prepacked` and `try_forward_prepacked_2d` methods return
`layer::DenseLayerError`. Its `Invalid` variant retains authored shape refusals;
`Allocation` identifies storage that cannot be represented or reserved.
Weight replacements and caller-owned batch output commit only after complete
success. The existing Rust convenience signatures remain available; constructor,
weight-copy and refresh convenience methods retain their documented panic
behavior. Python bindings use the fallible paths and Python-owned outputs.

## Fallible GPU dense storage

GPU-feature wheels re-export `GpuDenseLayer`, `GpuLifBatch`, `GpuKuramoto` and
`GpuIzhikevichBatch` from `sc_neurocore_engine`; these are the native class
objects registered in `sc_neurocore_engine.sc_neurocore_engine`. Default wheels
omit the GPU exports.

With the `gpu` feature, `gpu::GpuDenseLayer::try_new_checked` returns a typed
`gpu::dense::GpuDenseError` or `Ok(None)` for unavailable GPU hardware.
`try_forward_gpu` and `try_forward_batch_gpu` validate inputs and reserve host
storage before submitting the original Philox shader kernels. Construction
checks shader index widths, device buffer limits and dispatch limits. A zero
sample batch returns an empty result. Calls sharing persistent GPU buffers
serialize submission and readback; rejected shapes leave packed weights intact.
The original Rust convenience signatures remain available with documented panic
behavior. Python uses the checked methods and separately owned list outputs.

GPU feature qualification requires the dedicated Rust integration surface and
an installed GPU-feature wheel for the Python surface:

```bash
cargo test --locked --features gpu --test test_gpu_dense -- --test-threads=1
python -m pytest engine/tests/python/test_gpu_dense_layer_engine_binding.py
```

These GPU tests require real hardware; a default wheel does not provide this
optional binding. The shader's Philox inputs remain distinct from the CPU
encoder's random streams, so exact same-seed CPU outputs are not its oracle.

## Panics and integer overflow in release builds

The wheels are built with the workspace release profile (`Cargo.toml`,
`[profile.release]`), which sets `panic = "unwind"` and no `overflow-checks`.
Two consequences follow, and they are the engine's contract:

- **A panic does not end the Python process.** A panic that reaches a binding
  is raised in Python as PyO3's `PanicException`. Before this release the
  profile set `panic = "abort"`, and a panic ended the interpreter. The Python
  bindings refuse the documented inputs that panic before calling the Rust code,
  so these raise `ValueError` or `MemoryError`. Examples are
  `RustHybridLinearAttentionNeuron(dim=0)` and an allocation failure in
  `DifferentiableDenseLayer.get_weights`. `PanicException` derives from
  `BaseException`, so `except Exception` does not catch it.
- **Integer overflow wraps.** Release builds compute integer arithmetic modulo
  the type's width, where a debug build panics. The integer hardware models
  accumulate in `i32`, for example TrueNorth's `v += weighted_input - leak`, so
  an input large enough to leave the range wraps silently. The Rust panics
  documented under `# Panics` stay as they are for Rust callers. Examples are a
  time constant of 0 in the Loihi steps, which the Python bindings cannot set.

## Building from Source

```bash
cd engine
cargo build --release
cargo test
cargo doc --open
```

## Python Bindings (PyO3)

The engine is exposed to Python via the local `sc_neurocore_engine` bridge.
For source checkouts:

```bash
cd bridge
maturin develop --release
```

```python
import sc_neurocore_engine as engine

# Compile an IR graph
graph = engine.IRGraph()
graph.add_encode(0, 1024, 0xACE1)
graph.verify()
sv_code = graph.emit_sv()
```

## Predictive codec components

The Python bridge publishes the native context encoder and decoder and the
world-model weight update. The
[predictive codec API](spike_codec.md#predictive-codec) combines them with
the existing EMA, LFSR and ISI bindings. The world-model path evaluates
probabilities with NumPy and uses the same numerical profile for encoding
and decoding.

## Inter-channel delta packets

The compiled module exports `delta_encode_raster` and `delta_decode_raster`.
The encoder accepts a contiguous binary int8 raster, group size, timing
precision and ISI entropy selection. It returns a complete DSCX packet
with the original and non-reference residual spike counts. The decoder
validates group geometry before decoding the ISI body. The public
[delta codec API](spike_codec.md#delta-codec-neural-probes) handles
backend selection and builds the compression report.

## Framed spike rasters

The compiled module exports `raster_stream_encode_frame`,
`raster_stream_decode_frame`, `raster_stream_encode_raster` and
`raster_stream_decode_raster`. The frame API uses contiguous binary int8
rasters. The whole-raster API emits SSCF and reports total spikes, active
channels summed over frames, and the largest frame size. The decoder
validates lengths, dimensions and padding before allocating the raster.
[Public streaming codec](spike_codec.md#streaming-codec-real-time).

## Address-event packets

The compiled module exports `address_event_encode_raster` and
`address_event_decode_raster`. The encoder accepts contiguous binary int8
samples, timestamp escape width and minimum neuron address width. It
returns the complete packet plus original spike and encoded event counts.
The decoder checks timestamp and address bounds before allocating the
raster. [Public AER codec](spike_codec.md#aer-codec-neuromorphic).

## Installed-wheel interface evidence

The [engine bridge contracts](../guides/engine_bridge_contracts.md#capturing-the-installed-interface)
document `tools/engine_abi_inventory.py`, which captures the facade and compiled
module after wheel installation. Its installed-origin guard refuses a checkout
facade; retained captures include module hashes, complete interface metadata and
exact alias identities. Compare inventories under an equal Python/NumPy/feature
profile, then use per-binding behavioral tests for input layouts, exception and
mutation contracts, numerical goldens and supported pickle state.

The qualified default interface reference is
`tests/fixtures/engine_abi_default.json`, for Linux x86-64, CPython 3.12 and
NumPy 2.5.4 with default engine features. Its matching wheel job compares all
interface fields; other runtime profiles retain separate measured captures.
See [model compatibility and migration](../guides/engine_bridge_contracts.md#model-compatibility-and-migration)
for canonical source models and their explicit retained project profiles.
