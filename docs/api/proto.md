# Protobuf Schemas

Language-agnostic message definitions for the multi-FPGA + HIL telemetry
surface. Two `.proto` files form the wire contract between the Python
controller, the Go services (`aer_router`, `hil_debugger`,
`services`, `services_ext`) and any third-party tooling that consumes
live SC-NeuroCore traces.

---

## 1. `core.proto` — primitive payloads

Package: `vision2030.core`. Go package alias:
`github.com/anulum/sc-neurocore/vision2030/proto/core`.

### `Tensor`

```proto
message Tensor {
  repeated uint32 shape = 1;
  oneof data {
    bytes float_data = 2;   // Packed float32
    bytes int_data   = 3;   // Packed int32
    bytes bit_data   = 4;   // Packed bitstream (bool8)
  }
}
```

`shape` is row-major. `data` is a `oneof` — exactly one encoding is
populated per message. `bit_data` uses 8-bit boolean packing to match
the SC bitstream format used by the Rust engine's
`BitStreamTensor`.

### `BitstreamMetadata`

```proto
message BitstreamMetadata {
  uint32 length       = 1;
  float  correlation  = 2;
  uint32 popcount     = 3;
}
```

Lightweight summary packet attached to tensors that carry packed SC
bitstreams: `length` is the bitstream count, `correlation` the SCC
against a reference, `popcount` the total bit count — enough to
reconstruct the bitstream's mean activity without shipping it.

---

## 2. `telemetry.proto` — HIL debugger frames

Package: `vision2030.telemetry`. Imports `core.proto`.

### `HILFrame`

```proto
message HILFrame {
  int64  timestamp_ms  = 1;
  string layer_id      = 2;
  vision2030.core.BitstreamMetadata metrics = 3;
  vision2030.core.Tensor           sample_spikes = 4;
}
```

One `HILFrame` per layer per millisecond is emitted by the Go
`hil_debugger` service. `layer_id` matches the IR layer name. `metrics`
is the compact summary; `sample_spikes` is an optional raw trace for
the layers under active inspection.

---

## 3. Code generation

Both files carry explicit `go_package` options so `protoc --go_out`
produces the types in the expected import path.

Typical build:

```bash
protoc \
  --proto_path=src/sc_neurocore/proto \
  --go_out=. \
  --go_opt=paths=source_relative \
  src/sc_neurocore/proto/core.proto src/sc_neurocore/proto/telemetry.proto
```

Python bindings (when needed) are generated with `protoc --python_out=`
against the same files — no SC-NeuroCore-specific tooling is required.

---

## 4. Limitations

- No versioning metadata in the messages themselves. Wire-compatibility
  relies on proto3's standard forward/backward rules: add fields, do
  not renumber.
- `sample_spikes` in `HILFrame` is unbounded. Callers should respect
  HIL bandwidth budgets — the debugger's streaming pipeline caps the
  sample rate, but naive consumers can still overflow.

---

## Reference

- `src/sc_neurocore/proto/core.proto`
- `src/sc_neurocore/proto/telemetry.proto`
- Go services that consume these messages:
  `src/sc_neurocore/accel/go/services/{aer_router,hil_debugger,services,services_ext}/`.
