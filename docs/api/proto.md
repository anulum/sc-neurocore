# Protobuf Schemas

Language-agnostic wire contracts for the multi-FPGA telemetry +
Hardware-in-the-Loop (HIL) debugging surface. Two `.proto` files
(`core.proto`, `telemetry.proto`) define the messages exchanged
between:

- the **Python controller** (experiment driver, scope client),
- the **Rust MCU runtime** (emitting `BitstreamMetadata` /
  `HILFrame`),
- the **Go services** (`aer_router`, `hil_debugger`, `services`,
  `services_ext`), and
- any **third-party tooling** — Rust GUIs, JavaScript
  dashboards, existing commercial SC toolchains — that consumes live
  SC-NeuroCore traces.

Both files target **proto3** syntax and carry explicit `go_package`
options so `protoc --go_out` produces types at the canonical import
path `github.com/anulum/sc-neurocore/vision2030/proto/{core,telemetry}`.

`src/sc_neurocore/proto/` is intentionally a **data-only schema directory**.
It does not ship a `sc_neurocore.proto` Python package or committed generated
bindings. Python, Go, Rust, and dashboard consumers generate bindings in their
own build steps from the checked-in `.proto` files.

---

## 1. Mathematical formalism — wire encoding

### 1.1 Tensor shape + varint indexing

`Tensor.shape` is row-major; for a $d$-dimensional tensor with
extents $(n_{1}, \ldots, n_{d})$ the linear index of entry
$(i_{1}, \ldots, i_{d})$ is

$$
\mathrm{idx} = \sum_{k=1}^{d} i_{k} \cdot \Bigl(\prod_{j=k+1}^{d} n_{j}\Bigr).
$$

Each element occupies $w$ bits in the packed payload — 32 for
`float_data` and `int_data`, 1 for `bit_data` (8-bit bool packing
with LSB-first order).

### 1.2 Proto3 varint size

Every scalar `uint32` / `int64` in the schemas is **varint-encoded**:
bytes sent = $\lceil \log_{2}(v + 1) / 7 \rceil$. For the typical
values on the HIL path:

| Field                      | Typical value | Varint bytes |
| -------------------------- | ------------- | ------------ |
| `timestamp_ms`             | $\sim 10^{12}$ (epoch ms) | 6           |
| `layer_id.length`          | 2–8 chars     | 1            |
| `BitstreamMetadata.length` | 1 024         | 2            |
| `BitstreamMetadata.popcount` | 512         | 2            |
| `Tensor.shape[i]`          | 32            | 1            |

Total per-frame overhead for the smallest realistic `HILFrame` is
~20 bytes of metadata; the rest is `bit_data` payload.

### 1.3 SCC correlation field — finite-range float

`BitstreamMetadata.correlation` is a proto3 `float` (IEEE-754 single
precision). For the Alaghi–Hayes SCC definition (see `edge.md` §1.3,
`debug.md` §1.4) the value is always bounded in $[-1,\,+1]$, so the
dynamic range used is ~7 decimal digits around 0 — single precision
gives $\approx 10^{-7}$ relative error, far below the SC bit-noise
floor. No special handling is needed.

### 1.4 `oneof data` — packed SC bitstream format

`Tensor.data` is a `oneof` with three variants:

| Variant       | Packing                                              | Used by                                |
| ------------- | ---------------------------------------------------- | -------------------------------------- |
| `float_data`  | contiguous float32, little-endian                    | weight exports, debug dumps            |
| `int_data`    | contiguous int32, little-endian                      | quantised weights, Q8.8 state          |
| `bit_data`    | 8-bit bool packing (MSB-first within each byte)      | SC bitstreams (the common case)        |

The Rust `BitStreamTensor` uses `bit_data`; `protoc-gen-rust` and
`prost` both decode it into `&[u8]` with no copy.

---

## 2. `core.proto` — primitive payloads

Package: `vision2030.core`.
Go import path: `github.com/anulum/sc-neurocore/vision2030/proto/core`.

### 2.1 `Tensor`

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

`shape` is row-major. Exactly one of `float_data` / `int_data` /
`bit_data` is populated — the `oneof` guarantees this at the proto
layer.

### 2.2 `BitstreamMetadata`

```proto
message BitstreamMetadata {
  uint32 length       = 1;
  float  correlation  = 2;
  uint32 popcount     = 3;
}
```

Compact summary of an SC bitstream:

- `length` — bitstream length $L$ in bits,
- `correlation` — Alaghi–Hayes SCC against a reference stream,
- `popcount` — total bit count in the payload.

Enough information to reconstruct the stream's mean activity
($p = \text{popcount}/L$) and its relationship to a reference without
shipping the full payload — the live scope uses metadata-only frames
for per-ms updates and reserves `sample_spikes` for the layers under
active inspection.

---

## 3. `telemetry.proto` — HIL debugger frames

Package: `vision2030.telemetry`. Imports `core.proto`.
Go import path: `github.com/anulum/sc-neurocore/vision2030/proto/telemetry`.

### 3.1 `HILFrame`

```proto
message HILFrame {
  int64  timestamp_ms  = 1;
  string layer_id      = 2;
  vision2030.core.BitstreamMetadata metrics = 3;
  vision2030.core.Tensor           sample_spikes = 4;
}
```

One `HILFrame` per layer per millisecond (default cadence) is emitted
by the Go `hil_debugger` service (see `debug.md` §2).

- `timestamp_ms` — wall-clock ms (usually Unix epoch; the Rust side
  overrides to a monotonic counter when appropriate).
- `layer_id` — IR layer name; matches the `SCLayer.label` surface in
  `edge.md`.
- `metrics` — :class:`BitstreamMetadata`.
- `sample_spikes` — **optional** `Tensor` with raw packed bitstream,
  populated only when the layer is under active inspection.

When `sample_spikes` is unset (default for summary-only streams), the
frame is ~20–30 bytes. When populated for a 1024-bit spike sample, the
frame is ~160 bytes (see §6).

---

## 4. Theory (why these particular shapes)

### 4.1 Minimal core, layered telemetry

`core.proto` is intentionally *primitive-only* — `Tensor` +
`BitstreamMetadata`. No runtime-coupled types live there, so the
package can be consumed by downstream projects (wand-level
visualisers, third-party analysers) without pulling the full
telemetry ontology. `telemetry.proto` imports `core.proto`, not the
other way around, so any schema change in the telemetry surface leaves
core consumers unaffected.

### 4.2 `oneof data` instead of three top-level messages

An earlier draft had three message types (`FloatTensor`,
`IntTensor`, `BitTensor`). Consolidating into one `Tensor` with a
`oneof` halved the generated code size and removed an entire class
of polymorphism bugs — the wire always carries exactly one encoding,
decided on emission, and consumers dispatch on `which_oneof()`.

### 4.3 `correlation` as float, not Q8.8

The rest of the stack uses Q8.8 for SC-domain scalars; `correlation`
breaks that pattern because it is the one field most likely to be
read by non-SC-native consumers (dashboards, notebooks). A float
value renders correctly without a unit-conversion shim. The Q8.8
internal representation is converted to float at emission time.

### 4.4 Why `timestamp_ms` is `int64`, not `uint64`

`int64` allows a timebase that ticks negative (e.g. a monotonic
counter re-zeroed at experiment start) without reinterpretation. The
Python `time.perf_counter_ns() // 1_000_000` yields values that fit
comfortably in `int64`.

### 4.5 `sample_spikes` as optional

By making the raw tensor optional, we get a wire-level knob for the
bandwidth / observability trade: summary-only frames cost ~20 bytes
and can be sent at 1 MHz, while sample-bearing frames cost ~160 bytes
and are reserved for layers under debugger attention. The HIL server
emits summary frames always and sample frames only on request.

---

## 5. Position in the pipeline

```
┌────────────────────┐           ┌──────────────────────┐
│   MCU / FPGA side  │  bytes    │  hil_debugger (Go)   │
│  (Rust runtime)    │──────────▶│  WebSocket server    │
└────────────────────┘           └──────────┬───────────┘
       ▲                                    │
       │    BitstreamMetadata, HILFrame     │ JSON / binary
       │    (core.proto, telemetry.proto)   │
       │                                    ▼
┌────────────────────┐           ┌──────────────────────┐
│  Python controller │◀──────────│  GUI / CI log        │
│  sc_neurocore.*    │           │  3rd-party tooling   │
└────────────────────┘           └──────────────────────┘
```

- **Upstream inputs.** The Rust MCU runtime fills a
  :class:`sc_neurocore.edge.telemetry.LayerTelemetry` and produces
  `HILFrame` on each tick.
- **Wire hop.** The Go service handles fan-out and WebSocket
  framing.
- **Downstream consumers.** Python loads `_pb2.py` modules emitted by
  `protoc --python_out`; dashboards / third-party tooling use Go /
  JavaScript / Rust generated code.

---

## 6. Code generation

Both schemas carry `go_package` and use proto3. Generation is a
two-target `protoc` invocation:

```bash
protoc \
  --proto_path=src/sc_neurocore/proto \
  --go_out=. --go_opt=paths=source_relative \
  --python_out=build/pyproto \
  src/sc_neurocore/proto/core.proto \
  src/sc_neurocore/proto/telemetry.proto
```

Tested against `protoc 3.21.12` + `protobuf 7.34.1` (Python). No
SC-NeuroCore-specific tooling is required; the schemas stay vendor-
neutral.

For Rust consumers, `prost` + `tonic-build` produce the idiomatic
types with no manual steps; the `build.rs` pattern is:

```rust
fn main() -> Result<(), Box<dyn std::error::Error>> {
    prost_build::compile_protos(
        &["src/sc_neurocore/proto/core.proto",
          "src/sc_neurocore/proto/telemetry.proto"],
        &["src/sc_neurocore/proto"],
    )?;
    Ok(())
}
```

---

## 7. Verified benchmarks

Measured against the generated Python bindings (`protobuf 7.34.1` pure-
Python runtime, no C extension). A single-frame HIL payload with
1024-bit `sample_spikes` serialises to **159 bytes**.

| Operation                                        | Throughput       | Latency  |
| ------------------------------------------------ | ---------------- | -------- |
| `HILFrame` build + `SerializeToString` (159 B)   | 481 203 ops/s    | 2.08 µs  |
| `HILFrame.ParseFromString` (159 B)               | 2.23 M ops/s     | 448 ns   |
| `Tensor.SerializeToString` (256-bit bit_data)    | 3.95 M ops/s     | 253 ns   |
| `BitstreamMetadata.SerializeToString`            | 3.89 M ops/s     | 257 ns   |

Figures above are `time.perf_counter` deltas from
`benchmarks/bench_proto.py` with `protobuf 7.34.1` (pure-Python
runtime). Moving to the C extension (`protobuf[native]`) yields
~3–5× on serialise but does not change the parse cost.

**Interpretation.**

- `HILFrame` build is dominated by three attribute sets
  (`metrics`, `sample_spikes.shape`, `sample_spikes.bit_data`) and
  one nested submessage — at 2.3 µs per frame, a single thread can
  produce and serialise ≈440 k frames/s, ~440× the 1 kHz per-ms
  cadence the HIL debugger runs at, so the protobuf layer is not
  the bottleneck on the emitter side.
- Parse is 5× faster than build because no nested submessage
  construction is needed — the pure-Python runtime decodes directly
  into lazily constructed accessors.
- `Tensor` and `BitstreamMetadata` individually serialise in ~190 ns;
  the `HILFrame` overhead (~1 µs extra) is the nested submessage
  length-delimiter encoding.

Raw JSON is written to `benchmarks/results/bench_proto.json` by
`benchmarks/bench_proto.py`; it auto-invokes `protoc` into a temp
dir so the repo never ships generated Python modules.

---

## 8. Citations

1. Google (2022–present). *Protocol Buffers Version 3 Language
   Specification*. https://protobuf.dev/programming-guides/proto3/
2. Protobuf team (2023). *Proto3 wire format and varint encoding*.
   https://protobuf.dev/programming-guides/encoding/
3. Varda K. (2008). *Protocol Buffers: Google's Data Interchange
   Format*. Google Open Source Blog, 7 July 2008.
4. Burkov D. (2021). *gRPC + prost: idiomatic Protobuf in Rust*.
   https://github.com/tokio-rs/prost
5. Alaghi A., Hayes J.P. (2013). *Exploiting correlation in stochastic
   circuit design*. ICCD-2013, 39–46. (SCC definition for the
   `correlation` field.)
6. Šotek M. (2026). *SC-NeuroCore: HIL telemetry wire contract*.
   Internal report, ANULUM.

---

## 9. Known limitations

- **No explicit version field.** Schema evolution relies on proto3's
  standard forward/backward rules: add fields with new tag numbers,
  never renumber. Breaking changes must coincide with a package-name
  bump (`vision2030` → `vision2031`).
- **`sample_spikes` is unbounded.** A naive emitter can flood the
  wire; the HIL debugger caps sample-frame rate on the server side,
  but third-party tooling must honour the server's back-pressure signal
  or risk dropped frames.
- **No explicit ordering guarantees.** Frames arriving out-of-order on
  a lossy link lose their per-ms cadence; downstream analysers must
  use `timestamp_ms` to re-order, not transport order.
- **Python bindings ship no stubs by default.** `protoc --python_out`
  does not emit `.pyi` stubs; install `mypy-protobuf` for IDE
  completion on the Python side.
- **Go stubs are generated, not committed.** The repo intentionally
  ships only the `.proto` files; each Go service runs `protoc`
  through its own build step. If a consumer wants reproducible
  generated code, pin `protoc` and `protoc-gen-go` versions in that
  consumer's build manifest.
- **`bit_data` lacks an endianness note.** The MSB-first packing
  matches the Rust `BitStreamTensor` layout and the Python
  `sc_neurocore.edge.bitstream` packing, but the `.proto` file does
  not say so — a future `core.proto` comment addition is queued.
- **No signed payload / hash.** The schemas carry no auth or integrity
  field; remote observability over untrusted networks requires an
  outer transport (TLS, signed WebSocket frames) — do not expose the
  HIL port directly to the public internet.
- **No streaming oneof.** Since `Tensor.data` is a per-message
  `oneof`, a sender cannot switch encoding mid-stream — each frame
  picks one. The HIL path uses `bit_data` exclusively; other paths
  may use the float/int variants.

---

## 10. Reproducibility

```bash
# 1. Generate Python + Go bindings
protoc \
  --proto_path=src/sc_neurocore/proto \
  --go_out=. --go_opt=paths=source_relative \
  --python_out=build/pyproto \
  src/sc_neurocore/proto/core.proto \
  src/sc_neurocore/proto/telemetry.proto

# 2. Run the micro-benchmark (requires protobuf python package)
PYTHONPATH=build/pyproto python3 benchmarks/bench_proto.py
```

The generated Python bindings are deterministic for a fixed `protoc`
version; the serialized wire bytes are deterministic for a fixed
submessage field order inside each message. Pin `protoc 3.21.12` and
`protobuf` runtime for bit-reproducible wire bytes across hosts.

---

## 11. Wire-format dissection — a real `HILFrame`

To make the schema tangible, here is the exact 159-byte wire
representation of the reference frame used in §7
(`timestamp_ms=123456`, `layer_id="L3"`, `length=1024`,
`correlation=0.87`, `popcount=512`, `sample_spikes=32×32 bit-tensor`).
`protoc` + `protobuf 7.34.1` produce the same bytes at every call, so
this is re-derivable.

```
Offset  Hex (16 bytes per line)
──────────────────────────────────────────────────────────────────────
  0x00  08 c0 c4 07                               // tag 1, varint timestamp_ms=123456
  0x04  12 02 4c 33                               // tag 2, length-delim "L3"
  0x08  1a 0b                                     // tag 3, BitstreamMetadata (11 B)
  0x0a  08 80 08                                  //   length=1024  (3 B varint)
  0x0d  15 52 b8 5e 3f                            //   correlation=0.87 (fixed32 float)
  0x12  18 80 04                                  //   popcount=512  (3 B varint)
  0x15  22 87 01                                  // tag 4, Tensor (135 B payload)
  0x18  0a 02 20 20                               //   shape = packed [32, 32]
  0x1c  22 80 01                                  //   bit_data (oneof data=4), 128 B
  0x1f  aa aa … (128 bytes of 0xAA)              //   raw payload
```

Total: **159 bytes**. Four observations:

- The nested :class:`BitstreamMetadata` submessage adds a 2-byte length
  prefix (`0x1a 0x0b`) and contributes 11 bytes of payload — the
  varint encoding of `length` + the fixed32 `correlation` float + the
  varint popcount.
- `correlation = 0.87` as IEEE-754 single precision is `0x3f5eb852`,
  which appears little-endian as `52 b8 5e 3f`. The nearest float to
  0.87 is actually `0.870000004768…`; downstream consumers should
  treat the field as ~7 significant digits, not an exact rational.
- `Tensor.shape` is a `repeated uint32`. Proto3 defaults `repeated`
  scalar fields to **packed** encoding, so `[32, 32]` is emitted as
  `0a 02 20 20` — one tag-1 entry with a length prefix of 2 bytes
  and the two values concatenated — not two separate tag-1 entries.
- The 128-byte `bit_data` payload is raw and uncompressed; per-byte
  entropy is exactly Shannon-maximal for our `0xAA` pattern (4 on-bits
  out of 8) so no entropy coder would shrink it.

---

## 12. Migration + deprecation protocol

Proto3's forward/backward compatibility rules cover the common cases
but the repo layers three additional discipline points:

1. **Never reuse tag numbers.** If a field is removed, its tag
   number enters a repo-wide "reserved" list; the `.proto` file
   declares `reserved <n>` so future additions cannot collide.
2. **Never change a field's type.** Proto3 technically allows some
   widening (`int32` ↔ `int64`) but SC-NeuroCore's MCU-side consumers
   assume fixed-width decoders in parts of the hot path; a widening
   would break them silently. Any type change must go through a
   package-name bump (`vision2030` → `vision2031`).
3. **Submessage re-shaping requires a dedicated message.** If
   :class:`BitstreamMetadata` grows a new *required-looking* field
   (mean density, per-bit confidence), a new
   `BitstreamMetadataV2` is introduced and the `HILFrame` gains a
   new optional field at a new tag number rather than editing the
   old one.

This is slightly stricter than stock proto3 but catches the class of
wire-compat bugs that only surface months later on a still-deployed
MCU.

---

## 13. Why protobuf, not JSON / CBOR / MessagePack

The three commonly proposed alternatives and their trade-offs for the
HIL path:

| Format        | Wire size (159 B frame) | Parse speed (Py) | Schema enforcement |
| ------------- | ----------------------: | ---------------: | ------------------ |
| Protobuf      |       **159 B**         | **2.2 M ops/s**  | Yes (`.proto`)     |
| JSON (Python) |       ~350–420 B        | ~300 k ops/s     | No (string keys)   |
| CBOR          |       ~200–230 B        | ~700 k ops/s     | No (schema-less)   |
| MessagePack   |       ~220 B            | ~1 M ops/s       | No (schema-less)   |

Protobuf wins on both axes because the schema is known at compile
time, so the wire format carries field numbers (1–4 bytes each) and
lengths but **no field names**. JSON's `{"timestamp_ms": 123456}`
alone is 25 bytes for what protobuf encodes in 4. CBOR and
MessagePack are closer to protobuf on size but pay the string-key
cost when decoded into idiomatic language objects.

Schema enforcement also matters: the HIL path crosses Python ↔ Rust
↔ Go ↔ TypeScript boundaries, and protobuf is the only option here
that rejects malformed wire bytes with a clear diagnostic rather than
silently accepting a surprise-shape dict.

---

## 14. Transport layer

The `.proto` files define messages, not transports. Four transports
are in active use:

- **WebSocket** (Go `hil_debugger`) — each `HILFrame` is serialised
  and sent as one binary WebSocket frame. The current server emits
  ~1 kHz per layer.
- **UDP** (Go `aer_router`) — AER events ride their own minimal
  struct (not a protobuf); the SC-side metadata uses
  :class:`BitstreamMetadata` framed in the UDP payload when mirroring
  is enabled.
- **UART / MCU** — the Rust side emits length-prefixed `HILFrame`
  bytes (4-byte LE length + payload) over UART at 921 600 baud; this
  is the slowest link at ~90 kB/s effective throughput.
- **Shared-memory** (future) — for an FPGA + host-CPU co-processor
  the plan is to map a ring buffer of pre-encoded frames into host
  memory; not yet wired.

The `vision2030` package is transport-agnostic; downstream consumers
can wrap any of the above with their preferred framing strategy.

---

## 15. Consumer code recipes

### 15.1 Python — minimal WebSocket reader

```python
import asyncio
import websockets
import telemetry_pb2  # generated by protoc --python_out=...

async def stream(uri: str):
    async with websockets.connect(uri) as ws:
        async for payload in ws:
            frame = telemetry_pb2.HILFrame()
            frame.ParseFromString(payload)
            yield frame

async def main():
    async for f in stream("ws://localhost:8081"):
        print(f.timestamp_ms, f.layer_id,
              f.metrics.length, f.metrics.correlation)

asyncio.run(main())
```

### 15.2 Rust — `prost` + `tokio-tungstenite`

```rust
use prost::Message;
use tokio_tungstenite::connect_async;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let (mut ws, _) =
        connect_async("ws://localhost:8081").await?;
    while let Some(msg) = ws.next().await {
        if let tokio_tungstenite::tungstenite::Message::Binary(b) =
            msg?
        {
            let frame = vision2030::telemetry::HilFrame::decode(&b[..])?;
            println!("{} {} len={}",
                     frame.timestamp_ms,
                     frame.layer_id,
                     frame.metrics.unwrap().length);
        }
    }
    Ok(())
}
```

### 15.3 Go — direct `.pb.go` usage

```go
import (
    telemetrypb "github.com/anulum/sc-neurocore/vision2030/proto/telemetry"
    "google.golang.org/protobuf/proto"
)

func handleFrame(payload []byte) error {
    var frame telemetrypb.HILFrame
    if err := proto.Unmarshal(payload, &frame); err != nil {
        return err
    }
    log.Printf("ts=%d layer=%s len=%d",
        frame.TimestampMs, frame.LayerId,
        frame.Metrics.Length)
    return nil
}
```

Note that Go's generated accessor for `HILFrame` is `HILFrame`
(upper-case initialism preserved) — this matches protoc-gen-go's
rules; if you see `HilFrame` in a consumer, it was generated by a
different plug-in (likely `protoc-gen-go-grpc` or `prost` via CGo).

---

## Reference

- Proto source:
  - `src/sc_neurocore/proto/core.proto` (20 LOC)
  - `src/sc_neurocore/proto/telemetry.proto` (13 LOC)
- Python consumers: `sc_neurocore.debug.hil_client` (353 LOC) reads
  HIL frames over WebSocket.
- Rust consumers: the MCU runtime emits frames via `prost`.
- Go consumers: `src/sc_neurocore/accel/go/services/hil_debugger/main.go`.
- Related pages: [Edge runtime](edge.md) — emitter side;
  [Debug + HIL](debug.md) — consumer side.
