# Spike-Level Debugger + HIL Telemetry

Two complementary debug surfaces:

1. **Offline spike trace analysis** — `SpikeTracer` + `Analyzer` tools
   for post-mortem debugging of SC runs (recorded spike trains,
   causality chains, divergence between an SC run and a reference).
2. **Hardware-in-the-Loop (HIL) telemetry** — `HILServerDaemon` +
   `HILDebugger` wrappers for a live, Go-backed WebSocket telemetry
   server that streams per-layer frames to a GUI or a CI log.

---

## 1. Offline spike trace — `tracer` + `analyzer`

`SpikeTracer` records an `ExecutionTrace` of every tick in an SC
network. The analyser walks that trace to find the first divergence
point against a reference trace, or to reconstruct the causal chain
behind a given output event.

::: sc_neurocore.debug.tracer
    options:
      show_root_heading: true
      members:
        - SpikeTracer
        - ExecutionTrace

::: sc_neurocore.debug.analyzer
    options:
      show_root_heading: true
      members:
        - find_divergence
        - causal_chain
        - spike_diff
        - DivergencePoint
        - CausalEvent

---

## 2. Live HIL telemetry — `HILServerDaemon` + `HILDebugger`

When the SC pipeline runs against real FPGA hardware (or simulated
hardware), per-layer `HILFrame` messages (see [Protobuf
Schemas](proto.md)) stream through a WebSocket. The Go binary
`accel/go/services/hil_debugger/` does the actual streaming; the Python
side manages its lifecycle.

```python
from sc_neurocore.debug.hil_debugger import HILDebugger

dbg = HILDebugger(port=8081)
dbg.start()
# ... run SC network, HIL events published to ws://localhost:8081 ...
dbg.stop()
```

### 2.1 `HILServerDaemon`

The low-level daemon lifecycle manager.

| Method                        | Purpose                                                                                         |
| ----------------------------- | ----------------------------------------------------------------------------------------------- |
| `__init__(port=8081)`         | Locates the Go source dir (source-tree first, then installed-scripts fallback).                 |
| `start(build=True) -> bool`   | Optional `go build -o hil_debugger main.go`, then spawns the binary with `HIL_PORT=<port>`.     |
| `stop()`                      | SIGTERMs the daemon, waits up to 3 s, then SIGKILLs if still alive.                             |
| `is_running` (property)       | ``True`` if the server process is live (poll-based).                                            |

Start-up is gated on a `GET /health` readiness probe with a 5-second
timeout; `start()` returns `False` (with a printed diagnostic) on timeout.

### 2.2 `HILDebugger`

Thin convenience wrapper around :class:`HILServerDaemon` — one `start`
/ `stop` call for the common case. Delegates every method to the
underlying daemon.

Use `HILDebugger` for quick experiments; drop to `HILServerDaemon`
when you need to customise `build`, port, or error handling.

### 2.3 What the server publishes

Each live frame is a `vision2030.telemetry.HILFrame` protobuf
message (see [Protobuf Schemas](proto.md)) carrying:

- `timestamp_ms` — wall-clock timestamp in ms.
- `layer_id` — IR layer name.
- `metrics` — compact `BitstreamMetadata` (length, correlation,
  popcount).
- `sample_spikes` — optional raw packed-bitstream tensor for layers
  under active inspection.

The frame cadence is controlled by the Go service, not the Python
daemon — typically one frame per ms per layer.

---

## 3. Toolchain expectations for HIL

- Go 1.21+ on `PATH` (only required when `start(build=True)`).
- A free TCP port for `HIL_PORT` (default `8081`).
- No WebSocket client is bundled — consume with any standard WS client
  (JavaScript, Python `websockets`, `wscat`, …).

---

## Reference

- Source: `src/sc_neurocore/debug/{tracer,analyzer,hil_server,hil_debugger}.py`.
- Go daemon: `src/sc_neurocore/accel/go/services/hil_debugger/` (main.go +
  main_test.go).
- Wire protocol: [Protobuf Schemas](proto.md).

::: sc_neurocore.debug.hil_server
    options:
      show_root_heading: true

::: sc_neurocore.debug.hil_debugger
    options:
      show_root_heading: true
