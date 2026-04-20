# Edge — AER Interconnect Router

Python façade over the Go-based AER (Address-Event Representation) UDP
mesh router used for multi-FPGA deployments. The Go service does the
actual packet routing; the Python :class:`AERRoutingDaemon` manages its
lifecycle (build, start, stop) so a Python experiment can spin up the
router, run traffic through it, then tear it down cleanly.

```python
from sc_neurocore.edge.aer_router import AERRoutingDaemon

router = AERRoutingDaemon(port=9000)
router.start(build=True)        # compile and launch the Go binary
# ... run experiment, publish AER events to udp://localhost:9000 ...
router.stop()
```

---

## `AERRoutingDaemon`

| Method                       | Purpose                                                                                |
| ---------------------------- | -------------------------------------------------------------------------------------- |
| `__init__(port=9000)`        | Locates `accel/go/services/aer_router/` relative to package root. Sets UDP port.       |
| `start(build=True)`          | Optional `go build`, then spawns `./aer_router` as a background subprocess.            |
| `stop()`                     | SIGTERMs the router and waits up to 2 s for it to exit; silently idempotent.           |

The underlying Go program binds to the configured UDP port, receives
AER packets (``{neuron_id, timestamp, fabric_id}``), forwards them
according to the active routing table, and replies with delivery
acknowledgements. See the Go source for the wire format.

---

## Related Go services

The `accel/go/services/` tree ships four Go modules that can be driven
from Python via subprocess wrappers:

| Module         | Purpose                                                      | Python wrapper                                         |
| -------------- | ------------------------------------------------------------ | ------------------------------------------------------ |
| `aer_router`   | UDP mesh router for AER events                               | :class:`AERRoutingDaemon` (this page)                  |
| `hil_debugger` | WebSocket telemetry server for live debugging                | :class:`sc_neurocore.debug.hil_server.HILServerDaemon` |
| `services`     | Phase 2 shared services (metrics, health, discovery)         | —                                                      |
| `services_ext` | Phase 6 extension services (multi-FPGA coordination)         | —                                                      |

Each Go module has its own `go.mod` so it can be built and tested in
isolation (`go test ./...`); the accompanying `main_test.go` files
exercise the service end-to-end.

Bench binaries (`services_bench`, `services_ext_bench`) are
intentionally **gitignored** — regenerate locally via
``go test -bench -c`` when needed.

---

## Toolchain expectations

- Go 1.21+ on `PATH` (for `AERRoutingDaemon.start(build=True)`).
- Linux / macOS host. No Windows support — the daemon spawns via
  Unix-style process APIs.

---

## Reference

- Python wrapper: `src/sc_neurocore/edge/aer_router.py`.
- Go source: `src/sc_neurocore/accel/go/services/aer_router/main.go` +
  `main_test.go`.
- Sibling services: `src/sc_neurocore/accel/go/services/{hil_debugger,services,services_ext}/`.

::: sc_neurocore.edge.aer_router
    options:
      show_root_heading: true
