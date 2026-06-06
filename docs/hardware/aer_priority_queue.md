<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
# AER Priority Queue and Backpressure Contract

`sc_aer_priority_queue.v` is the NEU-C.4 hardware-control queue for
event-driven SC-NeuroCore designs.  It sits between sparse AER fanout
generation and downstream event consumers when the design needs explicit
quality-of-service ordering under backpressure.

## Contract

Lower numeric priority is more urgent:

| Priority value | Meaning |
|----------------|---------|
| `0` | Critical control or safety event |
| `1` | High-priority inference/control event |
| `2` | Normal event |
| `3` | Best-effort event |

The queue guarantees:

- Strict priority: the lowest numeric priority present in the queue is selected.
- FIFO ties: events with the same priority leave in arrival order.
- Explicit input backpressure through `in_ready`.
- Sticky `dropped_event` trap when an upstream event arrives while the queue is full.
- Sticky `critical_deadline_violation` trap when a priority-0 event remains blocked longer than `MAX_LATENCY_CYCLES`.
- Bounded occupancy reported through `occupancy`.

## Router integration

`sc_aer_router.v` now exposes priority and ready/valid ports:

| Port | Direction | Purpose |
|------|-----------|---------|
| `in_event_ready` | output | Upstream admission signal for sparse source events. |
| `in_priority` | input | Priority copied to every fanout packet emitted for the source event. |
| `out_event_ready` | input | Downstream backpressure signal. |
| `out_priority` | output | Priority associated with the emitted target packet. |
| `queue_full` | output | Priority queue cannot accept another fanout packet without a drain. |
| `dropped_event` | output | Sticky queue-overflow trap. |
| `critical_deadline_violation` | output | Sticky priority-0 latency trap. |

With `PRIORITY_ENABLED=0`, the router preserves the legacy serialized fanout
path and treats an unconnected `out_event_ready` as ready for simulation
backward compatibility.  Production top-level designs should still connect
`out_event_ready` explicitly.

With `PRIORITY_ENABLED=1`, each fanout packet enters `sc_aer_priority_queue`.
The fanout walker stalls when the queue deasserts `in_ready`; downstream
consumers throttle through `out_event_ready`.

## Verification surfaces

The committed verification path is module-specific:

| Surface | File |
|---------|------|
| Python reference model | `src/sc_neurocore/hdl/aer_priority_queue_reference.py` |
| Behavioural tests and HDL simulation | `tests/test_aer_priority_queue.py` |
| Formal harness | `hdl/formal/sc_aer_priority_queue_formal.v` |
| SymbiYosys job | `hdl/formal/sc_aer_priority_queue.sby` |
| Benchmark evidence | `benchmarks/results/local_python_2026-06-04_aer_priority_queue.json` |

`sby`, Yosys, and `cvc5` are required to discharge the committed bounded formal
job.  The current job uses a 2-deep queue instance and a 12-cycle BMC window to
prove the NEU-C.4 edge cases: strict-priority overtaking, FIFO ties under equal
priority, full backpressure, sticky drop traps, and priority-0 deadline traps.
The separate NEU-C.2 timing-aware formal framework remains responsible for
broader reusable induction and multi-engine timing proofs.

## Benchmark boundary

The 2026-06-04 benchmark is not an FPGA timing claim.  It is a regression and
contract artefact that records Python reference timing, SystemVerilog Yosys
elaboration, zero priority-order violations, zero FIFO-tie violations,
backpressure rejections, deadline traps, CPU affinity, cgroup effective CPU set,
host load, governor, and frequency context.  The local run used a temporary
runtime cpuset shield rather than plain `taskset`; boot-time
`isolcpus`/`nohz_full` remains the stronger option for final production
throughput claims.  FPGA claims still require board-level timing and power
reports.
