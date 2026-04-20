# HDL Generation + Hardware Safety

Verilog / SystemVerilog generation from Python SC descriptions, plus a
dedicated hardware safety-monitor module backed by Lean 4 proofs (see
[Formal Proofs](formal.md)).

## Python-driven RTL generation

`generate_verilog()` converts SC layer / neuron descriptions to
synthesisable Verilog modules. Supports Q8.8 fixed-point, LFSR
encoders, popcount trees, and event-driven AER.

```python
from sc_neurocore.hdl_gen import generate_verilog
```

IR compiler pipeline: Python → intermediate representation →
SystemVerilog / MLIR (CIRCT backend). 19 hand-written Verilog modules
plus an equation-to-Verilog compiler for arbitrary ODEs.

---

## Hardware safety monitor — `neuro_safe_monitor`

`src/sc_neurocore/hdl_gen/safety/safety_monitor.sv` implements a
nanosecond-scale runtime monitor that halts the SC fabric the moment
any of six safety invariants is violated. Every invariant is proven in
Lean 4 (`safety_bounds.lean`); each monitor property carries a P-tag
pointing back at the matching theorem.

### Property map

| Tag | Invariant (hardware signal semantics)                 | Lean theorem                  |
| --- | ----------------------------------------------------- | ----------------------------- |
| P1  | `halt ↔ coherence < COHERENCE_LIMIT`                  | `monitor_soundness`           |
| P2  | coherence can only grow once the safe zone is entered | `safe_transition`             |
| P3  | SC variance proxy: `4·k·(N−k) ≤ N²`                   | `sc_precision_bound`          |
| P4  | SC addition result ≤ `SC_DENOM`                       | `sc_add_preserves_range`      |
| P5  | LIF membrane ≤ `LIF_V_MAX`                            | `lif_membrane_bounded`        |
| P6  | \|SCC numerator\| ≤ `SC_DENOM`                        | `correlation_range`           |

### Parameters

| Parameter          | Default         | Meaning                                           |
| ------------------ | --------------- | ------------------------------------------------- |
| `MAX_CURRENT`      | `16'h7FFF`      | Q8.8 current cap (~127.99)                        |
| `MAX_VOLTAGE`      | `16'hC000`      | Q8.8 voltage cap (sign-extended −1.0)             |
| `COHERENCE_LIMIT`  | `16'h0100`      | Monitor's safety-mode floor = 1.0 in Q8.8         |
| `SC_DENOM`         | `16'h0100`      | Bitstream length `N` = 256                        |
| `LIF_V_MAX`        | `16'hC000`      | Upper bound for LIF membrane potential            |

### Testbench

`src/sc_neurocore/hdl_gen/safety/tb_safety_monitor.sv` walks every
property with adversarial stimuli — drives `coherence` across the
limit boundary (P1/P2), produces out-of-range SC outputs (P3/P4),
over-excites the LIF (P5), and forces `|SCC|` above the denominator
(P6). A passing testbench is a necessary (not sufficient) condition
for the Lean proofs to apply to the RTL.

---

## ASIC synthesis flow

`src/sc_neurocore/hdl_gen/openroad_flow/run_asic_flow.sh` pushes the
safety monitor (or any user-specified `.sv`) through:

1. **Yosys** — gate-level synthesis with cell-library mapping.
2. **OpenROAD** (optional) — place & route with area / timing reports.

```
./run_asic_flow.sh                          # default target: safety_monitor
./run_asic_flow.sh --target custom.sv       # point at another SV module
./run_asic_flow.sh --docker                 # run through OpenROAD Docker image
```

Outputs:

- `build/synth/` — Yosys synthesis results.
- `build/reports/` — area, timing, cell-utilisation reports.

Requires `yosys` on `PATH`; OpenROAD is optional (the script prints a
clear diagnostic when it's missing).

---

## Reference

- Python API: `src/sc_neurocore/hdl_gen/` (package root).
- Safety RTL: `src/sc_neurocore/hdl_gen/safety/safety_monitor.sv` +
  `tb_safety_monitor.sv`.
- Flow driver: `src/sc_neurocore/hdl_gen/openroad_flow/run_asic_flow.sh`.
- Matching Lean proofs: [Formal Proofs](formal.md).

::: sc_neurocore.hdl_gen
    options:
      show_root_heading: true
