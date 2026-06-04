# HDL Generation + Hardware Safety

Two cooperating paths out of the Python design space and into silicon:

1. **Verilog / SystemVerilog emission** — `VerilogGenerator` converts a
   Python description of an SC network (dense SC-layer instances,
   neuron cores, LFSR encoders, popcount trees, event-driven AER) into
   a synthesisable top-level module.
2. **SPICE emission** — `SpiceGenerator` converts a NumPy weight matrix
   into a memristor-crossbar SPICE netlist for analogue simulation and
   post-layout verification.
3. **Formally grounded safety RTL** — a hand-written
   `neuro_safe_monitor` SystemVerilog module with six runtime
   invariants, each of which is the mirror of a theorem in
   `safety_bounds.lean` (see [Formal Proofs](formal.md)). An OpenROAD
   ASIC-flow driver pushes the monitor (or any user SV) through Yosys
   synthesis + optional OpenROAD place-and-route.

```python
from sc_neurocore.hdl_gen import VerilogGenerator, SpiceGenerator
```

---

## 1. Mathematical formalism

### 1.1 Q8.8 fixed-point number line

All monitor signals are Q8.8 (8 integer, 8 fractional), giving

$$
x_{\text{Q8.8}} = \lfloor x \cdot 2^{8} \rfloor,
\qquad
x \in \bigl[-128,\; 128 - 2^{-8}\bigr],
\qquad
\Delta = 2^{-8} \approx 0.0039.
$$

The sign-extended 16-bit representation wraps at $2^{15}$; the monitor
treats `probe_scc_numer` as signed, everything else as unsigned. Parameter
defaults correspond to:

- `MAX_CURRENT = 16'h7FFF` → $+127.9961$,
- `MAX_VOLTAGE = 16'hC000` → $-16384/256 = -64.0$ in the sign-extended
  reading, used as the *upper bound on the saturation sign-magnitude*,
- `COHERENCE_LIMIT = 16'h0100` → $+1.0$,
- `SC_DENOM = 16'h0100` → stream length $N = 256$,
- `LIF_V_MAX = 16'hC000` → same upper bound on the LIF membrane.

### 1.2 Safety invariants (SystemVerilog ↔ Lean)

The monitor's six properties each latch a 1-bit violation flag when
true; any of the seven Boolean literals below asserts `hardware_halt`:

$$
\begin{aligned}
[P1]\; & v_{\text{cv}} = (I > I_{\max}) \vee (V > V_{\max}), \\
[P1]\; & v_{\text{coh}} = (\mathrm{coh} < \Theta), \\
[P2]\; & v_{\text{mono}} = (\mathrm{coh}_{t} < \mathrm{coh}_{t-1}), \\
[P3]\; & v_{\text{prec}} = (k > N), \\
[P4]\; & v_{\text{sc+}} = (a \oplus b > N), \\
[P5]\; & v_{\text{mem}} = (V_{\text{mem}} > V_{\max}), \\
[P6]\; & v_{\text{scc}} = (|\eta_{\mathrm{SCC}}| > d_{\mathrm{SCC}}).
\end{aligned}
$$

[P1–P2] correspond to the Lean theorems ``monitor_soundness`` and
``safe_transition`` (pure-core Lean 4, proved). [P3–P6] correspond to
three axiomatised theorems (``sc_precision_numerator_bound``,
``sc_add_preserves_range``, ``scc_bounded``) for which the Mathlib proof
roadmap is documented in `safety_bounds.lean`; [P6] is additionally
proved constructively at the hardware level by the absolute-value
computation on line 87 of the monitor.

### 1.3 Popcount tree latency

For an $N$-bit stream, the popcount tree has depth $\lceil \log_{2} N
\rceil$ — with $N = 256$ that is 8 pipeline stages. At a clock
period $T_{\text{clk}}$ the popcount result lags the input by

$$
t_{\text{pc}} = \lceil \log_{2} N \rceil \cdot T_{\text{clk}}.
$$

This is one of the input probes to [P3], so the monitor's end-to-end
response time is bounded by $t_{\text{pc}} + t_{\text{mon}}$ where
$t_{\text{mon}} \leq 1$ clock cycle. Absolute nanosecond figures
depend on the PDK's cell delays and are emitted by OpenROAD's
`report_checks`; they are not claimed here without a measured run.

### 1.4 SPICE crossbar conductance mapping

`SpiceGenerator.generate_crossbar(weights, …)` maps weights $w_{ij} \in
[0,\,1]$ to memristor conductances

$$
G_{ij} = G_{\mathrm{off}} + w_{ij} \cdot (G_{\mathrm{on}} - G_{\mathrm{off}}),
\qquad
R_{ij} = 1 / G_{ij},
$$

with $G_{\mathrm{on}} = 100\,\mu\mathrm{S}$ (10 kΩ) and
$G_{\mathrm{off}} = 1\,\mu\mathrm{S}$ (1 MΩ). Each row drives an
independent voltage source (`Vin_r`) and each column ties to a 1 kΩ
load resistor (`Rload_c`), giving a column voltage

$$
V_{\text{out},c} = \sum_{r} V_{\text{in},r} \cdot \frac{R_{\text{load}}}
                                     {R_{r,c} + R_{\text{load}}},
$$

which is the analogue MAC (multiply-accumulate) that the crossbar
implements physically.

---

## 2. Theory (why this particular design)

### 2.1 Separate compilers for synth vs analogue

The :class:`VerilogGenerator` is intentionally thin (~90 LOC) because
the heavy lifting lives in the 46 hand-written Verilog cores under
the repo-root `hdl/` tree (e.g. `sc_dense_layer_core.v`,
`sc_bitstream_encoder.v`, `sc_aer_router.v`, `sc_lif_neuron.v`,
`sc_firing_rate_bank.v`, plus 8 matching formal-property files under
`hdl/formal/`). The Python generator wires the pre-verified cores
together by name; it does **not** try to synthesise novel RTL on the
fly. This separation keeps the RTL part verifiable (the cores each
have their own testbench) and the Python part trivial (string
templating). Under `src/sc_neurocore/hdl_gen/safety/` the repo
additionally ships two SystemVerilog files — `safety_monitor.sv` and
its testbench — which is what the `neuro_safe_monitor` described in
the rest of this page refers to.

### 2.2 Formal-spec ↔ RTL 1:1 correspondence

Every property the monitor checks has a named Lean theorem. Crucially,
the monitor's expression is the *same shape* as the theorem's
conclusion — e.g. ``v_monotone = (probe_coherence < prev_coherence)``
is the negation of ``monotone_coherence (c1 c2 : Q8_8) : c2 ≥ c1``.
This makes it straightforward to audit the RTL against the proofs by
diff, and it is a precondition for future work that would turn the
correspondence into a machine-checked bridge (SymbiYosys + SVA → Lean).

### 2.3 Sticky violation flags

`violation_flags` are latched with `|=`-style sticky behaviour; once
a property has fired, it remains set until `rst_n` deasserts the
register file. This mirrors the way aviation flight-control monitors
behave (Rushby 1993): a single violation must not be "washed out" by a
subsequent good cycle — the safety-case analysis depends on every
violation being observable.

The precision-overflow trap follows the same fail-observable rule for
mixed-precision datapaths. `sc_precision_overflow_trap` exposes both
`trap_event_vector`/`trap_event`, which mirror accepted overflow lanes in
the same cycle, and `trap_vector`/`trap_latched`, which retain every lane
until the host asserts `clear_trap` or reset. Clear and reset dominate
concurrent overflow pulses, so host intervention cannot accidentally
re-latch stale saturation telemetry. Optional
`SC_NEUROCORE_ASSERTIONS` properties bind the no-silent-overflow and
sticky-latch contracts for formal or simulation audit runs.

Live-control parameter banks are generated from `MMIOUpdateSpec` with
`generate_live_parameter_bank(...)`. The emitted AXI4-Lite RTL uses
BRAM/distributed RAM style hints per bank, fixed control/status register
addresses, staged low/high write-data registers, an `update_valid|commit`
command, flattened `parameter_words` output, and host-visible trap clear/status
signals. This lets a deployed design hot-swap weights or phase-coupling
coefficients while keeping the precision and trap contracts auditable.

### 2.4 Nanosecond response budget

`hardware_halt` is a pure combinational OR of the seven violation
signals, latched on the next clock edge. The worst-case latency from
any input signal going bad to `hardware_halt` rising is therefore one
clock period plus the fan-in delay of the combinational OR tree.
Closed-loop f_max on a real PDK remains to be measured — the
generic-cell `synth` run in §7 does **not** produce a mapped critical
path number. Once run against SKY130 hd (or comparable) Liberty,
we expect the combinational path to stay well under the SC-tile
clock period (typically 2–4 ns at 250–500 MHz), but that is a
pending measurement, not a claim.

Either way, this is three to four orders of magnitude faster than the
~500 µs loop of the Python
:class:`sc_neurocore.safety_cert.stochastic_doctor.StochasticDoctor`
runtime check. The two layers are complementary: the hardware monitor
catches single-cycle excursions; the Python doctor catches slow
statistical drift.

### 2.5 OpenROAD vs commercial tools

The flow driver supports both "Yosys only" (always available) and
"Yosys + OpenROAD" (optional) modes. OpenROAD is chosen over
commercial PnR because (a) it is open-source and reproducible inside
Docker, (b) the safety-monitor module is small enough that OpenROAD's
gate counts and PPA (power-performance-area) results are directly
comparable to commercial tools on this design size, and (c) the
provenance chain is end-to-end inspectable — an auditor can re-run
every step.

### 2.6 SPICE as a sanity layer, not a specification

The memristor-crossbar netlist is emitted from Python so that the
*same* weight matrix used for the SC network can be pushed through
analogue SPICE and the two outputs compared. It is not the system's
source of truth — the Q8.8 / SC stream is — but it serves as a
ground-truth cross-check on the post-layout behaviour of mixed-signal
tiles.

---

## 3. Position in the pipeline

```
    ┌─────────────────────┐       ┌───────────────────────┐
    │  Python SC network  │──────▶│   VerilogGenerator    │
    │  (layers + cores)   │       │  (string templating)  │
    └─────────────────────┘       └──────────┬────────────┘
              │                              │
              │                              ▼
              │                       top.sv + cores/*.sv
              │                              │
              │                              ▼
              │                     ┌──────────────────┐
              │                     │ neuro_safe_monitor│◀── formal.md theorems
              │                     └─────────┬────────┘
              │                               │
              ▼                               ▼
     ┌─────────────────┐              ┌──────────────┐
     │ SpiceGenerator  │              │  run_asic_   │
     │ memristor x-bar │              │  flow.sh     │
     └─────────────────┘              └──────┬───────┘
              │                               │
              ▼                               ▼
      analogue .sp netlist         Yosys synth → OpenROAD PnR
```

- **Upstream.** The :class:`VerilogGenerator` is called by the
  `OrganismEmitter` in `evo_substrate.md` whenever a fit organism
  needs hardware deployment.
- **Downstream.** The generated SV feeds into the ASIC flow driver;
  the safety monitor hooks every tile's probe bus unconditionally.

---

## 4. Features

- Python-driven top-level Verilog emission.
- 46 hand-written Verilog cores under `hdl/` (dense SC layer, LFSR,
  AER router, popcount tree, bitstream encoder, LIF neuron, firing-
  rate bank, AXI-Lite cfg, DMA controller, …) plus 8 formal-property
  files under `hdl/formal/`, each with its own testbench.
- Equation-to-Verilog compiler for arbitrary ODEs (used by the HH,
  Izhikevich, FitzHugh-Nagumo tiles).
- 6-property runtime safety monitor mirroring Lean 4 theorems.
- Adversarial testbench (`tb_safety_monitor.sv`) that forces every
  property to fire.
- Sticky per-property violation flags.
- Nanosecond-budget `hardware_halt` output.
- OpenROAD / Yosys ASIC-flow driver with optional Docker fallback.
- Memristor crossbar SPICE netlist emitter with configurable
  $G_{\mathrm{on}}$ / $G_{\mathrm{off}}$ / load resistance.

---

## 5. Usage

### 5.1 Emit a 3-layer SC network

```python
from sc_neurocore.hdl_gen import VerilogGenerator

gen = VerilogGenerator(module_name="my_sc_net_top")
gen.add_layer("Dense", "l1", {"n_neurons": 32})
gen.add_layer("Dense", "l2", {"n_neurons": 32})
gen.add_layer("Dense", "l3", {"n_neurons": 10})
rtl = gen.generate()
gen.save_to_file("build/my_sc_net_top.sv")
```

Emits a module with `clk`, `rst_n`, `input_bus[7:0]`, `output_bus[7:0]`
and three `sc_dense_layer_core` instances chained via 8-bit wires.

### 5.2 Synthesise with the safety monitor

```bash
cd src/sc_neurocore/hdl_gen/openroad_flow
./run_asic_flow.sh                           # default: safety_monitor.sv
./run_asic_flow.sh --target my_sc_net_top.sv # point at generated RTL
./run_asic_flow.sh --docker                  # run through the OpenROAD image
```

Outputs:

- `build/synth/` — Yosys synthesis results (gate-level .v, stats).
- `build/reports/` — area, timing, cell-utilisation.

Real Yosys 0.33 run on the default monitor design (`synth` command,
generic cell library — no PDK mapping):

```
=== neuro_safe_monitor ===
  Number of wires:                333
  Number of wire bits:            493
  Number of public wires:          14
  Number of cells:                347
    $_ANDNOT_   104    $_AND_       2    $_DFFE_PN0P_   1
    $_DFF_PN0_   22    $_MUX_      15    $_NAND_       17
    $_NOR_       12    $_NOT_      18    $_ORNOT_      17
    $_OR_        92    $_XNOR_     14    $_XOR_        33
  Wall: 0.25 s (Yosys 0.33 + abc)
```

Those are the exact numbers emitted by

```
yosys -p "read_verilog -sv .../safety_monitor.sv;
          hierarchy -top neuro_safe_monitor;
          synth; stat"
```

on 2026-04-20. Mapping to SKY130 hd (for tape-out area / timing)
requires `dfflibmap -liberty` + `abc -liberty` with a Liberty file
that is not bundled with Yosys Debian — install the `sky130_fd_sc_hd`
PDK and re-run the OpenROAD flow driver to get PPA numbers.

### 5.3 Emit a memristor-crossbar SPICE netlist

```python
import numpy as np
from sc_neurocore.hdl_gen import SpiceGenerator

W = np.random.default_rng(7).random((16, 16))
SpiceGenerator.generate_crossbar(W, "build/xbar_16x16.sp")
```

Example generated block:

```
* Memristor Crossbar 16x16
.PARAM VDD=1.0

Vin_0 in_0 0 DC 0.0
...
R_0_0 in_0 out_0 12345.67
R_0_1 in_0 out_1 56789.01
...
Rload_0 out_0 0 1k
...
.END
```

---

## 6. API reference

### 6.1 `VerilogGenerator`

| Method                               | Purpose                                                                      |
| ------------------------------------ | ---------------------------------------------------------------------------- |
| `__init__(module_name)`              | Names the top-level module.                                                  |
| `add_layer(layer_type, name, params)`| Appends a Dense / LFSR / AER / popcount layer spec.                          |
| `generate() -> str`                  | Returns the top-level Verilog source as a string.                            |
| `save_to_file(path)`                 | Writes generated Verilog to disk; `OSError` raised on failure.               |

### 6.2 `SpiceGenerator`

| Method                                         | Purpose                                             |
| ---------------------------------------------- | --------------------------------------------------- |
| `generate_crossbar(weights, filename)` (static) | Emits `<filename>.sp` with sources, memristors, loads. |

### 6.3 `neuro_safe_monitor` (SystemVerilog)

| Port / parameter        | Direction | Width | Purpose                                 |
| ----------------------- | --------- | ----- | --------------------------------------- |
| `MAX_CURRENT`           | param     | 16    | Q8.8 current cap                        |
| `MAX_VOLTAGE`           | param     | 16    | Q8.8 voltage cap                        |
| `COHERENCE_LIMIT`       | param     | 16    | Q8.8 floor for [P1]                     |
| `SC_DENOM`              | param     | 16    | SC stream length $N$                    |
| `LIF_V_MAX`             | param     | 16    | upper bound for LIF membrane            |
| `clk`, `rst_n`          | in        | 1     | standard                                |
| `probe_current`         | in        | 16    | [P1]                                    |
| `probe_voltage`         | in        | 16    | [P1]                                    |
| `probe_coherence`       | in        | 16    | [P1/P2]                                 |
| `probe_popcount_k`      | in        | 16    | [P3]                                    |
| `probe_sc_add_result`   | in        | 16    | [P4]                                    |
| `probe_membrane`        | in        | 16    | [P5]                                    |
| `probe_scc_numer`       | in        | 16 (signed) | [P6]                              |
| `probe_scc_denom`       | in        | 16    | [P6]                                    |
| `hardware_halt`         | out       | 1     | asserts on any violation (sticky)       |
| `violation_flags[5:0]`  | out       | 6     | one sticky bit per property             |

### 6.4 ASIC-flow driver (`run_asic_flow.sh`)

| Flag                   | Purpose                                                             |
| ---------------------- | ------------------------------------------------------------------- |
| `--target <file.sv>`   | Override the default `safety_monitor.sv` target.                    |
| `--docker`             | Run the full Yosys + OpenROAD stack in the `openroad/flow` image.   |
| (no flag)              | Run Yosys synthesis only; skip OpenROAD if the binary is missing.   |

---

## 7. Verified benchmarks

The HDL subsystem is not latency-critical on the Python side — the
heavy lifting runs once at synthesis time. Still, we measure the
three Python entry points for repeatability:

| Operation                                          | Throughput    | Latency   |
| -------------------------------------------------- | ------------- | --------- |
| `VerilogGenerator.generate` (3-layer top, in-memory) | 281 822 gen/s |  3.55 µs |
| `SpiceGenerator.generate_crossbar` (16×16, disk write) | 2 551 gen/s | 392 µs  |
| `SpiceGenerator.generate_crossbar` (64×64, disk write) | 231 gen/s   | 4.33 ms |
| `yosys synth; stat` on `safety_monitor.sv`          | 4.06 runs/s  | 247 ms  |

Yosys `stat` report (Yosys 0.33, default abc mapping to generic
cell library — no PDK):

| Metric                       | Value |
| ---------------------------- | ----- |
| Wires                        | 333   |
| Wire bits                    | 493   |
| Public wires                 | 14    |
| Cells                        | 347   |
| DFFs (`$_DFF_PN0_`)          | 22    |
| DFF-enable (`$_DFFE_PN0P_`)  | 1     |
| Max combinational depth (reported by `abc`) | not emitted without liberty |

**Interpretation.**

- The Python emitters are negligible on the design-time path: one
  3-layer top costs 5 µs, one 64×64 memristor netlist costs 4 ms
  (dominated by `open(..., "w")` syscalls, not the string build).
- The full `synth; stat` flow on the monitor completes in ~250 ms
  cold from shell, so the safety-monitor synth gate fits comfortably
  in a pre-commit hook budget.
- Cell count (347) is ~2.5× the DFF count; the majority are combinational
  AND-NOT / MUX / OR terms implementing the seven Boolean violation
  conditions and their sticky-flag muxes. That matches the design: 6
  properties × ~50 gates each, plus the six 1-bit sticky registers
  and the 16-bit `prev_coherence` register (22 DFFs total = 16 +
  6).
- Mapped timing (f_max, critical path ns) is **not** emitted without a
  Liberty file — those numbers appear only after `abc -liberty <lib>`.
  The claim that the monitor "closes timing at ≥500 MHz" is therefore
  deferred to the real PDK run; the current release gate only asserts
  the synth completes without errors.

Python timings are `time.perf_counter` deltas from
`benchmarks/bench_hdl_gen.py`; Yosys figures are the literal `stat`
output of Yosys 0.33 on Ubuntu 24.04.

---

## 8. Citations

1. Rushby J. (1993). *Formal methods and digital systems validation for
   airborne systems*. NASA Contractor Report 4551. (Sticky-violation
   rationale.)
2. Wolf C. et al. (2012–present). *Yosys Open Synthesis Suite*.
   https://yosyshq.net/yosys/
3. Ajayi T. et al. (2019). *OpenROAD: Toward a Self-Driving, Open-Source
   Digital Layout Implementation Tool Chain*. GOMAC Tech.
4. Strukov D.B., Snider G.S., Stewart D.R., Williams R.S. (2008). *The
   missing memristor found*. Nature 453:80–83. (Memristor model basis.)
5. Nagel L.W., Pederson D.O. (1973). *SPICE (Simulation Program with
   Integrated Circuit Emphasis)*. UC Berkeley ERL Memo ERL-M382.
6. Chakrabarti C. et al. (2018). *Designing for reliability in
   stochastic computing*. ACM TRETS 11(3), Article 21. (Safety-monitor
   background.)
7. Šotek M. (2026). *SC-NeuroCore: formally grounded safety RTL*.
   Internal report, ANULUM.

---

## 9. Cell-level breakdown — where the 347 gates go

The Yosys `stat` output in §7 lists 12 cell types; the physical reason
each category exists is worth documenting because it gives a direct
handle on future optimisation.

| Cell type       | Count | What it implements in this design                                         |
| --------------- | ----: | ------------------------------------------------------------------------- |
| `$_ANDNOT_`     | 104   | violation terms of shape `a & !b` (range checks, `coh < Θ`, `k > N`, …)   |
| `$_OR_`         |  92   | inner OR trees of the seven violation expressions                         |
| `$_XOR_`        |  33   | the `probe_scc_numer < 0 ? ~x+1 : x` two's-complement path                |
| `$_DFF_PN0_`    |  22   | 16-bit `prev_coherence` + 6-bit sticky `violation_flags`                  |
| `$_NOT_`        |  18   | inverters for the `!rst_n` + signed-abs path                              |
| `$_ORNOT_`      |  17   | `a \| !b` fragments of the monotone check                                 |
| `$_NAND_`       |  17   | synth-mapper output of mixed AND-OR patterns                              |
| `$_MUX_`        |  15   | latch-vs-new selection of the 6 sticky violation bits on the DFF clock    |
| `$_XNOR_`       |  14   | equality comparators on the 16-bit probes                                 |
| `$_NOR_`        |  12   | synth-mapper output                                                       |
| `$_AND_`        |   2   | residual AND gates                                                        |
| `$_DFFE_PN0P_`  |   1   | `hardware_halt` edge-triggered register                                   |

22 DFFs + 1 DFFE = 23 stateful cells. Of the 324 combinational cells,
~220 are directly attributable to the six violation expressions (≈37
gates per property after sharing); the remainder are the
two's-complement path for signed `probe_scc_numer` and the sticky-flag
muxes.

---

## 10. Reproducibility + determinism

Every number in §7 and §9 can be re-derived from the committed repo
with a clean clone + two commands:

```bash
python benchmarks/bench_hdl_gen.py            # Python emission + yosys synth
./src/sc_neurocore/hdl_gen/openroad_flow/run_asic_flow.sh   # full flow
```

The benchmark script writes `benchmarks/results/bench_hdl_gen.json`
atomically; a CI check that diffs this JSON across runs flags any
regression in generation throughput. The Yosys cell counts are
deterministic — the same `read_verilog; synth; stat` sequence on Yosys
0.33 against the same `safety_monitor.sv` produces byte-identical
stat output across runs. If your Yosys version differs, the cell
breakdown will too; pin Yosys through the `nixpkgs.yosys_0_33` or
`apt install yosys=0.33*` channel for bit-reproducible gate counts.

`run_asic_flow.sh` writes its artefacts into `build/synth/` and
`build/reports/`, both of which are gitignored; the script writes a
`build/run_metadata.json` with host, Yosys version, commit SHA, and a
SHA-256 digest of the input `.sv` file so post-hoc audit of any
synthesised bitstream can verify the provenance chain.

---

## 11. Known limitations

- **No equivalence check between Python-emitted SV and the pre-built
  cores.** The top-level module chains instances by name; if a user
  mistyped `sc_dense_layer_core`'s port list in Python, synthesis will
  fail with a port-mismatch error rather than a helpful Python-side
  diagnostic.
- **ODE-to-Verilog compiler lives outside this module.** The
  `VerilogGenerator` class does not expose it yet — users who want the
  HH / Izhikevich / FHN RTL paths consume the pre-generated `.sv`
  files under the repo-root `hdl/` tree directly.
- **SPICE emitter ignores wire parasitics.** The netlist contains only
  ideal memristors and load resistors; BEOL stack capacitance and
  access-transistor resistance must be added by hand for sub-100 nm
  nodes.
- **No power reporting.** Yosys synthesis does not produce switching-
  activity estimates; real power figures require OpenROAD with a VCD
  trace or a commercial tool. The flow driver does not wire that yet.
- **Monitor parameters are hard-coded in the RTL.** Changing
  `COHERENCE_LIMIT` from 1.0 to 0.75 requires editing the SV file or
  overriding the parameter on instantiation — there is no Python-side
  API for reparametrising a generated top-level monitor yet.
- **No SVA (SystemVerilog Assertions).** The six properties are
  encoded as combinational Boolean expressions plus sticky flags. A
  future refactor will express them as `assert property`
  SystemVerilog assertions so they can be proved by SymbiYosys
  directly, closing the gap to the Lean specification.
- **No formal RTL-vs-spec equivalence.** The SystemVerilog monitor and
  the Lean theorems are hand-aligned 1:1 by matching the shape of the
  Boolean expressions (see §2.2). A machine-checked proof that the RTL
  *implements* the Lean statements would require a SystemVerilog → Lean
  embedding such as Kôika or Verilog-Lean; neither is wired in yet.
- **Flow driver is Yosys-only by default.** The OpenROAD path is
  optional and needs the OpenROAD binary (or Docker image) on the
  host. Without it, the driver prints a clear diagnostic and exits
  zero after the Yosys stage; the release gate does not block on
  OpenROAD.
- **Memristor model is ideal-linear.** `SpiceGenerator` maps weights
  linearly onto $[G_{\mathrm{off}},\,G_{\mathrm{on}}]$; no non-linear
  I–V curve, no drift, no endurance / retention model. Analogue
  verification against device silicon requires an augmented model
  (e.g. the VTEAM or Yakopcic memristor).

---

## Reference

- Python API:
  - `src/sc_neurocore/hdl_gen/__init__.py` (package root, 19 LOC)
  - `src/sc_neurocore/hdl_gen/verilog_generator.py` (86 LOC)
  - `src/sc_neurocore/hdl_gen/spice_generator.py` (54 LOC)
- Safety RTL:
  - `src/sc_neurocore/hdl_gen/safety/safety_monitor.sv` (118 LOC)
  - `src/sc_neurocore/hdl_gen/safety/tb_safety_monitor.sv` (202 LOC)
- Flow driver: `src/sc_neurocore/hdl_gen/openroad_flow/run_asic_flow.sh`
  (229 LOC).
- Matching Lean proofs: [Formal Proofs](formal.md).

::: sc_neurocore.hdl_gen
    options:
      show_root_heading: true
