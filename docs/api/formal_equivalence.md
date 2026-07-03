<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->

# Machine-Checked RTL Equivalence & Properties

**Modules:** `sc_neurocore.compiler.equivalence_miter`,
`sc_neurocore.compiler.equivalence_check`,
`sc_neurocore.compiler.formal_property_check`,
`sc_neurocore.compiler.formal_evidence`
**Shared runner:** `sc_neurocore.compiler._sby_runner`
**Tools:** [SymbiYosys](https://github.com/YosysHQ/sby) (`sby`) + Yosys + an SMT
engine (`z3` by default)

Two complementary machine-checked flows share one SymbiYosys spine. The
**equivalence** flow proves the compiler's **generated Verilog** computes the same
function as an **independent reference** module — not for a sampled set of
stimuli, but for *every* input sequence up to a bounded depth. The **property**
flow proves a *single* RTL module satisfies its own safety obligations (bounded
error, bounded length, no overflow) expressed as bound SystemVerilog assertions.
Both replace the prior text-only equivalence *sketch*
(`intelligence.equivalence_sketch`) and the standalone `.sby` script generator
(`sby_formal`) with a runnable proof that returns a real verdict.

The single subprocess boundary — write the `.sby`, invoke `sby`, parse the
verdict, extract the counterexample — lives in `_sby_runner` so both runners
interpret `sby` output identically.

---

## 1. Method — sequential miter + bounded model checking

A **miter** instantiates the device-under-test (DUT) and the reference side by
side, drives both with identical free inputs and a shared reset, and asserts
that their outputs agree on every post-reset cycle:

```
        ┌──────────── free inputs (checker explores all values) ────────────┐
   clk ─┤                                                                    │
        │   ┌───────────┐  outputs_dut                                       │
        ├──▶│    DUT     ├──────────────┐                                    │
        │   └───────────┘               ▼                                    │
 rst_n ─┤                          assert(==) ── every post-reset cycle      │
 (gen.) │   ┌───────────┐               ▲                                    │
        └──▶│ reference  ├──────────────┘                                    │
            └───────────┘  outputs_ref                                       │
```

Feeding the miter to bounded model checking (BMC) asks the SMT solver whether
*any* input sequence of up to `depth` cycles can drive the two outputs apart.
`PASS` means none can — a proof of equivalence to that depth. `FAIL` returns a
concrete counterexample trace.

### Reset discipline

The miter uses neither an `initial` block nor a simulation-only `always #5 clk`
construct — both over-constrain the formal initial state into `PREUNSAT`.
Instead a free-running counter (initialised to zero, the one initial value the
checker honours) holds the active-low reset asserted for `reset_cycles` clocks,
and the equivalence assertions are gated on the post-reset window so the two
modules are compared only once both have been driven into their reset state.

### Bounded vs unbounded

BMC establishes equivalence up to `depth` cycles from reset. Unbounded proof by
k-induction (`mode="prove"`) is available but is **not** the default: the
induction step starts from an arbitrary state where the outputs agree yet the two
instances' *hidden* state may differ (an unreachable configuration), and then
diverges. It converges only when a **reachable-state invariant** ties the internal
states together.

### Unbounded equivalence via whitebox state taps

The invariant one wants is a hierarchical reference — `dut.v_reg == ref.v`. yosys
0.33 does not resolve hierarchical references (it parses `dut.v_reg` as one
escaped, undriven identifier) and silently ignores SystemVerilog `bind`, so
neither route reaches an instance's internal state. The working alternative is to
**expose** that state: `whitebox_taps.expose_state_taps` instruments each module
with continuous-assign taps that surface the relevant registers as extra output
ports. The miter then compares those taps like any other output, and the tap
equality *is* the state-matching invariant that makes k-induction converge.

Taps are pure observation (an `output wire` plus one `assign`; no new register, no
rewritten logic), so the instrumented module is behaviourally identical on its
original ports; a tap whose source is a constant lets two structurally different
modules (e.g. one with a refractory counter, one without) present the same tap
interface. With the membrane register and refractory counter tapped, the LIF miter
proves **unbounded** by k-induction.

The remaining bound is SMT tractability, not soundness: `z3` closes the induction
step for a narrow (4-bit) fixed-point datapath in seconds but slows sharply as the
multiplier widens — abstracting the multiplier (uninterpreted-function reasoning by
congruence) is the lever for wide datapaths and remains future work.

---

## 2. Usage

```python
from sc_neurocore.compiler.equivalence_miter import parse_module_interface
from sc_neurocore.compiler.equivalence_check import (
    formal_tools_available,
    prove_equivalence,
)

dut_verilog = open("hdl/sc_lif_neuron.v").read()
ref_verilog = open("hdl/equiv/sc_lif_reference.v").read()

# Resolve the shared interface (parameter-dependent widths need their values).
ports = parse_module_interface(ref_verilog, "sc_lif_reference", params={"DATA_WIDTH": 16})

if formal_tools_available():
    result = prove_equivalence(
        dut_verilog,
        ref_verilog,
        ports,
        dut_top="sc_lif_neuron",
        ref_top="sc_lif_reference",
        dut_params={"DATA_WIDTH": 16, "FRACTION": 8, "V_THRESHOLD": 256, "REFRACTORY_PERIOD": 0},
        ref_params={"DATA_WIDTH": 16, "FRACTION": 8, "V_THRESHOLD": 256},
        depth=4,
    )
    if result.proven:
        print(f"equivalent to depth {result.depth}")
    else:
        print(f"counterexample: {result.counterexample}\ntrace: {result.trace_path}")
```

`parse_module_interface` also accepts an explicit `list[MiterPort]` if you prefer
not to parse the header; `build_equivalence_miter` returns the miter Verilog
directly for inspection or a custom flow.

---

## 3. Reference

### `equivalence_miter` (pure)

| Symbol | Description |
|--------|-------------|
| `MiterPort(name, width, signed, direction)` | One port of the shared interface. |
| `parse_module_interface(verilog, top, *, params=None)` | Parse the ANSI port list; resolve parameter-dependent widths. |
| `build_equivalence_miter(dut_top, ref_top, io_ports, *, ...)` | Emit the sequential-equivalence miter Verilog. |

### `equivalence_check` (runner)

| Symbol | Description |
|--------|-------------|
| `formal_tools_available()` | `True` when `sby` and `yosys` are on `PATH`. |
| `prove_equivalence(dut_verilog, ref_verilog, io_ports, *, ...)` | Build the miter, run `sby`, return an `EquivalenceResult`. |
| `EquivalenceResult` | `proven`, `verdict`, `mode`, `depth`, `engine`, `returncode`, `counterexample`, `trace_path`, `summary`. |

A `PASS` sets `proven=True`; a `FAIL` sets `proven=False` with the failing
assertion and counterexample-trace path. An `sby` tool/setup failure (as opposed
to a disproof) raises `RuntimeError`, as does a timeout or absent toolchain.

---

## 4. Property proofs — adaptive-precision evidence

An adaptive-precision plan assigns each synapse a bit width and a stochastic
bitstream length under a total-error budget. `formal_evidence` renders that
budget into a **bounded-error monitor** (a synthesisable datapath, the RTL `.v`)
and a **bound assertion checker** (the SVA `.sv`) whose immediate `assert` /
`assume` statements encode three obligations:

1. **bounded error** — the accumulated quantisation-plus-stochastic error never
   exceeds the claimed total error bound over the bitstream length;
2. **bounded length** — the length sequencer never runs past the declared
   bitstream length;
3. **no overflow** — the error accumulator is wide enough that the next
   accumulation never wraps, which keeps obligation 1 sound.

`formal_property_check.prove_property` proves the RTL satisfies the bound SVA the
same way the equivalence runner proves a miter: emit a `.sby`, run it through
`_sby_runner`, parse the verdict. Because the accumulator stops updating after
`max_bitstream_length` steps, a BMC to `max_bitstream_length + 2` cycles exhausts
the reachable state space — the bounded proof is *complete*, not merely bounded.

### Unbounded k-induction

The checker also carries a **strengthening lemma**,
`err_acc <= step_count * per_step`, which — unlike the bound `err_acc <= total`
on its own — is **1-inductive**. With it, `unbounded=True` (or
`prove_property(mode="prove")`) proves the obligations by **k-induction**, an
*unbounded* proof whose depth is a small constant **independent of the bitstream
length** (BMC completeness, by contrast, scales with the length). The lemma is
trivially true under BMC too, so both modes share one checker.

k-induction has a third outcome besides proved and disproved: when the base case
holds but the induction step does not converge, the run is **inconclusive** —
`proven=False`, `verdict="UNKNOWN"`, no counterexample — the property may well
hold but was not proved. This is recorded honestly (never as a pass); it is *not*
a `RuntimeError` (which is reserved for tool/setup failures).

> **Toolchain note.** yosys 0.33 silently ignores SystemVerilog `bind`
> directives, so the checker is instantiated explicitly inside the monitor under
> `` `ifdef FORMAL `` (a macro `read -formal` defines) rather than bound in. Plain
> synthesis strips it; formal builds elaborate it. The yosys frontend also rejects
> concurrent `assert property (@(posedge clk) …)`, so the obligations are written
> as **immediate** `assert`/`assume` in a clocked `always` block.

```python
from sc_neurocore.compiler.adaptive_precision import (
    assign_synapse_precisions,
    write_precision_formal_evidence_bundle,
)

assignments = assign_synapse_precisions(layer_weights, target_error=0.05)

# execute=False (default): write the bundle only — deterministic, no tools.
# execute=True: machine-check it when sby/yosys/z3 are present, and record the
# real verdict (a skip reason is recorded instead when the toolchain is absent —
# never a fabricated pass).
# unbounded=True: prove by k-induction (mode prove) instead of bounded BMC.
manifest = write_precision_formal_evidence_bundle(
    out_dir, assignments, execute=True, unbounded=True
)
claim = manifest["formal_claim"]
print(claim["symbiyosys_executed"], claim["formal_proof_passed"], claim["proof_verdict"])
```

`prove_property(rtl_verilog, sva_verilog, *, top, mode="bmc", depth=..., ...)`
returns a `PropertyProofResult` (`proven`, `verdict`, `mode`, `depth`, `engine`,
`returncode`, `counterexample`, `trace_path`, `summary`) with the same
`PASS → proven`, `FAIL → counterexample`, inconclusive → `proven=False` with an
`UNKNOWN` verdict and no counterexample, tool-failure → `RuntimeError` contract as
the equivalence runner.

---

## 5. Limitations

1. **Bounded depth (BMC).** BMC proves a property only up to `depth` cycles. It
   is *complete* when the design reaches a stationary state within that depth (the
   precision monitor's saturating accumulator); otherwise unbounded proof needs
   k-induction with an inductive invariant.
2. **k-induction needs an invariant.** k-induction is unbounded but only converges
   with an inductive invariant. The precision monitor ships the strengthening
   lemma it needs; the equivalence miter needs a reachable-state invariant, which
   is supplied by exposing the internal state as whitebox taps (yosys 0.33 cannot
   reference an instance's internals directly). BMC stays the equivalence default
   because the taps require instrumenting the modules.
3. **Interface-compatible modules.** DUT and reference must share the same I/O
   ports (parameters may differ per instance).
4. **Toolchain required.** Proofs need `sby` + `yosys` + a solver; without them
   `formal_tools_available()` is `False` and the proof functions raise.
5. **SMT tractability.** Wide-multiplier datapaths bound the practical depth (BMC)
   and the practical width (k-induction) on general-purpose SMT engines; the LIF
   miter proves unbounded at a narrow 4-bit datapath and slows sharply as the
   fixed-point multiplier widens.

---

*© 2020–2026 Miroslav Šotek / ANULUM. AGPL-3.0-or-later.*
