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
k-induction (`mode="prove"`) is available but is **not** the default: for
datapaths with wide signed multipliers (the fixed-point neuron update) the
induction step reports spurious counterexamples from unreachable mid-states
unless the reachable-state invariant is supplied, so a bounded proof to a
solver-tractable depth is the honest default. On the fixed-point LIF datapath
`z3` proves the miter quickly to depth ≈ 4 and slows sharply beyond that.

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

> **Toolchain note.** yosys 0.33 silently ignores SystemVerilog `bind`
> directives, so the checker is instantiated explicitly inside the monitor under
> `` `ifdef FORMAL `` (a macro `read -formal` defines) rather than bound in. Plain
> synthesis strips it; formal builds elaborate it.

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
manifest = write_precision_formal_evidence_bundle(out_dir, assignments, execute=True)
claim = manifest["formal_claim"]
print(claim["symbiyosys_executed"], claim["formal_proof_passed"], claim["proof_verdict"])
```

`prove_property(rtl_verilog, sva_verilog, *, top, mode="bmc", depth=..., ...)`
returns a `PropertyProofResult` (`proven`, `verdict`, `mode`, `depth`, `engine`,
`returncode`, `counterexample`, `trace_path`, `summary`) with the same
`PASS → proven`, `FAIL → counterexample`, tool-failure → `RuntimeError` contract
as the equivalence runner.

---

## 5. Limitations

1. **Bounded depth.** BMC proves equivalence only up to `depth` cycles; deeper
   proof needs more solver time or an invariant for k-induction.
2. **Interface-compatible modules.** DUT and reference must share the same I/O
   ports (parameters may differ per instance).
3. **Toolchain required.** Proofs need `sby` + `yosys`; without them
   `formal_tools_available()` is `False` and the proof functions raise.
4. **SMT tractability.** Wide-multiplier datapaths bound the practical depth on
   general-purpose SMT engines.

---

*© 2020–2026 Miroslav Šotek / ANULUM. AGPL-3.0-or-later.*
