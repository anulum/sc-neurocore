# Formal Proofs — Lean 4 Safety Bounds

Formal verification of the six safety invariants that SC-NeuroCore's
hardware monitor (``safety_monitor.sv``) enforces at runtime. Each
theorem in ``safety_bounds.lean`` maps 1:1 to a property in the
SystemVerilog monitor. The :class:`FormalProofEngine` Python façade
shells out to Lean 4 to type-check the proofs.

```python
from sc_neurocore.formal.lean_bridge import FormalProofEngine

engine = FormalProofEngine()
if engine.is_available():
    assert engine.check_proofs(), "Lean proofs must type-check"
```

---

## 1. `FormalProofEngine`

| Method                   | Purpose                                                                                  |
| ------------------------ | ---------------------------------------------------------------------------------------- |
| `__init__()`             | Locates ``lean`` on ``PATH`` via ``shutil.which``; resolves the bundled proof file path. |
| `is_available() -> bool` | True iff Lean 4 is callable **and** ``safety_bounds.lean`` exists in the package.        |
| `check_proofs() -> bool` | Runs ``lean <proof_file>``. Returns ``True`` when the file type-checks with no errors.   |

The engine treats any occurrence of ``"error"`` (case-insensitive) in
the Lean output as a failure — matching how Lean 4 reports elaboration
errors.

---

## 2. Theorem inventory — `safety_bounds.lean`

The Lean 4 file (`src/sc_neurocore/formal/proofs/safety_bounds.lean`,
133 lines) proves the following. Each theorem is referenced by the
matching P-property in `safety_monitor.sv`:

| # | Theorem                     | Statement                                               | Monitor property |
| - | --------------------------- | ------------------------------------------------------- | ---------------- |
| 1 | `monitor_soundness`         | `halt = false ↔ coherence ≥ limit`                      | P1               |
| 2 | `safe_transition`           | Monotone coherence transitions preserve safety          | P2               |
| 3 | `sc_precision_bound`        | Stochastic bitstream variance proxy: `4·k·(N−k) ≤ N²`   | P3               |
| 4 | `sc_add_preserves_range`    | SC addition result never exceeds the denominator        | P4               |
| 5 | `lif_membrane_bounded`      | LIF membrane potential stays `≤ v_max` by construction  | P5               |
| 6 | `correlation_range`         | SCC numerator magnitude never exceeds the denominator   | P6               |

`monitor_soundness` is a biconditional — the halt signal is exactly the
negation of the coherence invariant. `safe_transition` captures the
staircase property: coherence can only climb, so once safe, always
safe. The three SC-specific theorems (`sc_precision_bound`,
`sc_add_preserves_range`, `correlation_range`) bound stochastic
arithmetic; the neuron theorem (`lif_membrane_bounded`) anchors the
neuron dynamics.

---

## 3. Hardware side — `neuro_safe_monitor`

The matching SystemVerilog monitor lives at
`src/sc_neurocore/hdl_gen/safety/safety_monitor.sv`. It is parameter­ised
on the same bounds the Lean proofs assume:

| Parameter          | Default            | Meaning                                          |
| ------------------ | ------------------ | ------------------------------------------------ |
| `MAX_CURRENT`      | `16'h7FFF`         | Q8.8 current cap                                 |
| `MAX_VOLTAGE`      | `16'hC000`         | Q8.8 voltage cap                                 |
| `COHERENCE_LIMIT`  | `16'h0100`         | Monitor's safety-mode floor                      |
| `SC_DENOM`         | `16'h0100` (= 256) | Bitstream length `N` in fixed-point              |
| `LIF_V_MAX`        | `16'hC000`         | Upper bound for LIF membrane                     |

A Yosys-plus-OpenROAD synthesis script
(`src/sc_neurocore/hdl_gen/openroad_flow/run_asic_flow.sh`) pushes the
monitor through gate-level synthesis with area / timing reports.

---

## 4. Toolchain expectations

- Lean 4 (`lean` on `PATH`). Ubuntu: `elan install` from
  <https://leanprover.github.io/>.
- The engine prints a clear diagnostic when Lean is missing — it never
  raises on construction, so imports remain cheap.

---

## 5. Limitations

- The proofs target *stated* invariants, not the full monitor RTL.
  Equivalence of the Lean model to the SystemVerilog implementation is
  checked by running both against the same Q8.8 traces in
  ``tests/test_safety/``.
- Lean elaboration is slow (seconds, not milliseconds) — don't gate
  unit tests on `check_proofs()`; run it in a dedicated CI step.

---

## Reference

- Python bridge: `src/sc_neurocore/formal/lean_bridge.py`.
- Lean 4 proofs: `src/sc_neurocore/formal/proofs/safety_bounds.lean`.
- Hardware monitor: `src/sc_neurocore/hdl_gen/safety/safety_monitor.sv`.
- ASIC flow: `src/sc_neurocore/hdl_gen/openroad_flow/run_asic_flow.sh`.

::: sc_neurocore.formal.lean_bridge
    options:
      show_root_heading: true
