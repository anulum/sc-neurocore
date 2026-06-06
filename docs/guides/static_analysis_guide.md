<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->

# Static Analysis Guide

SC-NeuroCore provides compile-time static analysis for guard bits, overflow
proofs, Q-format envelope proofs, formal assertions, pipeline depth, and power
estimation. These tools operate before deployment so unsafe fixed-point
contracts can fail closed without relying on simulation luck.

## 1. Guard-Bit Auto-Computation

When summing multiple values in fixed-point arithmetic, intermediate
accumulators can overflow silently. Guard bits prevent this.

### Theory

For N additions, the accumulator needs `⌈log₂(N+1)⌉` extra MSBs:

| Additions | Terms | Guard Bits | Example |
|:---------:|:-----:|:----------:|---------|
| 0 | 1 | 0 | `a * b` |
| 1 | 2 | 1 | `a + b` |
| 3 | 4 | 2 | `a + b + c + d` |
| 7 | 8 | 3 | 8-term sum |
| 15 | 16 | 4 | 16-term sum |

### API

```python
from sc_neurocore.compiler.static_analysis import (
    compute_guard_bits,
    compute_guard_bits_multi,
)

# Single expression
bits = compute_guard_bits("-(v - v_rest) / tau_m + R * I / C")
print(f"Guard bits needed: {bits}")  # → 2

# Multi-ODE system (e.g. Izhikevich)
guard = compute_guard_bits_multi({
    "v": "0.04 * v * v + 5 * v + 140 - u + I",
    "u": "a * (b * v - u)",
})
print(f"v needs {guard['v']} guard bits")
print(f"u needs {guard['u']} guard bits")
```

### How It Works

The function parses the expression into a Python AST and counts `Add` and
`Sub` nodes. The formula `⌈log₂(count + 1)⌉` gives the minimum guard bits.

---

## 2. Formal Overflow Proof (Interval Arithmetic)

Instead of simulating millions of cycles and hoping overflow doesn't occur,
you can **prove** it mathematically.

### Theory

Interval arithmetic propagates value bounds through every operation:

```
[a, b] + [c, d] = [a+c, b+d]
[a, b] − [c, d] = [a−d, b−c]
[a, b] × [c, d] = [min(ac,ad,bc,bd), max(ac,ad,bc,bd)]
[a, b] / [c, d] = [a,b] × [1/d, 1/c]  (if 0 ∉ [c,d])
```

If the output interval fits within the Q-format range, overflow is
**provably impossible**.

### API

```python
from sc_neurocore.compiler.static_analysis import prove_no_overflow

result = prove_no_overflow(
    "-(v - v_rest) / tau_m + R * I / C",
    bounds={
        "v": (-128, 127),       # Q8.8 range
        "v_rest": (-65, -65),   # Constant
        "tau_m": (10, 10),      # Constant
        "R": (1, 1),            # Constant
        "I": (0, 100),          # Input range
        "C": (1, 1),            # Constant
    },
    data_width=16, fraction=8,
)

if result.proven_safe:
    print("✅ PROVEN: No overflow possible at Q8.8")
    print(f"   Output range: [{result.expr_interval.lo:.2f}, {result.expr_interval.hi:.2f}]")
    print(f"   Q8.8 range:   [{result.q_min:.2f}, {result.q_max:.2f}]")
    print(f"   Margin:       lo={result.margin_lo:.2f}, hi={result.margin_hi:.2f}")
else:
    print("⚠ UNSAFE: Overflow possible!")
    print(f"   Expression can reach [{result.expr_interval.lo:.2f}, {result.expr_interval.hi:.2f}]")
    print(f"   Q8.8 only covers [{result.q_min:.2f}, {result.q_max:.2f}]")
```

### Result Fields

| Field | Type | Description |
|-------|------|-------------|
| `proven_safe` | `bool` | `True` if overflow is provably impossible |
| `expr_interval` | `Interval` | Output value range `[lo, hi]` |
| `q_min` | `float` | Minimum representable Q-format value |
| `q_max` | `float` | Maximum representable Q-format value |
| `margin_lo` | `float` | Distance from output min to Q-format min (positive = safe) |
| `margin_hi` | `float` | Distance from output max to Q-format max (positive = safe) |

### Limitations

- Division by an interval containing zero returns `(-∞, +∞)` — proof fails
  (conservatively safe).
- Interval arithmetic is **sound but not complete**: it may report "unsafe"
  for expressions that are actually safe (over-approximation).
- Does not handle non-linear correlations between variables.

---

## 3. Fixed-Point Envelope Width Proofs

Mixed Q8.8/Q16.16 and block-floating dense layers compute conservative
absolute Q-code bounds per output lane.  A realised dot product can be small
because positive and negative terms cancel, but the hardware accumulator still
needs enough width for the absolute product envelope.  `prove_fixed_point_envelope`
turns those bounds into a static, machine-checkable Q-format safety proof.

### API

```python
from sc_neurocore.compiler.static_analysis import prove_fixed_point_envelope

proof = prove_fixed_point_envelope(
    [531_400],          # conservative absolute Q16.16 bound codes
    total_bits=32,
    fractional_bits=16,
)

assert proof.static_overflow_proven_safe
assert proof.required_total_bits == 21
assert proof.required_integer_bits == 5
assert proof.width_headroom_bits == 11
```

For a saturating mixed-dense probe:

```python
probe = prove_fixed_point_envelope(
    [17_454_214_414_336],
    total_bits=32,
    fractional_bits=16,
)

assert probe.saturation_required
assert probe.required_total_bits == 45
assert probe.required_integer_bits == 29
assert probe.width_headroom_bits == -13
```

### Result Fields

| Field | Type | Description |
|-------|------|-------------|
| `proof_kind` | `str` | `signed_symmetric_fixed_point_width` for signed Q formats |
| `output_format` | `str` | Human-readable Q format, e.g. `Q16.16` |
| `conservative_safe_bound_code` | `int` | Largest signed or unsigned code representable without saturation |
| `max_abs_bound_code` | `int` | Largest conservative absolute output bound supplied to the proof |
| `min_headroom_code` | `int` | Remaining code headroom; negative means saturation is required |
| `required_total_bits` | `int` | Total width needed by the conservative bound, including sign when signed |
| `required_integer_bits` | `int` | Integer-side Q-format width needed after fractional bits are reserved |
| `width_headroom_bits` | `int` | `total_bits - required_total_bits` |
| `saturation_required` | `bool` | True when the conservative bound exceeds the target format |
| `static_overflow_proven_safe` | `bool` | True when no saturation is required by the static bound |

The `manifest()` method returns the same fields as plain JSON-compatible
values.  Benchmark evidence gates use those manifest fields to lock Python and
Rust Q16.16 proof parity.  `PrecisionEnvelopeReport` in the quantizer delegates
its proof fields to this API, so dense mixed-precision and block-floating
deployment manifests cannot drift from the standalone static-analysis proof.

### Safety Boundary

- Signed proofs use absolute magnitudes and reserve one sign bit.
- Unsigned proofs reject negative Q-code inputs.
- Empty bound vectors, invalid widths, and non-integer Q-code values are
  rejected before any manifest is produced.
- The proof is conservative by design: it does not infer safety from
  cancellation in the realised output value.

---

## 4. SystemVerilog Assertion (SVA) Generation

For safety-critical designs (DO-254, IEC 61508), simulation coverage is
not sufficient — you need formal verification properties.

### API

```python
from sc_neurocore.compiler.static_analysis import generate_sva

sva = generate_sva(
    state_vars=["v", "u"],
    data_width=16,
    fraction=8,
    module_name="sc_izh",
    input_bounds={"I_t": (-1000, 25600)},
)

# Write to file
with open("sc_izh_sva.sv", "w") as f:
    f.write(sva)
```

### Generated Properties

The SVA module contains four categories of formal properties:

#### Overflow Assertions
```systemverilog
a_no_overflow_v: assert property (
    disable iff (!rst_n)
    $signed(v_reg) >= 16'sd-32768 && $signed(v_reg) <= 16'sd32767
) else $error("OVERFLOW: v_reg = %0d", v_reg);
```

#### Reachability Covers
```systemverilog
c_spike_reachable: cover property (
    disable iff (!rst_n) spike_out == 1'b1
);
```

Proves that the spike output **can** be reached — a basic sanity check.

#### Input Assumptions
```systemverilog
m_I_t_bound: assume property (
    disable iff (!rst_n)
    $signed(I_t) >= 16'sd-1000 && $signed(I_t) <= 16'sd25600
);
```

Constrains formal verification tools to only explore valid input ranges.

#### Stability Checks
```systemverilog
a_v_not_stuck_max: assert property (
    disable iff (!rst_n)
    not (v_reg == 16'sd32767 [*100])
) else $warning("v_reg stuck at max for 100+ cycles");
```

Detects latch-up conditions where a state variable saturates permanently.

### Binding

The generated SVA module is bound to the DUT in the testbench:

```systemverilog
bind sc_izh sc_izh_sva sva_inst (.*);
```

### Tool Compatibility

| Tool | Support |
|------|---------|
| Xilinx Vivado Formal | Full |
| Synopsys VC Formal | Full |
| Cadence JasperGold | Full |
| Mentor Questa Formal | Full |
| SymbiYosys (open-source) | Partial (assert/cover only) |

---

## Cross-References

- [Hardware Profiles Guide](hardware_profiles.md) — 65 platform profiles
- [Precision Modes Guide](precision_modes.md) — 11 Q-format modes
- [Co-Simulation Guide](cosimulation_guide.md) — Python↔Verilog verification
- [SoC Integration Guide](soc_integration_guide.md) — bus wrappers + drivers
- [Deployment Guide](deployment_guide.md) — constraints, TCL, bitstream
