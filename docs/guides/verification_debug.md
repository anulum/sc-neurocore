<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->

# Verification & Debug Guide

SC-NeuroCore provides a complete verification stack from pre-silicon formal
analysis through post-silicon debug. This guide covers the **14 verification,
security, and debug features** available in the compiler.

## Verification Feature Map

| Stage | Feature | § | Tool |
|-------|---------|---|------|
| Pre-RTL | ODE Stability | 43 | Eigenvalue analysis |
| Pre-RTL | Model Complexity | 38 | Compute/memory/comm classifier |
| RTL | Formal Equivalence | 26 | SAT-based sketch |
| RTL | CDC Analysis | 52 | Clock domain crossing |
| RTL | Side-Channel Lint | 31 | Timing/power leakage |
| RTL | Auto-Testbench | 51 | Cocotb / UVM |
| RTL | Bit-True Kernel | 37 | C/Rust golden model |
| RTL | **Hardware Trojan Lint** | 60 | Dormant trigger detection |
| Post-Si | Debug Probes | 46 | ILA / SignalTap |
| Post-Si | **HIL Calibration** | 62 | Drift compensation protocol |
| Post-Si | **Digital Twin** | 63 | Software shadow monitor |
| Space | **SEU Scrub Scheduler** | 65 | Configuration scrubbing |
| QA | Regression Watchdog | 55 | Metric delta detection |
| Report | Compilation Report | 59 | Markdown summary |

---

## Pre-RTL Verification

### ODE Stability Check (§43)

**Run this FIRST** — before generating any RTL. Verifies that the
discretized ODE system is numerically stable at the chosen timestep.

```python
from sc_neurocore.compiler.intelligence import verify_ode_stability

# Check LIF stability
result = verify_ode_stability(
    equations={"v": "-(v - v_rest) / tau + I"},
    dt=0.1,
    time_constants={"v": 10.0},
)
print(f"Stable: {result.stable}")
print(f"Critical dt: {result.critical_dt} ms")
print(f"Max eigenvalue: {result.max_eigenvalue}")

# Larger timesteps may be unstable
result_bad = verify_ode_stability(
    equations={"v": "-(v)/tau"},
    dt=100.0,
    time_constants={"v": 0.5},
)
assert not result_bad.stable  # Catches divergent hardware
```

**Mathematical basis**: Forward Euler stability requires |1 + λ·dt| < 1
where λ = -1/τ. Critical dt = 2τ.

### Model Complexity Classification (§38)

Determines whether your model is compute-bound, memory-bound, or
communication-bound, and recommends the optimal platform class:

```python
from sc_neurocore.compiler.intelligence import classify_model_complexity

cls = classify_model_complexity(
    equations={"v": "g*m*m*m*h*(E-v) + g*n*n*n*n*(E-v)"},
    num_neurons=100000,
    num_synapses=10000000,
)
print(f"Classification: {cls.classification}")
print(f"Recommended class: {cls.recommended_class}")
```

---

## RTL Verification

### Formal Equivalence Sketch (§26)

Generates SystemVerilog assertions that prove bit-accurate equivalence
between the Python ODE model and the generated Verilog:

```python
from sc_neurocore.compiler.intelligence import generate_equivalence_sketch

sketch = generate_equivalence_sketch(
    equations={"v": "-(v - v_rest) / tau + I"},
    data_width=16, fraction=8,
)
with open("equiv_check.sv", "w") as f:
    f.write(sketch)
# Run with: JasperGold, VC Formal, or EBMC
```

### CDC Analysis (§52)

Identifies clock domain crossings when neuron state variables are in
different clock domains (common in multi-timescale partitioned designs):

```python
from sc_neurocore.compiler.intelligence import analyze_cdc

report = analyze_cdc(
    {"v": "u + I", "u": "v - threshold", "w": "slow_adaptation"},
    clock_domains={"v": "clk_fast", "u": "clk_fast", "w": "clk_slow"},
)
print(f"Total crossings: {report.total_crossings}")
print(f"All safe: {report.safe}")
for c in report.crossings:
    print(f"  {c['signal']}: {c['sync_type']}")
```

### Auto-Testbench Generation (§51)

Generate complete testbenches in Cocotb (Python) or UVM (SystemVerilog):

```python
from sc_neurocore.compiler.intelligence import generate_testbench

# Cocotb — preferred for rapid iteration
tb = generate_testbench(
    "sc_lif", {"v": "expr", "u": "expr"},
    framework="cocotb", num_cycles=10000,
)
# Generates test_sc_lif_reset() and test_sc_lif_run()
with open("test_sc_lif.py", "w") as f:
    f.write(tb)

# UVM — for ASIC tapeout flows
tb_uvm = generate_testbench(
    "sc_lif", {"v": "expr"}, framework="uvm",
)
with open("sc_lif_uvm_test.sv", "w") as f:
    f.write(tb_uvm)
```

### Bit-True Simulation Kernel (§37)

Generate C or Rust code that produces identical output to the Verilog RTL:

```python
from sc_neurocore.compiler.intelligence import generate_bittrue_kernel

c_code = generate_bittrue_kernel(
    equations={"v": "-(v - v_rest) / tau + I"},
    data_width=16, fraction=8, language="c",
)
with open("sc_lif_bittrue.c", "w") as f:
    f.write(c_code)
# Compile: gcc -O2 sc_lif_bittrue.c -o bittrue_sim
```

### Side-Channel Lint (§31)

Detect timing and power side-channel vulnerabilities:

```python
from sc_neurocore.compiler.intelligence import lint_side_channels

result = lint_side_channels(
    equations={"v": "-(v) / tau + I * secret_weight"},
)
for finding in result.findings:
    print(f"  ⚠️ {finding}")
```

### Hardware Trojan Lint (§60)

Detect suspicious combinational paths that could hide hardware trojans:

```python
from sc_neurocore.compiler.intelligence import lint_hardware_trojans

result = lint_hardware_trojans(
    {"v": "-(v)/tau + I", "u": "a*(b*v - u)"},
)
print(f"Risk: {result.risk_level}")  # LOW / MEDIUM / HIGH
print(f"Suspicious paths: {len(result.suspicious_paths)}")
print(f"Total checks: {result.total_checks}")
```

---

## Post-Silicon Debug

### Debug Probe Insertion (§46)

Auto-insert ILA (Xilinx Vivado) or SignalTap (Intel Quartus) probes:

```python
from sc_neurocore.compiler.intelligence import insert_debug_probes

# Xilinx ILA
probes = insert_debug_probes(
    "sc_lif", {"v": "expr", "u": "expr"},
    vendor="xilinx", depth=4096,
)
with open("insert_ila.tcl", "w") as f:
    f.write(probes.tcl_commands)
# Source in Vivado: source insert_ila.tcl

# Intel SignalTap
probes_intel = insert_debug_probes(
    "sc_lif", {"v": "expr"}, vendor="intel", depth=2048,
)
```

### HIL Calibration Protocol (§62)

Generate a step-by-step hardware-in-the-loop calibration procedure
for compensating analog drift, mismatch, and process variation:

```python
from sc_neurocore.compiler.intelligence import generate_hil_calibration

cal = generate_hil_calibration(
    "sc_lif", {"v": "expr", "u": "expr"},
    parameters={"tau": (5.0, 50.0), "threshold": (0.5, 2.0)},
)
for step in cal.protocol_steps:
    print(step)
```

### Digital Twin Shadow (§63)

Generate a software shadow model that mirrors deployed hardware state
for runtime anomaly detection:

```python
from sc_neurocore.compiler.intelligence import generate_digital_twin

twin = generate_digital_twin("sc_lif", {"v": "-(v)/tau"}, "artix7")
with open("sc_lif_twin.py", "w") as f:
    f.write(twin)
# twin.step() mirrors hardware, twin.compare() detects drift
```

---

## Space-Grade Verification

### SEU Scrub Scheduler (§65)

Generate scrubbing schedules for space-grade configuration memory:

```python
from sc_neurocore.compiler.intelligence import schedule_seu_scrubbing

schedule = schedule_seu_scrubbing(
    config_bits=1_000_000,
    orbit_altitude_km=800,
    shielding_mm_al=5.0,
    strategy="hybrid",
)
print(f"Scrub interval: {schedule.interval_ms:.0f} ms")
print(f"Frames/cycle: {schedule.frames_per_cycle}")
print(f"Expected SEU rate: {schedule.expected_seu_rate:.4f}/day")
```

---

## Quality Assurance

### Regression Watchdog (§55)

Detect performance regressions between compilation runs:

```python
from sc_neurocore.compiler.intelligence import check_regression

# Save baseline metrics after first compilation
baseline = {"area_luts": 1200, "fmax_mhz": 250, "power_mw": 85}

# After code change, compare
current = {"area_luts": 1350, "fmax_mhz": 240, "power_mw": 90}

checks = check_regression(baseline, current, threshold_pct=5.0)
for c in checks:
    icon = "⚠️" if c.regression else "✅"
    print(f"  {icon} {c.metric}: {c.delta_pct:+.1f}%")
```

### Compilation Report (§59)

Generate a comprehensive markdown report:

```python
from sc_neurocore.compiler.intelligence import generate_compilation_report

report = generate_compilation_report(
    "sc_lif", {"v": "-(v)/tau + I"}, "artix7",
    include_carbon=True,
    include_reliability=True,
)
with open("report.md", "w") as f:
    f.write(report)
```

---

## Complete Verification Flow

```
┌─────────────────────────────────────────────────┐
│  1. verify_ode_stability()          — §43       │
│  2. classify_model_complexity()     — §38       │
│  3. lint_hardware_trojans()         — §60       │
│  4. analyze_cdc()                   — §52       │
│  5. lint_side_channels()            — §31       │
│  6. generate_equivalence_sketch()   — §26       │
│  7. generate_testbench()            — §51       │
│  8. generate_bittrue_kernel()       — §37       │
│  9. insert_debug_probes()           — §46       │
│ 10. generate_hil_calibration()      — §62       │
│ 11. generate_digital_twin()         — §63       │
│ 12. schedule_seu_scrubbing()        — §65       │
│ 13. check_regression()              — §55       │
│ 14. generate_compilation_report()   — §59       │
└─────────────────────────────────────────────────┘
```

## Further Reading

- [Compiler Intelligence Guide](compiler_intelligence.md) — all 67 features
- [Static Analysis Guide](static_analysis_guide.md) — guard bits, SVA
- [Co-Simulation Guide](cosimulation_guide.md) — Python↔Verilog
- [Safety Certification Guide](safety_certification.md) — DO-254/ISO 26262
- [Multi-Target Deployment Guide](multi_target_deployment.md) — heterogeneous deploy
- [Deployment Guide](deployment_guide.md) — constraints, bitstream
