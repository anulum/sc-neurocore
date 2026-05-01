<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->

# Safety Certification Guide

SC-NeuroCore provides an automated certification pipeline for safety-critical
neuromorphic deployments. This guide covers the features used for DO-254
Level A (avionics), IEC 61508 (industrial), and ISO 26262 ASIL-D (automotive)
certification.

## Certification Feature Map

| Feature | § | DO-254 | IEC 61508 | ISO 26262 |
|---------|---|:------:|:---------:|:---------:|
| Compliance Matrix | 29 | ✅ | ✅ | ✅ |
| Provenance Chain | 28 | ✅ | ✅ | ✅ |
| Fault Tree (FTA) | 50 | ✅ | ✅ | ✅ |
| Reliability (MTTF) | 49 | ✅ | ✅ | ✅ |
| SEU/TMR Hardening | 7 | ✅ | — | ✅ |
| Formal Equivalence | 26 | ✅ | ✅ | — |
| ODE Stability | 43 | ✅ | ✅ | ✅ |
| Auto-Testbench | 51 | ✅ | ✅ | ✅ |
| Side-Channel Lint | 31 | — | — | ✅ |
| CDC Analysis | 52 | ✅ | ✅ | ✅ |

## Complete Certification Pipeline

### Step 1: Generate Compliance Matrix

```python
from sc_neurocore.compiler.advanced_features import generate_compliance_matrix

matrix = generate_compliance_matrix(
    standard="DO-254",
    equations={"v": "-(v - v_rest) / tau + I"},
    profile_name="bae_rad750_sq",
)
# Matrix maps SC-NeuroCore features to DO-254 objectives
for req in matrix.requirements:
    print(f"  {req['id']}: {req['description']} — {req['status']}")
```

### Step 2: Verify ODE Stability

```python
from sc_neurocore.compiler.advanced_features import verify_ode_stability

result = verify_ode_stability(
    equations={"v": "-(v - v_rest) / tau + I"},
    dt=0.1,
    time_constants={"v": 10.0},
)
assert result.stable, f"UNSTABLE: critical dt = {result.critical_dt}"
```

### Step 3: Generate Fault Tree

```python
from sc_neurocore.compiler.advanced_features import generate_fault_tree

ft = generate_fault_tree("sc_lif", {"v": "-(v)/tau + I"})
print(f"Top event: {ft.top_event}")
print(f"Basic events: {len(ft.basic_events)}")
print(f"Minimal cut sets: {len(ft.mcs)}")

# Each basic event has failure rate for quantitative FTA
for e in ft.basic_events:
    print(f"  {e['id']}: λ = {e['rate']:.0e} /hr — {e['description']}")
```

### Step 4: Predict Reliability

```python
from sc_neurocore.compiler.advanced_features import predict_reliability

r = predict_reliability(
    voltage_v=0.9,
    temperature_c=85.0,  # worst-case junction temp
    node_nm=28,
)
print(f"MTTF: {r.mttf_years:.1f} years")
print(f"Dominant failure: {r.failure_mode}")
```

### Step 5: Generate Provenance Chain

```python
from sc_neurocore.compiler.advanced_features import generate_provenance_chain

chain = generate_provenance_chain(
    equations={"v": "-(v)/tau + I"},
    profile_name="bae_rad750_sq",
    author="avionics_team",
)
print(f"Chain hash: {chain.chain_hash}")
# Hash is reproducible — same inputs produce same hash
```

### Step 6: CDC Analysis

```python
from sc_neurocore.compiler.advanced_features import analyze_cdc

report = analyze_cdc(
    {"v": "-(v)/tau + I", "w": "a*(b*v - w)"},
    clock_domains={"v": "clk_100mhz", "w": "clk_10mhz"},
)
assert report.safe, f"CDC violations: {report.violations}"
```

### Step 7: Generate Testbench

```python
from sc_neurocore.compiler.advanced_features import generate_testbench

tb = generate_testbench(
    "sc_lif", {"v": "-(v)/tau + I"},
    framework="cocotb", num_cycles=10000,
)
with open("test_sc_lif_cert.py", "w") as f:
    f.write(tb)
```

### Step 8: Formal Equivalence

```python
from sc_neurocore.compiler.advanced_features import generate_equivalence_sketch

sketch = generate_equivalence_sketch(
    equations={"v": "-(v - v_rest) / tau + I"},
    data_width=16, fraction=8,
)
with open("sc_lif_equiv.sv", "w") as f:
    f.write(sketch)
```

### Step 9: Generate Full Report

```python
from sc_neurocore.compiler.advanced_features import generate_compilation_report

report = generate_compilation_report(
    "sc_lif", {"v": "-(v)/tau + I"}, "bae_rad750_sq",
    include_carbon=True, include_reliability=True,
)
with open("certification_report.md", "w") as f:
    f.write(report)
```

## Space-Qualified Workflow

For space-grade deployments, combine the certification pipeline with:

1. **SEU hardening** (§7) — Triple Modular Redundancy
2. **Supply chain risk** (§36) — ITAR/sole-source analysis
3. **Thermal envelope** (§40) — junction temp under vacuum/radiation
4. **License compliance** (§56) — export control verification

```python
from sc_neurocore.compiler.advanced_features import (
    score_supply_chain_risk, estimate_thermal_envelope,
    check_license_compliance,
)

# Supply chain
risk = score_supply_chain_risk("bae_rad750_sq")

# Thermal (in vacuum — higher theta_ja)
thermal = estimate_thermal_envelope(
    power_mw=2000, theta_ja=40.0, t_ambient=60.0,
)
assert thermal.pass_fail == "PASS"

# License (export control)
lic = check_license_compliance("proprietary", {
    "sc_neurocore": "AGPL-3.0",
})
```

## Further Reading

- [Compiler Intelligence Guide](compiler_intelligence.md) — all 67 features
- [Static Analysis Guide](static_analysis_guide.md) — guard bits, SVA
- [Deployment Guide](deployment_guide.md) — constraints, bitstream
- [Frontier Platforms Guide](frontier_platforms.md) — space-qualified profiles
