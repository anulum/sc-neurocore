<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->

# Compiler Intelligence Guide

SC-NeuroCore includes **59 compiler intelligence features** that transform the
compiler from a passive code generator into an active decision-making system.
This guide covers the features added in Waves 6–9 (§26–§59) — capabilities
that **no other neuromorphic compiler in the world provides**.

## Table of Contents

1. [Wave 6 — The Moat (§26–§33)](#wave-6--the-moat-2633)
2. [Wave 7 — The Brain (§34–§41)](#wave-7--the-brain-3441)
3. [Wave 8 — Full-Stack Intelligence (§42–§51)](#wave-8--full-stack-intelligence-4251)
4. [Wave 9 — Extensibility Layer (§52–§59)](#wave-9--extensibility-layer-5259)
5. [Quick Reference Table](#quick-reference-table)
6. [Usage Patterns](#usage-patterns)

---

## Wave 6 — The Moat (§26–§33)

### §26. Formal Equivalence Sketch — `generate_equivalence_sketch()`

Generates a formal equivalence checking sketch (SAT-based) to verify that
the compiled Verilog exactly matches the original ODE semantics within
fixed-point tolerance bounds.

```python
from sc_neurocore.compiler.advanced_features import generate_equivalence_sketch

sketch = generate_equivalence_sketch(
    equations={"v": "-(v - v_rest) / tau + I"},
    data_width=16, fraction=8,
)
# sketch contains SystemVerilog assertions + SAT constraints
print(sketch[:200])
```

**Use case**: Post-compilation formal verification. Proves bit-accurate
equivalence between Python model and generated RTL.

---

### §27. Multi-Timescale Partitioner — `partition_timescales()`

Separates fast (membrane dynamics, ~1ms) from slow (synaptic plasticity,
~100ms) variables, enabling heterogeneous clock domains and power savings.

```python
from sc_neurocore.compiler.advanced_features import partition_timescales

result = partition_timescales(
    equations={"v": "fast dynamics", "w": "slow adaptation"},
    time_constants={"v": 1.0, "w": 100.0},
    threshold_ratio=10.0,
)
print(result.fast_partition)  # ['v']
print(result.slow_partition)  # ['w']
```

**Use case**: Power reduction via clock gating slow variables.

---

### §28. Provenance Chain — `generate_provenance_chain()`

Creates a cryptographic audit trail from model definition through compilation
to deployment, enabling full traceability for safety-critical applications.

```python
from sc_neurocore.compiler.advanced_features import generate_provenance_chain

chain = generate_provenance_chain(
    equations={"v": "-(v)/tau + I"},
    profile_name="artix7",
    author="lab_name",
)
print(chain.chain_hash)  # SHA-256 of full pipeline
print(chain.stages)      # list of provenance records
```

**Use case**: DO-254 audit compliance, reproducibility.

---

### §29. Compliance Matrix — `generate_compliance_matrix()`

Auto-generates a certification compliance matrix mapping SC-NeuroCore features
to regulatory requirements (DO-254, IEC 61508, ISO 26262).

```python
from sc_neurocore.compiler.advanced_features import generate_compliance_matrix

matrix = generate_compliance_matrix(
    standard="DO-254",
    equations={"v": "expr"},
    profile_name="artix7",
)
for req in matrix.requirements[:3]:
    print(f"{req['id']}: {req['status']}")
```

---

### §30. Energy Schedule — `generate_energy_schedule()`

Generates a time-domain power schedule for duty-cycled operation, including
sleep/wake transitions and energy budgeting.

```python
from sc_neurocore.compiler.advanced_features import generate_energy_schedule

sched = generate_energy_schedule(
    power_active_mw=100.0,
    power_sleep_mw=0.1,
    duty_cycle=0.3,
)
print(f"Average: {sched.avg_power_mw} mW")
```

---

### §31. Side-Channel Lint — `lint_side_channels()`

Detects potential side-channel leakage (timing, power) in neuron
implementations that could compromise model IP or cryptographic parameters.

```python
from sc_neurocore.compiler.advanced_features import lint_side_channels

warnings = lint_side_channels(
    equations={"v": "-(v - v_rest) / tau + I * weight"},
)
for w in warnings.findings:
    print(w)  # e.g., "data-dependent branch on 'weight'"
```

---

### §32. Drift Compensation — `generate_drift_compensator()`

Generates analog drift compensation logic for memristive and ferroelectric
devices where synaptic weights degrade over time.

```python
from sc_neurocore.compiler.advanced_features import generate_drift_compensator

verilog = generate_drift_compensator(
    num_weights=256,
    drift_model="logarithmic",
)
print(verilog[:200])  # Verilog refresh controller
```

---

### §33. Heterogeneous Dispatch — `plan_heterogeneous_dispatch()`

Plans multi-target compilation where different neuron populations execute on
different hardware platforms simultaneously (e.g., sensory on FPGA, cortical
on neuromorphic, motor on MCU).

```python
from sc_neurocore.compiler.advanced_features import plan_heterogeneous_dispatch

plan = plan_heterogeneous_dispatch(
    populations={"sensory": "artix7", "cortical": "loihi2", "motor": "rp2040"},
    connections=[("sensory", "cortical"), ("cortical", "motor")],
)
for p, target in plan.assignments.items():
    print(f"{p} → {target}")
```

---

## Wave 7 — The Brain (§34–§41)

### §34. Auto-Target Recommender — `recommend_target()`

Given a set of constraints (power budget, latency, cost), automatically
recommends the optimal hardware platform from all 175 profiles.

```python
from sc_neurocore.compiler.advanced_features import recommend_target

recs = recommend_target(
    constraints={"max_power_mw": 500, "min_freq_mhz": 100, "max_width": 16},
    top_k=5,
)
for r in recs:
    print(f"{r['name']} (score: {r['score']:.2f})")
```

**Use case**: "Just tell me what chip to use for my model."

---

### §35. Partial Reconfiguration Planner — `plan_partial_reconfiguration()`

Plans FPGA Dynamic Partial Reconfiguration (DPR) to time-multiplex different
SNN layers on the same fabric, reducing area by 2-4×.

```python
from sc_neurocore.compiler.advanced_features import plan_partial_reconfiguration

plan = plan_partial_reconfiguration(
    regions={"region_a": 500, "region_b": 300},
    total_luts=20000,
)
print(plan.schedule)
```

---

### §36. Supply Chain Risk Scorer — `score_supply_chain_risk()`

Profiles geopolitical, sole-source, and ITAR risk for each hardware target,
enabling informed procurement decisions.

```python
from sc_neurocore.compiler.advanced_features import score_supply_chain_risk

risk = score_supply_chain_risk("loihi2")
print(f"Risk: {risk.overall_risk}")
print(f"Factors: {risk.risk_factors}")
```

---

### §37. Bit-True Simulation Kernel — `generate_bittrue_kernel()`

Generates C or Rust source code that produces bit-identical output to the
Verilog RTL, enabling software-in-the-loop (SIL) verification.

```python
from sc_neurocore.compiler.advanced_features import generate_bittrue_kernel

code = generate_bittrue_kernel(
    equations={"v": "-(v - v_rest) / tau + I"},
    data_width=16, fraction=8,
    language="c",
)
print(code[:300])  # C source with fixed-point arithmetic
```

---

### §38. Model Complexity Classifier — `classify_model_complexity()`

Classifies a model as memory-bound, compute-bound, or communication-bound,
then routes to the optimal paradigm (PIM, FPGA, or optical).

```python
from sc_neurocore.compiler.advanced_features import classify_model_complexity

cls = classify_model_complexity(
    equations={"v": "a*b + c*d", "u": "e*f"},
    num_neurons=10000,
    num_synapses=1000000,
)
print(cls.classification)     # "memory_bound"
print(cls.recommended_class)  # "in_memory"
```

---

### §39. Cross-Compilation Cache — `CompilationCache`

Memoized compilation cache enabling O(1) retrieval of repeated compilations.

```python
from sc_neurocore.compiler.advanced_features import CompilationCache

cache = CompilationCache()
cache.store("lif_artix7", {"verilog": "...", "timing": "..."})
hit = cache.lookup("lif_artix7")
print(hit is not None)  # True
```

---

### §40. Thermal Envelope Estimator — `estimate_thermal_envelope()`

Predicts junction temperature from power dissipation using thermal resistance
models, catching thermal violations at compile time.

```python
from sc_neurocore.compiler.advanced_features import estimate_thermal_envelope

t = estimate_thermal_envelope(power_mw=500, theta_ja=25.0, t_ambient=40.0)
print(f"T_j = {t.t_junction}°C, Margin = {t.thermal_margin}°C")
print(f"Status: {t.pass_fail}")
```

---

### §41. Network Topology Optimizer — `optimize_network_topology()`

Minimizes inter-chip spike bandwidth in multi-chip SNN deployments using
graph partitioning.

```python
from sc_neurocore.compiler.advanced_features import optimize_network_topology

result = optimize_network_topology(
    adjacency={"a": ["b", "c"], "b": ["c"], "c": ["a"]},
    num_chips=2,
)
print(f"Bandwidth reduction: {result.bandwidth_reduction:.1%}")
```

---

## Wave 8 — Full-Stack Intelligence (§42–§51)

### §42. NIR / ONNX-SNN Import — `import_nir_graph()`

Imports trained SNN models from snnTorch, Norse, Sinabs, or Nengo via the
Neuromorphic Intermediate Representation (NIR) standard, bridging the gap
between ML training frameworks and hardware compilation.

```python
from sc_neurocore.compiler.advanced_features import import_nir_graph

# NIR graph from snnTorch export
nir_data = {
    "nodes": {
        "layer0": {"type": "LIF", "tau": 20.0},
        "layer1": {"type": "LIF", "tau": 10.0},
    },
    "edges": [["layer0", "layer1"]],
}
graph = import_nir_graph(nir_data, framework="snnTorch")
print(graph.equations)  # ODE equations ready for compilation
```

**Use case**: Train in PyTorch/snnTorch → export NIR → compile to any of 165 targets.

---

### §43. ODE Stability Verifier — `verify_ode_stability()`

Verifies that the discretized ODE system is numerically stable using
eigenvalue analysis of the linearized system. Prevents divergent hardware.

```python
from sc_neurocore.compiler.advanced_features import verify_ode_stability

result = verify_ode_stability(
    equations={"v": "-(v)/tau + I", "u": "a*(b*v - u)"},
    dt=0.1,
    time_constants={"v": 10.0, "u": 100.0},
)
print(f"Stable: {result.stable}")
print(f"Critical dt: {result.critical_dt} ms")
print(f"Max eigenvalue: {result.max_eigenvalue}")
```

**Mathematical basis**: Forward Euler stability region |1 + λ·dt| < 1,
where λ = -1/τ for each linearized ODE.

---

### §44. Power Intent Generator — `generate_power_intent()`

Generates IEEE 1801 Unified Power Format (UPF) for multi-voltage neuron
arrays, including power domains, isolation cells, and retention strategies.

```python
from sc_neurocore.compiler.advanced_features import generate_power_intent

upf = generate_power_intent("sc_lif_array", num_domains=4, always_on=True)
# Write to file for use in Synopsys/Cadence flow
with open("sc_lif_array.upf", "w") as f:
    f.write(upf)
```

**Output includes**: `create_power_domain`, `set_isolation`, `set_retention`,
`add_power_state` — compatible with all major EDA tools.

---

### §45. Carbon Footprint Estimator — `estimate_carbon_footprint()`

Estimates lifecycle CO₂ emissions (manufacturing + operation) per compilation
target, enabling ESG-compliant hardware selection.

```python
from sc_neurocore.compiler.advanced_features import estimate_carbon_footprint

# Compare carbon impact across targets
for target in ["artix7", "loihi2", "finalspark_neuroplatform"]:
    c = estimate_carbon_footprint(target, power_mw=100)
    print(f"{target}: {c.total_5yr_kg_co2:.1f} kg CO₂ (5yr)")
```

**Regulatory context**: EU carbon labelling becomes mandatory in 2027.
SC-NeuroCore is the only neuromorphic compiler that generates this data.

---

### §46. Debug Probe Inserter — `insert_debug_probes()`

Auto-inserts ILA (Xilinx) or SignalTap (Intel) debug probes for post-silicon
debugging, with vendor-specific TCL commands.

```python
from sc_neurocore.compiler.advanced_features import insert_debug_probes

probes = insert_debug_probes(
    "sc_lif", {"v": "expr", "u": "expr"},
    vendor="xilinx", depth=4096,
)
print(probes.signals)       # ['v', 'u', 'spike_out', 'clk', 'rst_n']
print(probes.tcl_commands)  # Vivado TCL script
```

---

### §47. Memory Map Generator — `generate_memory_map()`

Generates address decoders for multi-neuron SoC arrays, producing both the
address map specification and synthesisable Verilog.

```python
from sc_neurocore.compiler.advanced_features import generate_memory_map

mmap = generate_memory_map(
    "sc_lif_array",
    {"v": "expr", "u": "expr", "I": "expr"},
    num_neurons=1024, data_width=16,
    base_address=0x4000_0000,
)
print(f"Total address space: {mmap.total_bytes} bytes")
print(mmap.decoder_verilog)
```

---

### §48. Model Portability Scorer — `score_portability()`

Scores 0–100 how portable a neuron model is across all 175 profiles,
identifying blockers that limit cross-platform deployment.

```python
from sc_neurocore.compiler.advanced_features import score_portability

# Simple LIF — highly portable
s = score_portability({"v": "-(v - v_rest) / tau + I"})
print(f"Score: {s.score}/100 ({s.compatible_profiles}/{s.total_profiles})")

# Complex Hodgkin-Huxley — limited portability
s = score_portability({"v": "g_na*m*m*m*h*(E_na-v) + g_k*n*n*n*n*(E_k-v)"})
print(f"Score: {s.score}/100, Blockers: {s.blockers}")
```

---

### §49. Aging/Reliability Predictor — `predict_reliability()`

Predicts Mean Time To Failure (MTTF) using Arrhenius temperature acceleration
and voltage stress models.

```python
from sc_neurocore.compiler.advanced_features import predict_reliability

r = predict_reliability(voltage_v=0.9, temperature_c=85, node_nm=7)
print(f"MTTF: {r.mttf_years:.1f} years")
print(f"Failure mode: {r.failure_mode}")  # TDDB, HCI, or NBTI
print(f"Temp acceleration: {r.temp_accel:.1f}×")
```

**Physics**: Uses Ea=0.7eV Arrhenius model with cubic voltage acceleration.

---

### §50. Fault Tree Generator — `generate_fault_tree()`

Auto-generates Fault Tree Analysis (FTA) and FMEA for DO-254 Level A and
ISO 26262 ASIL-D certification.

```python
from sc_neurocore.compiler.advanced_features import generate_fault_tree

ft = generate_fault_tree("sc_lif", {"v": "expr", "u": "expr"})
print(f"Top event: {ft.top_event}")
print(f"Basic events: {len(ft.basic_events)}")
print(f"Minimal cut sets: {len(ft.mcs)}")
for e in ft.basic_events[:3]:
    print(f"  {e['id']}: rate={e['rate']:.0e}")
```

---

### §51. Auto-Testbench Generator — `generate_testbench()`

Generates complete Cocotb or UVM verification testbenches with reset
verification and multi-cycle runtime checks.

```python
from sc_neurocore.compiler.advanced_features import generate_testbench

# Cocotb testbench
tb = generate_testbench("sc_lif", {"v": "expr"}, framework="cocotb")
with open("test_sc_lif.py", "w") as f:
    f.write(tb)

# UVM testbench
tb_uvm = generate_testbench("sc_lif", {"v": "expr"}, framework="uvm")
```

---

## Wave 9 — Extensibility Layer (§52–§59)

### §52. CDC Analyzer — `analyze_cdc()`

Formal clock domain crossing analysis across neuron array boundaries.
Identifies unsynchronized crossings and recommends synchroniser types.

```python
from sc_neurocore.compiler.advanced_features import analyze_cdc

report = analyze_cdc(
    {"v": "u + I", "u": "v - threshold"},
    clock_domains={"v": "clk_fast", "u": "clk_slow"},
)
print(f"Crossings: {report.total_crossings}")
print(f"Safe: {report.safe}")
for c in report.crossings:
    print(f"  {c['signal']}: {c['src_domain']}→{c['dst_domain']} ({c['sync_type']})")
```

---

### §53. TOML Profile Auto-Loader — `load_profiles_from_toml()`

**The extensibility endgame.** Users and vendors can define custom hardware
profiles in a `.toml` file without modifying SC-NeuroCore source code.

```toml
# my_custom_profiles.toml
[[profile]]
name = "my_asic_v2"
vendor = "InternalDesign"
family = "ASIC-v2"
platform_class = "asic"
data_width = 24
fraction = 12
overflow = "saturate"
rounding = "nearest"
max_freq_mhz = 800
notes = "Internal 24-bit ASIC for production deployment."
```

```python
from sc_neurocore.compiler.advanced_features import load_profiles_from_toml
from sc_neurocore.compiler.hardware_profiles import get_profile

loaded = load_profiles_from_toml("my_custom_profiles.toml")
print(loaded)  # ['my_asic_v2']

p = get_profile("my_asic_v2")
print(f"{p.vendor} {p.family}: Q{p.data_width-p.fraction}.{p.fraction}")
```

**Any future hardware platform can be added with zero code changes.**

---

### §54. Multi-Die Floorplanner — `plan_multi_die_floorplan()`

Assigns neuron blocks to chiplet/die positions using first-fit-decreasing
bin packing for 3D-stacked and 2.5D chiplet architectures.

```python
from sc_neurocore.compiler.advanced_features import plan_multi_die_floorplan

result = plan_multi_die_floorplan(
    blocks={"visual_cortex": 800, "motor_cortex": 600,
            "prefrontal": 400, "cerebellum": 500},
    die_capacity=1000, num_dies=4,
)
for block, die in result.die_assignment.items():
    print(f"  {block} → Die {die}")
print(f"Dies used: {result.total_dies}")
```

---

### §55. Regression Watchdog — `check_regression()`

Detects performance regressions between compilations by comparing metric
deltas against configurable thresholds.

```python
from sc_neurocore.compiler.advanced_features import check_regression

checks = check_regression(
    baseline={"area_luts": 1200, "fmax_mhz": 250, "power_mw": 85},
    current={"area_luts": 1350, "fmax_mhz": 240, "power_mw": 90},
    threshold_pct=5.0,
)
for c in checks:
    status = "⚠️ REGRESSED" if c.regression else "✅ OK"
    print(f"  {c.metric}: {c.baseline}→{c.current} ({c.delta_pct:+.1f}%) {status}")
```

---

### §56. License Compliance Checker — `check_license_compliance()`

Verifies IP core licensing compatibility using an SPDX-based compatibility
matrix, preventing copyleft contamination.

```python
from sc_neurocore.compiler.advanced_features import check_license_compliance

result = check_license_compliance(
    project_license="Apache-2.0",
    dependencies={
        "numpy": "BSD-3",
        "scipy": "BSD-3",
        "gpl_ip_core": "GPL-3.0",
    },
)
print(f"Compatible: {result.compatible}")
for c in result.conflicts:
    print(f"  ⚠️ {c}")
```

---

### §57. Power State Machine — `generate_power_state_machine()`

Generates synthesisable Verilog FSMs for ultra-low-power operation with
configurable sleep/wake/hibernate states.

```python
from sc_neurocore.compiler.advanced_features import generate_power_state_machine

verilog = generate_power_state_machine("sc_lif")
# Default states: ACTIVE → IDLE → SLEEP → HIBERNATE
with open("sc_lif_power_fsm.v", "w") as f:
    f.write(verilog)
```

---

### §58. Platform Discovery Hook — `register_platform_hook()`

Plugin API enabling third-party vendors to register hardware profiles at
runtime without modifying SC-NeuroCore source.

```python
from sc_neurocore.compiler.advanced_features import (
    register_platform_hook, discover_platforms,
)
from sc_neurocore.compiler.hardware_profiles import HardwareProfile

def vendor_x_profiles():
    return [HardwareProfile(
        name="vendorx_nn_v3",
        vendor="VendorX", family="NN-v3",
        platform_class="accelerator",
        data_width=16, fraction=8,
        overflow="saturate", rounding="nearest",
    )]

register_platform_hook(vendor_x_profiles)
discovered = discover_platforms()
print(f"Discovered: {discovered}")
```

---

### §59. Compilation Report — `generate_compilation_report()`

One-click comprehensive markdown report consolidating target info, carbon
footprint, reliability prediction, and module specification.

```python
from sc_neurocore.compiler.advanced_features import generate_compilation_report

report = generate_compilation_report(
    "sc_lif", {"v": "-(v)/tau + I"}, "artix7",
)
with open("compilation_report.md", "w") as f:
    f.write(report)
```

---

## Quick Reference Table

| § | Function | Category |
|---|----------|----------|
| 26 | `generate_equivalence_sketch()` | Verification |
| 27 | `partition_timescales()` | Optimisation |
| 28 | `generate_provenance_chain()` | Traceability |
| 29 | `generate_compliance_matrix()` | Certification |
| 30 | `generate_energy_schedule()` | Power |
| 31 | `lint_side_channels()` | Security |
| 32 | `generate_drift_compensator()` | Reliability |
| 33 | `plan_heterogeneous_dispatch()` | Deployment |
| 34 | `recommend_target()` | Auto-Targeting |
| 35 | `plan_partial_reconfiguration()` | FPGA |
| 36 | `score_supply_chain_risk()` | Risk |
| 37 | `generate_bittrue_kernel()` | Verification |
| 38 | `classify_model_complexity()` | Analysis |
| 39 | `CompilationCache` | Performance |
| 40 | `estimate_thermal_envelope()` | Thermal |
| 41 | `optimize_network_topology()` | Deployment |
| 42 | `import_nir_graph()` | Import |
| 43 | `verify_ode_stability()` | Verification |
| 44 | `generate_power_intent()` | Power |
| 45 | `estimate_carbon_footprint()` | ESG |
| 46 | `insert_debug_probes()` | Debug |
| 47 | `generate_memory_map()` | SoC |
| 48 | `score_portability()` | Analysis |
| 49 | `predict_reliability()` | Reliability |
| 50 | `generate_fault_tree()` | Safety |
| 51 | `generate_testbench()` | Verification |
| 52 | `analyze_cdc()` | Verification |
| 53 | `load_profiles_from_toml()` | Extensibility |
| 54 | `plan_multi_die_floorplan()` | Deployment |
| 55 | `check_regression()` | QA |
| 56 | `check_license_compliance()` | Legal |
| 57 | `generate_power_state_machine()` | Power |
| 58 | `register_platform_hook()` | Extensibility |
| 59 | `generate_compilation_report()` | Reporting |

## Usage Patterns

### Pattern 1: Train → Import → Compile → Deploy

```python
# 1. Train in snnTorch (external)
# 2. Export to NIR (external)
# 3. Import into SC-NeuroCore
graph = import_nir_graph(nir_data, framework="snnTorch")

# 4. Verify stability
stability = verify_ode_stability(graph.equations, dt=0.1)
assert stability.stable

# 5. Auto-select target
recs = recommend_target(constraints={"max_power_mw": 500})

# 6. Compile + generate report
report = generate_compilation_report("my_snn", graph.equations, recs[0]["name"])
```

### Pattern 2: Safety-Critical Certification Pipeline

```python
# 1. Generate compliance matrix
matrix = generate_compliance_matrix("DO-254", equations, "artix7")

# 2. Generate fault tree
ft = generate_fault_tree("sc_lif", equations)

# 3. Generate provenance chain
chain = generate_provenance_chain(equations, "artix7", "lab_name")

# 4. Predict reliability
rel = predict_reliability(voltage_v=0.9, temperature_c=85)

# 5. Generate testbench
tb = generate_testbench("sc_lif", equations)
```

### Pattern 3: Custom Platform Registration

```python
# Option A: TOML file (no code changes)
loaded = load_profiles_from_toml("my_platforms.toml")

# Option B: Runtime hook (vendor SDK)
register_platform_hook(my_vendor_discovery_fn)
discover_platforms()

# Option C: Auto-construct from spec sheet (Wave 10)
from sc_neurocore.compiler.hardware_profiles import HardwareProfile
p = HardwareProfile.from_constraints("my_chip", vendor="MyVendor", max_power_budget_mw=5)
```

### Pattern 4: Security-Hardened Deployment (Wave 10)

```python
from sc_neurocore.compiler.advanced_features import (
    lint_hardware_trojans, obfuscate_ip, embed_watermark,
    generate_sbom, schedule_seu_scrubbing,
)

# 1. Trojan detection
trojan = lint_hardware_trojans(equations)
assert trojan.risk_level == "LOW"

# 2. IP protection
obf = obfuscate_ip("sc_lif", equations, key_length=128)

# 3. Watermark
wm = embed_watermark("sc_lif", equations, owner_id="MyLab")

# 4. Compliance
sbom = generate_sbom("sc_lif", "artix7")

# 5. Space-grade scrubbing
scrub = schedule_seu_scrubbing(1_000_000, orbit_altitude_km=800)
```

---

## Wave 10 — Security, Sovereignty & Endgame (§60–§67)

### §60. Hardware Trojan Lint — `lint_hardware_trojans()`

Detects suspicious combinational paths in the ODE dependency graph that
could hide hardware trojans — dormant triggers and payload injection.

```python
from sc_neurocore.compiler.advanced_features import lint_hardware_trojans

result = lint_hardware_trojans(
    {"v": "-(v)/tau + I", "u": "a*(b*v - u)"},
)
print(f"Risk: {result.risk_level}")
print(f"Checks: {result.total_checks}")
```

---

### §61. SBOM/HBOM Generator — `generate_sbom()`

Generates CycloneDX/SPDX Bill of Materials. **Required by EU CRA (2026).**

```python
from sc_neurocore.compiler.advanced_features import generate_sbom

sbom = generate_sbom("sc_lif", "artix7",
    dependencies={"numpy": "1.26.0"}, sbom_format="CycloneDX")
print(f"Components: {sbom.total_components}")
```

---

### §62. HIL Calibration Stub — `generate_hil_calibration()`

Step-by-step hardware-in-the-loop calibration for analog drift compensation.

```python
from sc_neurocore.compiler.advanced_features import generate_hil_calibration

cal = generate_hil_calibration("sc_lif", {"v": "expr", "u": "expr"},
    parameters={"tau": (5.0, 50.0)})
for step in cal.protocol_steps:
    print(step)
```

---

### §63. Digital Twin Shadow — `generate_digital_twin()`

Generates a Python class that mirrors deployed hardware state in real-time.

```python
from sc_neurocore.compiler.advanced_features import generate_digital_twin

twin_code = generate_digital_twin("sc_lif", {"v": "-(v)/tau"}, "artix7")
# Contains: __init__, step(), compare()
```

---

### §64. UCIe Protocol Mapper — `map_ucie_protocol()`

Maps neuron blocks to UCIe die-to-die protocol lanes for chiplet architectures.

```python
from sc_neurocore.compiler.advanced_features import map_ucie_protocol

mapping = map_ucie_protocol({"cortex": 256, "motor": 128})
print(f"Total bandwidth: {mapping.total_bandwidth_gbps} Gbps")
```

---

### §65. SEU Scrub Scheduler — `schedule_seu_scrubbing()`

Generates scrubbing schedules for space-grade configuration memory.

```python
from sc_neurocore.compiler.advanced_features import schedule_seu_scrubbing

s = schedule_seu_scrubbing(1_000_000, orbit_altitude_km=800)
print(f"Interval: {s.interval_ms:.0f} ms, SEU rate: {s.expected_seu_rate:.4f}/day")
```

---

### §66. IP Obfuscation — `obfuscate_ip()`

Logic locking + structural transformation for IP reverse-engineering protection.

```python
from sc_neurocore.compiler.advanced_features import obfuscate_ip

result = obfuscate_ip("sc_lif", {"v": "expr"}, key_length=128)
print(f"Key: {result.key_bits} bits, Methods: {result.techniques_applied}")
```

---

### §67. Model Watermark — `embed_watermark()`

Embeds verifiable watermarks into compiled netlists. Survives synthesis.

```python
from sc_neurocore.compiler.advanced_features import embed_watermark

wm = embed_watermark("sc_lif", {"v": "expr"}, owner_id="MyLab")
print(f"Hash: {wm.watermark_hash}, Overhead: {wm.overhead_percent}%")
```

---

## Quick Reference Table

| § | Function | Category |
|---|----------|----------|
| 26 | `generate_equivalence_sketch()` | Verification |
| 27 | `partition_timescales()` | Optimisation |
| 28 | `generate_provenance_chain()` | Traceability |
| 29 | `generate_compliance_matrix()` | Certification |
| 30 | `generate_energy_schedule()` | Power |
| 31 | `lint_side_channels()` | Security |
| 32 | `generate_drift_compensator()` | Reliability |
| 33 | `plan_heterogeneous_dispatch()` | Deployment |
| 34 | `recommend_target()` | Auto-Targeting |
| 35 | `plan_partial_reconfiguration()` | FPGA |
| 36 | `score_supply_chain_risk()` | Risk |
| 37 | `generate_bittrue_kernel()` | Verification |
| 38 | `classify_model_complexity()` | Analysis |
| 39 | `CompilationCache` | Performance |
| 40 | `estimate_thermal_envelope()` | Thermal |
| 41 | `optimize_network_topology()` | Deployment |
| 42 | `import_nir_graph()` | Import |
| 43 | `verify_ode_stability()` | Verification |
| 44 | `generate_power_intent()` | Power |
| 45 | `estimate_carbon_footprint()` | ESG |
| 46 | `insert_debug_probes()` | Debug |
| 47 | `generate_memory_map()` | SoC |
| 48 | `score_portability()` | Analysis |
| 49 | `predict_reliability()` | Reliability |
| 50 | `generate_fault_tree()` | Safety |
| 51 | `generate_testbench()` | Verification |
| 52 | `analyze_cdc()` | Verification |
| 53 | `load_profiles_from_toml()` | Extensibility |
| 54 | `plan_multi_die_floorplan()` | Deployment |
| 55 | `check_regression()` | QA |
| 56 | `check_license_compliance()` | Legal |
| 57 | `generate_power_state_machine()` | Power |
| 58 | `register_platform_hook()` | Extensibility |
| 59 | `generate_compilation_report()` | Reporting |
| 60 | `lint_hardware_trojans()` | Security |
| 61 | `generate_sbom()` | Compliance |
| 62 | `generate_hil_calibration()` | Calibration |
| 63 | `generate_digital_twin()` | Deployment |
| 64 | `map_ucie_protocol()` | Chiplet |
| 65 | `schedule_seu_scrubbing()` | Space |
| 66 | `obfuscate_ip()` | Security |
| 67 | `embed_watermark()` | IP Protection |

## Further Reading

- [Hardware Profiles Guide](hardware_profiles.md) — all 175 profiles
- [Frontier Platforms Guide](frontier_platforms.md) — 31 platform classes
- [Platform Extensibility Guide](platform_extensibility.md) — TOML + hook + from_constraints
- [Safety Certification Guide](safety_certification.md) — DO-254, IEC 61508, ISO 26262
- [Carbon & Sustainability Guide](carbon_sustainability.md) — ESG features
- [Verification & Debug Guide](verification_debug.md) — 10 V&V features
- [Multi-Target Deployment Guide](multi_target_deployment.md) — heterogeneous deploy
- [Static Analysis Guide](static_analysis_guide.md) — guard bits, overflow
- [Deployment Guide](deployment_guide.md) — constraints, TCL, bitstream
- [Co-Simulation Guide](cosimulation_guide.md) — Python↔Verilog
- [Precision Modes Guide](precision_modes.md) — Q-format modes
