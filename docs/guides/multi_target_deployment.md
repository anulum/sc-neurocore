<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->

# Multi-Target Deployment Guide

SC-NeuroCore can deploy a single SNN model to **any combination** of 175
hardware profiles across 31 platform classes. This guide covers the features
for multi-target, multi-die, chiplet, and heterogeneous deployment.

## Auto-Target Recommendation (§34)

When you don't know which chip to use, the compiler recommends the best match:

```python
from sc_neurocore.compiler.advanced_features import recommend_target

recs = recommend_target(
    constraints={
        "max_power_mw": 500,
        "min_freq_mhz": 100,
        "max_width": 16,
    },
    top_k=5,
)
for r in recs:
    print(f"  {r['name']:30s} score={r['score']:.2f}")
```

## Portability Scoring (§48)

Before targeting multiple platforms, check how portable your model is:

```python
from sc_neurocore.compiler.advanced_features import score_portability

# Simple LIF — runs on almost everything
s = score_portability({"v": "-(v - v_rest) / tau + I"})
print(f"Portable to {s.compatible_profiles}/{s.total_profiles} profiles")
print(f"Score: {s.score}/100")

# Complex model — may have blockers
s = score_portability({"v": "g*m*m*m*h + g*n*n*n*n", "m": "...", "h": "...", "n": "..."})
if s.blockers:
    print("Blockers:")
    for b in s.blockers:
        print(f"  ⚠️ {b}")
```

## Heterogeneous Dispatch (§33)

Deploy different neuron populations on different hardware:

```python
from sc_neurocore.compiler.advanced_features import plan_heterogeneous_dispatch

plan = plan_heterogeneous_dispatch(
    populations={
        "retina": "max78000",           # Edge MCU (sensor)
        "visual_cortex": "artix7",      # FPGA (processing)
        "decision": "loihi2",           # Neuromorphic (learning)
        "motor": "rp2040",              # MCU (actuation)
    },
    connections=[
        ("retina", "visual_cortex"),
        ("visual_cortex", "decision"),
        ("decision", "motor"),
    ],
)
```

## Multi-Die Floorplanning (§54)

For chiplet and 3D-stacked architectures, assign neuron blocks to dies:

```python
from sc_neurocore.compiler.advanced_features import plan_multi_die_floorplan

result = plan_multi_die_floorplan(
    blocks={
        "visual_cortex": 800,
        "auditory_cortex": 600,
        "motor_cortex": 400,
        "prefrontal": 500,
        "cerebellum": 900,
        "hippocampus": 300,
    },
    die_capacity=1000,
    num_dies=4,
)
for block, die in result.die_assignment.items():
    print(f"  {block:20s} → Die {die}")
print(f"\nDie utilisation:")
for die, util in result.die_utilization.items():
    print(f"  Die {die}: {util:.0%}")
```

## Network Topology Optimisation (§41)

Minimise inter-chip spike bandwidth using graph partitioning:

```python
from sc_neurocore.compiler.advanced_features import optimize_network_topology

result = optimize_network_topology(
    adjacency={
        "V1": ["V2", "V4"],
        "V2": ["V1", "V4", "IT"],
        "V4": ["V1", "V2", "IT"],
        "IT": ["V2", "V4", "PFC"],
        "PFC": ["IT", "M1"],
        "M1": ["PFC"],
    },
    num_chips=2,
)
print(f"Bandwidth reduction: {result.bandwidth_reduction:.1%}")
```

## Partial Reconfiguration (§35)

Time-multiplex SNN layers on the same FPGA fabric:

```python
from sc_neurocore.compiler.advanced_features import plan_partial_reconfiguration

plan = plan_partial_reconfiguration(
    regions={
        "conv_layer_1": 5000,  # LUTs
        "conv_layer_2": 4000,
        "fc_layer": 3000,
    },
    total_luts=10000,
)
print(f"Schedule: {plan.schedule}")
```

## Memory Map Integration (§47)

Generate SoC address decoders for multi-neuron arrays:

```python
from sc_neurocore.compiler.advanced_features import generate_memory_map

mmap = generate_memory_map(
    "sc_cortex",
    {"v": "expr", "u": "expr", "I_syn": "expr"},
    num_neurons=4096,
    data_width=16,
    base_address=0x4000_0000,
)
print(f"Address space: {mmap.total_bytes:,} bytes")
print(f"First 5 entries:")
for e in mmap.entries[:5]:
    print(f"  0x{e['address']:08X}: {e['name']} ({e['width']}b)")
```

## Cross-Compilation Cache (§39)

Avoid redundant compilations when targeting multiple platforms:

```python
from sc_neurocore.compiler.advanced_features import CompilationCache

cache = CompilationCache()

# First compilation — cache miss
result = compile_neuron(equations, "artix7")
cache.store("lif_artix7", result)

# Second time — instant retrieval
cached = cache.lookup("lif_artix7")
assert cached is not None
```

## Supply Chain Risk (§36)

Evaluate geopolitical risk before committing to a hardware platform:

```python
from sc_neurocore.compiler.advanced_features import score_supply_chain_risk

for target in ["artix7", "loihi2", "bae_rad750_sq", "tsmc_cim_n7"]:
    risk = score_supply_chain_risk(target)
    print(f"  {target:20s} Risk: {risk.overall_risk}")
```

## UCIe Chiplet Protocol Mapping (§64)

Map neuron array blocks to UCIe die-to-die protocol lanes for
chiplet-based multi-die architectures:

```python
from sc_neurocore.compiler.advanced_features import map_ucie_protocol

mapping = map_ucie_protocol(
    {"visual_cortex": 256, "motor_cortex": 128, "prefrontal": 64},
    lane_bandwidth_gbps=32.0,
    protocol_version="UCIe 2.0",
)
for block, lanes in mapping.lanes.items():
    print(f"  {block}: {lanes} UCIe lanes")
print(f"Total: {mapping.total_bandwidth_gbps} Gbps")
```

## Digital Twin Shadow (§63)

Generate a software shadow that mirrors deployed hardware for runtime
monitoring and anomaly detection:

```python
from sc_neurocore.compiler.advanced_features import generate_digital_twin

twin = generate_digital_twin("sc_cortex", equations, "artix7")
# Deploy twin alongside hardware — compare on every cycle
```

## SBOM for Deployment Compliance (§61)

Generate Bill of Materials for every deployed target (EU CRA mandatory):

```python
from sc_neurocore.compiler.advanced_features import generate_sbom

for target in ["artix7", "loihi2", "sifive_x280_ai"]:
    sbom = generate_sbom("sc_cortex", target)
    print(f"  {target}: {sbom.total_components} components")
```

## Complete Deployment Workflow

```
┌─────────────────────────────────────────────────┐
│  1. score_portability()             — §48       │
│  2. recommend_target()              — §34       │
│  3. score_supply_chain_risk()       — §36       │
│  4. estimate_carbon_footprint()     — §45       │
│  5. plan_heterogeneous_dispatch()   — §33       │
│  6. plan_multi_die_floorplan()      — §54       │
│  7. map_ucie_protocol()             — §64       │
│  8. optimize_network_topology()     — §41       │
│  9. generate_memory_map()           — §47       │
│ 10. generate_power_intent()         — §44       │
│ 11. generate_sbom()                 — §61       │
│ 12. generate_digital_twin()         — §63       │
│ 13. generate_compilation_report()   — §59       │
└─────────────────────────────────────────────────┘
```

## Further Reading

- [Compiler Intelligence Guide](compiler_intelligence.md) — all 67 features
- [Hardware Profiles Guide](hardware_profiles.md) — all 175 profiles
- [Frontier Platforms Guide](frontier_platforms.md) — 31 platform classes
- [Platform Extensibility Guide](platform_extensibility.md) — TOML + hook + from_constraints
- [Verification & Debug Guide](verification_debug.md) — 14 V&V features
- [Carbon & Sustainability Guide](carbon_sustainability.md) — ESG features
