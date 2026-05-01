<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->

# Carbon & Sustainability Guide

SC-NeuroCore is the world's first neuromorphic compiler with built-in
**carbon footprint estimation** and **energy-aware compilation**. This guide
covers the ESG (Environmental, Social, Governance) features required for
EU carbon labelling compliance (mandatory from 2027).

## Carbon Footprint Estimation (§45)

### Basic Usage

```python
from sc_neurocore.compiler.intelligence import estimate_carbon_footprint

# Compare CO₂ impact across targets
targets = ["artix7", "loihi2", "finalspark_neuroplatform", "max78000"]
for t in targets:
    c = estimate_carbon_footprint(t, power_mw=100)
    print(f"{t:30s} Mfg: {c.manufacturing_kg_co2:6.1f} kg  "
          f"5yr: {c.total_5yr_kg_co2:8.1f} kg CO₂")
```

### What Is Estimated

The estimator calculates two components:

1. **Manufacturing CO₂** — Embodied carbon from semiconductor fabrication,
   packaging, and assembly. Scales with die area and technology node.

2. **Operational CO₂** — Energy consumption over a 5-year lifetime using
   grid-average carbon intensity (0.4 kg CO₂/kWh global average, adjustable).

```
Total CO₂ = Manufacturing + (Power × Hours × Grid_Intensity)
```

### Custom Grid Intensity

Different regions have different grid carbon intensities:

| Region | kg CO₂/kWh |
|--------|----------:|
| France (nuclear) | 0.05 |
| Norway (hydro) | 0.02 |
| Germany (mixed) | 0.35 |
| USA average | 0.40 |
| India (coal) | 0.70 |
| China (coal) | 0.60 |

```python
# Override for French data centre (nuclear grid)
c = estimate_carbon_footprint("artix7", power_mw=100)
# Adjust operational component
french_5yr = c.manufacturing_kg_co2 + (
    0.1 * 8760 * 5 * 0.05 / 1000  # 100mW, 5yr, 0.05 kg/kWh
)
```

## Energy Schedule Generation (§30)

### Duty-Cycled Operation

Most neuromorphic systems are event-driven and spend most time sleeping.
The energy schedule generator models this:

```python
from sc_neurocore.compiler.intelligence import generate_energy_schedule

sched = generate_energy_schedule(
    power_active_mw=100.0,   # during spike processing
    power_sleep_mw=0.01,     # deep sleep
    duty_cycle=0.05,         # active 5% of time
)
print(f"Average power: {sched.avg_power_mw:.2f} mW")
print(f"Annual energy: {sched.annual_kwh:.4f} kWh")
```

## Power State Machine (§57)

### Ultra-Low-Power FSM

Generate hardware FSMs that implement sleep/wake transitions:

```python
from sc_neurocore.compiler.intelligence import generate_power_state_machine

# Default: ACTIVE → IDLE → SLEEP → HIBERNATE
verilog = generate_power_state_machine("sc_lif")
with open("sc_lif_power_fsm.v", "w") as f:
    f.write(verilog)
```

### Custom States

```python
# Event-driven neuron with fast wake
verilog = generate_power_state_machine(
    "sc_lif",
    states=["ACTIVE", "DROWSY", "RETENTION", "SHUTDOWN"],
)
```

## Power Intent (UPF) Generation (§44)

### IEEE 1801 UPF

Generate power domains, isolation cells, and retention for multi-voltage SoCs:

```python
from sc_neurocore.compiler.intelligence import generate_power_intent

upf = generate_power_intent("sc_lif_array", num_domains=8, always_on=True)
with open("sc_lif_array.upf", "w") as f:
    f.write(upf)
# Use in Synopsys Design Compiler / Cadence Innovus
```

## Thermal Envelope Estimation (§40)

### Compile-Time Thermal Check

Catch thermal violations before silicon:

```python
from sc_neurocore.compiler.intelligence import estimate_thermal_envelope

t = estimate_thermal_envelope(
    power_mw=2000,
    theta_ja=25.0,  # package thermal resistance
    t_ambient=40.0,  # worst-case ambient
)
print(f"Junction temp: {t.t_junction}°C")
print(f"Margin to Tj_max: {t.thermal_margin}°C")
print(f"Status: {t.pass_fail}")
```

## Green Hardware Selection Workflow

### Step 1: Identify Low-Carbon Targets

```python
from sc_neurocore.compiler.intelligence import estimate_carbon_footprint
from sc_neurocore.compiler.platforms import list_profile_names

# Find the greenest targets
results = []
for name in list_profile_names():
    c = estimate_carbon_footprint(name, power_mw=100)
    results.append((c.total_5yr_kg_co2, name))

results.sort()
print("Top 10 greenest targets:")
for co2, name in results[:10]:
    print(f"  {name:30s} {co2:.1f} kg CO₂/5yr")
```

### Step 2: Generate Report

```python
from sc_neurocore.compiler.intelligence import generate_compilation_report

report = generate_compilation_report(
    "sc_lif", {"v": "-(v)/tau + I"},
    results[0][1],  # greenest target
    include_carbon=True,
)
```

## Regulatory Timeline

| Year | Regulation | Impact |
|------|-----------|--------|
| 2024 | EU CSRD | ESG reporting for large companies |
| 2025 | EU Taxonomy | Sustainable activity classification |
| 2027 | EU Carbon Label | **Mandatory product carbon labelling** |
| 2028 | EU CBAM | Carbon border adjustment mechanism |
| 2030 | EU Climate Law | 55% emission reduction target |

SC-NeuroCore's `estimate_carbon_footprint()` provides the data required
for all of these regulations.

## Further Reading

- [Compiler Intelligence Guide](compiler_intelligence.md) — all 67 features
- [Hardware Profiles Guide](hardware_profiles.md) — all 175 profiles
- [Deployment Guide](deployment_guide.md) — constraints, bitstream
- [Safety Certification Guide](safety_certification.md) — DO-254/ISO 26262
