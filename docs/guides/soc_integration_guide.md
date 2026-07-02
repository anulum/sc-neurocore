<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->

# SoC Integration Guide

This guide covers integrating compiled SC-NeuroCore neurons into
System-on-Chip (SoC) designs: bus wrappers, mixed-precision, host drivers,
and IP packaging.

## Bus Interface Generation

Every compiled neuron is a standalone Verilog module. To integrate it into a
real SoC (Zynq, Nios, RISC-V), it needs a bus wrapper that maps parameters
to memory-mapped registers.

### AXI4-Lite (Xilinx / AMD)

```python
from sc_neurocore.hdl_gen.bus_interface import generate_bus_wrapper

axi = generate_bus_wrapper(
    "sc_lif",
    params={"P_V_REST": 16, "P_V_THRESH": 16, "P_TAU_M": 16},
    bus="axi_lite",
    data_width=16,
)

with open("sc_lif_axi_lite.v", "w") as f:
    f.write(axi)
```

### Wishbone B4 (LiteX / Open-Source RISC-V)

```python
wb = generate_bus_wrapper(
    "sc_lif",
    params={"P_V_REST": 16, "P_V_THRESH": 16, "P_TAU_M": 16},
    bus="wishbone",
    data_width=16,
)
```

### Register Map

All wrappers share the same register layout:

| Offset | Register | Access | Description |
|:------:|----------|:------:|-------------|
| `0x00` | `CTRL` | R/W | bit 0 = enable, bit 1 = reset |
| `0x04` | `I_T` | R/W | Input current (Q-format) |
| `0x08` | `SPIKE_COUNT` | R | Spike counter (auto-increment) |
| `0x0C` | `P_V_REST` | R/W | Resting potential |
| `0x10` | `P_V_THRESH` | R/W | Threshold voltage |
| `0x14` | `P_TAU_M` | R/W | Membrane time constant |

```python
from sc_neurocore.hdl_gen.bus_interface import generate_register_map

rmap = generate_register_map(
    {"P_V_REST": 16, "P_V_THRESH": 16, "P_TAU_M": 16},
    base_address=0x4000_0000,
)
# → {'CTRL': 0x40000000, 'I_T': 0x40000004, 'SPIKE_COUNT': 0x40000008, ...}
```

### Features

- **Spike interrupt**: `irq_spike` output — connect to GIC/NVIC
- **Control register**: enable/disable/reset via software
- **Spike counter**: auto-incrementing, read-only, reset-clearable

---

## Mixed-Precision Per Variable

Different state variables can use different bit widths within the same module,
saving 30–50% of FPGA resources for multi-variable models.

### Dict API (Explicit Control)

```python
from sc_neurocore.compiler.mixed_precision import MixedPrecisionSpec, PrecisionConfig

spec = MixedPrecisionSpec({
    "v": PrecisionConfig(16, 8),   # Q7.8 for membrane voltage
    "u": PrecisionConfig(8, 4),    # Q3.4 for recovery variable
})

print(spec.total_bits)    # 24 (vs 32 for uniform Q8.8)
print(spec.summary())
# Mixed-Precision Allocation (24 bits total):
#   v            → Q7.8      (16-bit)  range=[-128.0, 127.0]  res=0.003906
#   u            → Q3.4      (8-bit)   range=[-8.0, 7.0]      res=0.062500
```

### Constraint Solver (Automatic)

```python
from sc_neurocore.compiler.mixed_precision import solve_precision

spec = solve_precision(
    bounds={"v": (-128, 127), "u": (-10, 10)},
    min_resolution={"v": 0.01, "u": 0.1},
    max_total_bits=24,          # Budget constraint
    align_to=8,                 # Byte-align
)

for var in spec.variables:
    cfg = spec.get(var)
    print(f"{var}: {cfg.q_label} ({cfg.data_width}-bit)")
```

### Preset Shorthand

```python
from sc_neurocore.compiler.mixed_precision import from_preset

spec = from_preset({"v": "q88", "u": "q44"})
```

Available presets: `q17`, `q44`, `q88`, `q412`, `q115`, `q99`, `q1212`,
`q1413`, `q2012`, `q1616`, `q824`, `q1818`.

---

## Host Driver Generation

Auto-generate C/Python host-side drivers that match the bus wrapper register
map. Includes Q-format encoding/decoding.
Module and parameter names are sanitized before source emission. Names that
sanitize to empty or collide after sanitization are rejected, preserving valid
Python classes, C include guards, register constants, and setter functions.

### Python Driver

```python
from sc_neurocore.compiler.deployment import generate_host_driver

drv = generate_host_driver(
    "sc_lif",
    params={"P_V_REST": 16, "P_V_THRESH": 16, "P_TAU_M": 16},
    language="python",
    base_address=0x4000_0000,
    data_width=16,
    fraction=8,
)

with open("sc_lif_driver.py", "w") as f:
    f.write(drv)
```

Generated driver usage:

```python
# On the SoC (e.g. PYNQ, MicroPython)
from sc_lif_driver import ScLifDriver

drv = ScLifDriver(read_fn=mmio.read, write_fn=mmio.write)
drv.reset()
drv.set_v_rest(-65.0)
drv.set_v_thresh(-50.0)
drv.set_tau_m(10.0)
drv.enable()
drv.set_current(50.0)
print(f"Spikes: {drv.get_spike_count()}")
```

### C Driver

```python
c_drv = generate_host_driver("sc_lif", params, language="c")
with open("sc_lif_driver.h", "w") as f:
    f.write(c_drv)
```

Generated C usage:

```c
#include "sc_lif_driver.h"

sc_lif_reset();
sc_lif_set_current(50.0f);
sc_lif_set_v_rest(-65.0f);
sc_lif_set_v_thresh(-50.0f);
sc_lif_set_tau_m(10.0f);
sc_lif_enable();
uint32_t spikes = sc_lif_get_spikes();
```

---

## IP-XACT Packaging (Vivado IP Integrator)

Generate IEEE 1685 IP-XACT XML for drag-and-drop integration in Vivado:

```python
from sc_neurocore.hdl_gen.ip_xact import generate_ip_xact

xml = generate_ip_xact(
    "sc_lif",
    vendor="anulum.li",
    version="1.0",
    data_width=16,
    params={"P_V_REST": 16, "P_V_THRESH": 16},
    bus="axi_lite",
)

with open("component.xml", "w") as f:
    f.write(xml)
```

The generated XML includes:
- Bus interfaces (AXI4-Lite slave, clock, reset)
- Port definitions with bit widths
- File sets referencing the Verilog source
- Parameter definitions

---

## VHDL-2008 Output (Aerospace/Defense)

For DO-254 / IEC 61508 compliance where VHDL is required:

```python
from sc_neurocore.compiler.intelligence import verilog_to_vhdl_wrapper

vhdl = verilog_to_vhdl_wrapper("sc_lif", data_width=16, signed=True)
with open("sc_lif_vhdl.vhd", "w") as f:
    f.write(vhdl)
```

The wrapper instantiates the Verilog module as a VHDL component, enabling
mixed-language simulation (Vivado, Questa, GHDL + Verilator).

---

## Cross-References

- [Hardware Profiles Guide](hardware_profiles.md) — 65 platform profiles
- [Static Analysis Guide](static_analysis_guide.md) — guard bits, overflow proof, SVA
- [Deployment Guide](deployment_guide.md) — constraints, TCL, bitstream
- [Precision Modes Guide](precision_modes.md) — 11 Q-format modes
