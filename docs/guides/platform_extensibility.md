<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->

# Platform Extensibility Guide

SC-NeuroCore provides three extensibility mechanisms for adding custom hardware
profiles **without modifying SC-NeuroCore source code**:

1. **TOML Profile Auto-Loader** (§53) — Static configuration files
2. **Platform Discovery Hook** (§58) — Runtime plugin API
3. **Generic Profile Constructor** — `HardwareProfile.from_constraints()`

Together, these ensure that **any hardware that will ever be invented** can
be supported with zero code changes.

## TOML Profile Auto-Loader

### TOML File Format

Create a `.toml` file containing one or more `[[profile]]` sections:

```toml
# custom_profiles.toml

[[profile]]
name = "my_custom_chip"          # Required: unique identifier
vendor = "MyCompany"             # Optional: vendor name
family = "ChipFamily-v2"        # Optional: chip family
platform_class = "asic"          # Optional: platform class
data_width = 16                  # Optional: bit width (default: 16)
fraction = 8                    # Optional: fraction bits (default: 8)
overflow = "saturate"            # Optional: "saturate" or "wrap"
rounding = "nearest"             # Optional: "nearest" or "truncate"
dsp_block = "DSP48E2"           # Optional: DSP block type
dsp_mult_a = 18                 # Optional: DSP multiplier width A
dsp_mult_b = 27                 # Optional: DSP multiplier width B
max_freq_mhz = 500              # Optional: max frequency
notes = "Description of chip."   # Optional: human-readable notes
```

### Loading Profiles

```python
from sc_neurocore.compiler.advanced_features import load_profiles_from_toml
from sc_neurocore.compiler.hardware_profiles import get_profile

# Load all profiles from TOML file
loaded = load_profiles_from_toml("custom_profiles.toml")
print(f"Loaded profiles: {loaded}")

# Use them immediately
p = get_profile("my_custom_chip")
print(f"Q{p.data_width - p.fraction}.{p.fraction} {p.overflow}")
```

### Multiple Files

Load multiple TOML files for different labs or projects:

```python
load_profiles_from_toml("lab_a_profiles.toml")
load_profiles_from_toml("lab_b_profiles.toml")
load_profiles_from_toml("vendor_sdk_profiles.toml")
```

### Example: University Lab Setup

```toml
# university_lab_profiles.toml

[[profile]]
name = "zynq_7020_lab"
vendor = "Xilinx"
family = "Zynq-7020"
platform_class = "fpga"
data_width = 16
fraction = 8
overflow = "wrap"
rounding = "truncate"
dsp_block = "DSP48E1"
dsp_mult_a = 18
dsp_mult_b = 18
max_freq_mhz = 200
notes = "Zynq-7020 on custom lab board with 512MB DDR3."

[[profile]]
name = "stm32f4_lab"
vendor = "STMicro"
family = "STM32F4"
platform_class = "edge_mcu"
data_width = 16
fraction = 8
overflow = "saturate"
rounding = "nearest"
max_freq_mhz = 168
notes = "STM32F407 on custom PCB with IMU sensor."
```

---

## Platform Discovery Hook

### Registering a Hook

A discovery hook is any Python callable that returns a list of
`HardwareProfile` instances:

```python
from sc_neurocore.compiler.advanced_features import (
    register_platform_hook, discover_platforms,
)
from sc_neurocore.compiler.hardware_profiles import HardwareProfile


def my_lab_discovery():
    """Auto-discover lab hardware via USB/serial enumeration."""
    # In production, this would scan connected devices
    return [
        HardwareProfile(
            name="lab_fpga_board_1",
            vendor="LabCustom", family="UltraScale+",
            platform_class="fpga", data_width=16, fraction=8,
            overflow="saturate", rounding="nearest",
            max_freq_mhz=300,
            notes="Auto-discovered FPGA board on USB port 3.",
        ),
    ]


# Register the hook
register_platform_hook(my_lab_discovery)

# Execute all registered hooks
discovered = discover_platforms()
print(f"Auto-discovered: {discovered}")
```

### Vendor SDK Integration

Hardware vendors can ship a Python package that auto-registers profiles:

```python
# In vendor_sdk/__init__.py
from sc_neurocore.compiler.advanced_features import register_platform_hook
from sc_neurocore.compiler.hardware_profiles import HardwareProfile


def _vendor_profiles():
    return [
        HardwareProfile(
            name="vendorx_nn_accel_v3",
            vendor="VendorX", family="NN-Accel-v3",
            platform_class="accelerator",
            data_width=8, fraction=4,
            overflow="saturate", rounding="nearest",
            max_freq_mhz=1000,
            notes="VendorX NN Accelerator v3 — 50 TOPS INT8.",
        ),
    ]

# Auto-register on import
register_platform_hook(_vendor_profiles)
```

Users then simply:

```python
import vendor_sdk  # auto-registers profiles

from sc_neurocore.compiler.advanced_features import discover_platforms
discover_platforms()  # VendorX profiles now available
```

---

## Workflow: TOML + Hook Together

```python
# 1. Load static profiles from config
from sc_neurocore.compiler.advanced_features import (
    load_profiles_from_toml,
    register_platform_hook, discover_platforms,
    generate_compilation_report,
)

load_profiles_from_toml("project_profiles.toml")

# 2. Discover runtime hardware
register_platform_hook(usb_scan_function)
discover_platforms()

# 3. Compile and report
report = generate_compilation_report(
    "my_snn", {"v": "-(v)/tau + I"}, "lab_fpga_board_1",
)
```

---

## Method 3: Auto-Construct from Constraints

`HardwareProfile.from_constraints()` is the **ultimate extensibility
mechanism**. Instead of manually defining every field, describe your
hardware constraints and let SC-NeuroCore auto-select the optimal
fixed-point configuration:

```python
from sc_neurocore.compiler.hardware_profiles import HardwareProfile

# Ultra-low-power sensor chip
p = HardwareProfile.from_constraints(
    "my_sensor_chip",
    vendor="SensorCo",
    platform_class="edge_mcu",
    max_power_budget_mw=5,      # Auto-selects 8-bit
    min_precision_bits=4,
    max_freq_mhz=100,
)
print(f"Auto-selected: Q{p.data_width}.{p.fraction}")

# High-performance AI accelerator
p2 = HardwareProfile.from_constraints(
    "my_ai_accel",
    vendor="AccelCo",
    platform_class="accelerator",
    data_width=32,              # Explicit override
    fraction=16,
    max_freq_mhz=2000,
    notes="Custom 32-bit AI accelerator.",
)
```

### Power-Aware Auto-Selection

The constructor uses power budget to select optimal data width:

| Power Budget | Auto-Selected Width |
|--------------|-------------------:|
| < 10 mW | 8-bit |
| 10–100 mW | 16-bit |
| > 100 mW | 16-bit (default) |

### When to Use Each Method

| Scenario | Method |
|----------|--------|
| Lab with known boards | TOML file |
| Vendor SDK integration | Discovery hook |
| Unknown future hardware | `from_constraints()` |
| One-off experiment | `from_constraints()` |
| CI/CD pipeline | TOML file |

## Further Reading

- [Compiler Intelligence Guide](compiler_intelligence.md) — all 67 features
- [Frontier Platforms Guide](frontier_platforms.md) — 31 platform classes
- [Hardware Profiles Guide](hardware_profiles.md) — all 175 profiles
- [Safety Certification Guide](safety_certification.md) — DO-254, IEC 61508
