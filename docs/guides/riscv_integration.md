<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->

# RISC-V SoC Integration

Integrate SC-NeuroCore compiled neurons into RISC-V-based SoCs with
auto-generated MMIO drivers for bare-metal, FreeRTOS, and Zephyr RTOS
environments.  This guide covers the complete integration pipeline from
driver generation through register map documentation to real-time tick
task implementation on PolarFire SoC and Efinix Titanium targets.

---

## 1. Mathematical Formalism

### 1.1 MMIO Register Map Layout

The neuron peripheral occupies a contiguous MMIO address space starting
at `BASE_ADDRESS` (default `0x40000000`).  The register map follows a
fixed 32-bit-aligned layout:

$$
\text{REG}[i] = \text{BASE} + 4i, \quad i \in \{0, 1, 2, \ldots, N_{\text{params}} + 2\}
$$

| Offset | Register | Width | R/W | Description |
|--------|----------|:-----:|:---:|-------------|
| `+0x00` | `CTRL` | 32 | RW | Control: bit 0 = enable, bit 1 = reset |
| `+0x04` | `I_T` | 32 | WO | Input current (Q-format) |
| `+0x08` | `SPIKES` | 32 | RO | Spike output register |
| `+0x0C` | `P0` | 32 | WO | Parameter 0 (e.g. `E_L`) |
| `+0x10` | `P1` | 32 | WO | Parameter 1 (e.g. `tau_m`) |
| ... | ... | ... | ... | ... |

### 1.2 Fixed-Point Encoding

Parameters and inputs are encoded in Q$m$.$f$ format before writing
to MMIO registers:

$$
Q(x) = \text{round}(x \cdot 2^f)
$$

The C encoder function:

```c
static inline int32_t encode(float v) {
    return (int32_t)(v * (1 << FRAC));
}
```

### 1.3 Real-Time Tick Period

For a neuron with time step $\Delta t$ and target simulation speed
$S$ (real-time ratio), the tick task period is:

$$
T_{\text{tick}} = \frac{\Delta t}{S}
$$

For $\Delta t = 0.1$ ms at real-time ($S = 1$):
$T_{\text{tick}} = 100\ \mu\text{s}$, requiring a 10 kHz timer interrupt.

### 1.4 Latency Budget

The total latency from input change to spike output detection:

$$
L_{\text{total}} = L_{\text{MMIO\_write}} + L_{\text{neuron}} + L_{\text{MMIO\_read}} + L_{\text{ISR}}
$$

Typical values on PolarFire SoC at 250 MHz:

| Component | Latency |
|-----------|:-------:|
| MMIO write (AXI4-Lite) | 2–4 cycles |
| Neuron compute | 1–8 cycles (depends on pipeline) |
| MMIO read | 2–4 cycles |
| ISR entry (RISC-V) | ~20 cycles |
| **Total** | **~30 cycles = 120 ns** |

---

## 2. Architecture

### 2.1 System-Level Integration

```
┌───────────────────────────────────────────────────────────┐
│  RISC-V SoC (PolarFire / Efinix Titanium)                │
│                                                           │
│  ┌────────────┐    AXI4-Lite     ┌──────────────────┐    │
│  │ RISC-V     │◄════════════════►│ Neuron Peripheral │    │
│  │ Core       │    0x40000000    │ (compiled LIF)    │    │
│  │ (rv64gc)   │                  │                    │    │
│  └─────┬──────┘                  │  CTRL | I_T        │    │
│        │                         │  SPIKES | params   │    │
│        │ SPI/I2C                 └──────────────────┘    │
│        ▼                                                  │
│  ┌────────────┐                  ┌──────────────────┐    │
│  │ Sensor     │                  │ Spike Monitor    │    │
│  │ Interface  │                  │ (GPIO / IRQ)     │    │
│  └────────────┘                  └──────────────────┘    │
└───────────────────────────────────────────────────────────┘
```

### 2.2 Driver Architecture

The generated C driver provides three abstraction levels:

```
┌──────────────────────────────────────┐
│ Application Layer                     │
│ • neuron_tick() — periodic update    │
│ • neuron_process_spikes()            │
├──────────────────────────────────────┤
│ RTOS Layer (optional)                │
│ • FreeRTOS task + timer              │
│ • Zephyr work queue + k_timer        │
├──────────────────────────────────────┤
│ HAL Layer                            │
│ • MMIO_WR(addr, val)                 │
│ • MMIO_RD(addr)                      │
│ • encode() / decode()                │
└──────────────────────────────────────┘
```

---

## 3. Supported Platforms

### 3.1 Target SoCs

| Platform | RISC-V Core | FPGA Fabric | Typical Use |
|----------|-------------|-------------|-------------|
| Microchip PolarFire SoC | SiFive U54 (rv64gc) × 4 | PolarFire (28nm) | Edge AI, safety-critical |
| Efinix Titanium Ti375 | VexRiscv (rv32im) | Titanium (16nm) | Ultra-low-power edge |
| Gowin GW5A | Gowin eMCU (rv32i) | Gowin 28nm | Cost-sensitive IoT |
| Lattice CrossLink-NX | RISC-V softcore | CrossLink-NX | Sensor bridge |

### 3.2 RTOS Templates

| RTOS | Template | Timer API | Task Model |
|------|----------|-----------|------------|
| Bare-metal | Polling loop | SysTick / MTIMER | Main loop |
| FreeRTOS | `xTaskCreate` + `vTaskDelay` | `pdMS_TO_TICKS()` | Preemptive task |
| Zephyr | `k_work_submit` + `k_timer` | `K_MSEC()` | Work queue |

---

## 4. Python API

### 4.1 Generate RISC-V Driver

```python
from sc_neurocore.compiler.deployment import generate_riscv_driver

driver = generate_riscv_driver(
    "sc_lif",
    params={"E_L": 16, "tau_m": 16, "C": 16},
    base_address=0x40000000,
    data_width=16,
    fraction=8,
    rtos="baremetal",
)

with open("sc_lif_riscv.h", "w") as f:
    f.write(driver)
```

### 4.2 FreeRTOS Driver

```python
driver = generate_riscv_driver(
    "sc_lif",
    params={"E_L": 16, "tau_m": 16, "C": 16},
    rtos="freertos",
)
```

This generates:
- Standard MMIO accessors (`enable`, `disable`, `reset`, `set_current`)
- Per-parameter setters (`set_e_l`, `set_tau_m`, `set_c`)
- A FreeRTOS tick task with `xTaskCreate` boilerplate
- Start function `sc_lif_start_rtos()`

### 4.3 Zephyr Driver

```python
driver = generate_riscv_driver(
    "sc_lif",
    params={"E_L": 16, "tau_m": 16},
    base_address=0x80000000,
    rtos="zephyr",
)
```

This generates:
- Zephyr-style `k_work` queue integration
- `k_timer` periodic tick at 1 ms
- DTS (Device Tree Source) compatible base address

### 4.4 Full Integration Example

```python
from sc_neurocore.neurons.equation_builder import from_equations
from sc_neurocore.compiler.equation_compiler import compile_to_verilog
from sc_neurocore.compiler.deployment import (
    generate_riscv_driver,
    generate_constraints,
    estimate_resources,
)

# 1. Define and compile neuron
neuron = from_equations(
    "dv/dt = -(v - E_L)/tau_m + I/C",
    threshold="v > -50", reset="v = -65",
    params=dict(E_L=-65, tau_m=10, C=1),
    init=dict(v=-65),
)
verilog = compile_to_verilog(neuron, module_name="sc_lif")

# 2. Generate driver for PolarFire SoC with FreeRTOS
driver = generate_riscv_driver(
    "sc_lif",
    params={"E_L": 16, "tau_m": 16, "C": 16},
    rtos="freertos",
)

# 3. Generate constraints
constraints = generate_constraints("sc_lif", freq_mhz=250)

# 4. Write all artefacts
with open("sc_lif.v", "w") as f:
    f.write(verilog)
with open("sc_lif_riscv.h", "w") as f:
    f.write(driver)
with open("sc_lif.xdc", "w") as f:
    f.write(constraints)
```

---

## 5. CLI Usage

### 5.1 Generate Driver Artefacts

```bash
# Generate RISC-V driver header
python -c "
from sc_neurocore.compiler.deployment import generate_riscv_driver
d = generate_riscv_driver(
    'sc_lif',
    params={'E_L': 16, 'tau_m': 16, 'C': 16},
    rtos='freertos',
)
open('sc_lif_riscv.h', 'w').write(d)
print(f'Generated: sc_lif_riscv.h ({len(d)} bytes)')
"
```

### 5.2 Build for PolarFire SoC

```bash
# Cross-compile with RISC-V GCC
riscv64-unknown-elf-gcc \
    -march=rv64gc -mabi=lp64d \
    -I. -include sc_lif_riscv.h \
    -o main.elf main.c \
    -T polarfire_soc.ld
```

### 5.3 Build for Efinix Titanium (rv32im)

```bash
riscv32-unknown-elf-gcc \
    -march=rv32im -mabi=ilp32 \
    -I. -include sc_lif_riscv.h \
    -o main.elf main.c
```

---

## 6. Generated Driver Structure

### 6.1 Bare-Metal Driver Example

```c
/* Auto-generated RISC-V driver for sc_lif */
/* SC-NeuroCore — RISC-V SoC integration (baremetal) */

#ifndef SC_LIF_RISCV_H
#define SC_LIF_RISCV_H

#include <stdint.h>

#define SC_LIF_BASE    0x40000000U
#define SC_LIF_FRAC    8
#define SC_LIF_CTRL    (SC_LIF_BASE + 0x00)
#define SC_LIF_I_T     (SC_LIF_BASE + 0x04)
#define SC_LIF_SPIKES  (SC_LIF_BASE + 0x08)
#define SC_LIF_E_L     (SC_LIF_BASE + 0x0C)
#define SC_LIF_TAU_M   (SC_LIF_BASE + 0x10)
#define SC_LIF_C       (SC_LIF_BASE + 0x14)

#define MMIO_WR(a,v) (*(volatile uint32_t*)(a) = (v))
#define MMIO_RD(a)   (*(volatile uint32_t*)(a))

static inline int32_t sc_lif_encode(float v) {
    return (int32_t)(v * (1 << SC_LIF_FRAC));
}

static inline void sc_lif_enable(void)  { MMIO_WR(SC_LIF_CTRL, 0x01); }
static inline void sc_lif_disable(void) { MMIO_WR(SC_LIF_CTRL, 0x00); }

static inline void sc_lif_reset(void) {
    MMIO_WR(SC_LIF_CTRL, 0x02);
    MMIO_WR(SC_LIF_CTRL, 0x01);
}

static inline void sc_lif_set_current(float I) {
    MMIO_WR(SC_LIF_I_T, (uint32_t)sc_lif_encode(I));
}

static inline uint32_t sc_lif_get_spikes(void) {
    return MMIO_RD(SC_LIF_SPIKES);
}

static inline void sc_lif_set_e_l(float v) {
    MMIO_WR(SC_LIF_E_L, (uint32_t)sc_lif_encode(v));
}

static inline void sc_lif_set_tau_m(float v) {
    MMIO_WR(SC_LIF_TAU_M, (uint32_t)sc_lif_encode(v));
}

static inline void sc_lif_set_c(float v) {
    MMIO_WR(SC_LIF_C, (uint32_t)sc_lif_encode(v));
}

#endif /* SC_LIF_RISCV_H */
```

### 6.2 FreeRTOS Tick Task Extension

```c
/* ── FreeRTOS neuron tick task ───────────────────── */
#include "FreeRTOS.h"
#include "task.h"

static void sc_lif_tick(void *p) {
    (void)p;
    sc_lif_reset();
    sc_lif_enable();
    for (;;) {
        float I = read_sensor_current();
        sc_lif_set_current(I);
        uint32_t spikes = sc_lif_get_spikes();
        if (spikes) process_spike_event(spikes);
        vTaskDelay(pdMS_TO_TICKS(1));
    }
}

static inline void sc_lif_start_rtos(void) {
    xTaskCreate(sc_lif_tick, "neuron", 256, NULL, 3, NULL);
}
```

### 6.3 Zephyr Work Queue Extension

```c
/* ── Zephyr neuron tick ─────────────────────────── */
#include <zephyr/kernel.h>

static struct k_work neuron_work;
static struct k_timer neuron_timer;

static void sc_lif_work_handler(struct k_work *work) {
    float I = read_sensor_current();
    sc_lif_set_current(I);
    uint32_t spikes = sc_lif_get_spikes();
    if (spikes) process_spike_event(spikes);
}

static void sc_lif_timer_handler(struct k_timer *timer) {
    k_work_submit(&neuron_work);
}

static inline void sc_lif_start_zephyr(void) {
    k_work_init(&neuron_work, sc_lif_work_handler);
    k_timer_init(&neuron_timer, sc_lif_timer_handler, NULL);
    sc_lif_reset();
    sc_lif_enable();
    k_timer_start(&neuron_timer, K_MSEC(1), K_MSEC(1));
}
```

---

## 7. Performance Characteristics

### 7.1 Latency by Platform

| Platform | Clock | MMIO Latency | Neuron Compute | Total |
|----------|:-----:|:------------:|:--------------:|:-----:|
| PolarFire SoC | 250 MHz | 16 ns | 4–32 ns | ~50 ns |
| Efinix Titanium | 150 MHz | 27 ns | 7–53 ns | ~80 ns |
| Gowin GW5A | 100 MHz | 40 ns | 10–80 ns | ~120 ns |

### 7.2 Resource Overhead

| Component | LUTs | FFs | BRAM | Notes |
|-----------|:----:|:---:|:----:|-------|
| AXI4-Lite slave | ~200 | ~150 | 0 | Bus interface |
| Register file | ~50 | ~100 | 0 | CTRL + params |
| Neuron (LIF Q8.8) | ~80 | ~30 | 0 | Compute datapath |
| **Total** | **~330** | **~280** | **0** | < 1% of PolarFire |

### 7.3 Interrupt vs Polling Decision Matrix

| Criterion | Polling | Interrupt-Driven |
|-----------|:-------:|:----------------:|
| Latency | Lower (no ISR) | Higher (+20 cycles) |
| CPU utilisation | 100% (dedicated core) | ~1% per neuron |
| Determinism | Better | Jitter from ISR |
| Multi-neuron | Wasteful | Efficient |
| Recommended | Single-neuron, real-time | Multi-neuron, mixed workload |

---

## 8. Test Suite and Verification

### 8.1 Driver Generation Test

```bash
python -c "
from sc_neurocore.compiler.deployment import generate_riscv_driver

for rtos in ['baremetal', 'freertos', 'zephyr']:
    d = generate_riscv_driver('test_neuron', {'v_th': 16}, rtos=rtos)
    assert 'MMIO_WR' in d
    assert 'MMIO_RD' in d
    assert 'test_neuron' in d.lower() or 'TEST_NEURON' in d
    if rtos == 'freertos':
        assert 'xTaskCreate' in d
    elif rtos == 'zephyr':
        assert 'k_work' in d
    print(f'{rtos}: PASS ({len(d)} bytes)')
"
```

### 8.2 Register Map Consistency Test

```bash
python -c "
from sc_neurocore.compiler.deployment import generate_riscv_driver
d = generate_riscv_driver('sc_lif', {'E_L': 16, 'tau_m': 16, 'C': 16})

# Verify register offsets are sequential and aligned
import re
offsets = [int(m, 16) for m in re.findall(r'0x([0-9A-F]+)\)', d)]
for i in range(1, len(offsets)):
    assert offsets[i] == offsets[i-1] + 4, f'Gap at offset {offsets[i]:#x}'
print('Register map alignment: PASS')
"
```

### 8.3 Multi-Parameter Driver Test

```bash
python -c "
from sc_neurocore.compiler.deployment import generate_riscv_driver
d = generate_riscv_driver('sc_izh', {
    'a': 16, 'b': 16, 'c': 16, 'd': 16, 'v_peak': 16
})
assert 'SC_IZH_A' in d
assert 'SC_IZH_B' in d
assert 'SC_IZH_C' in d
assert 'SC_IZH_D' in d
assert 'SC_IZH_V_PEAK' in d
# Verify 5 parameter setters generated
assert d.count('set_') >= 5
print(f'Multi-param: PASS ({d.count(chr(10))} lines)')
"
```

### 8.4 Base Address Customisation Test

```bash
python -c "
from sc_neurocore.compiler.deployment import generate_riscv_driver
d = generate_riscv_driver(
    'sc_lif', {'E_L': 16},
    base_address=0x80002000,
)
assert '0x80002000' in d
print('Custom base: PASS')
"
```

### 8.5 E2E Pipeline Test

```bash
python -m pytest tests/e2e/test_e2e_pipeline.py -v -k "riscv"
```

### 8.6 Multi-Neuron SoC Pattern

For SoCs with multiple neuron peripherals, instantiate each at a
different base address:

```python
from sc_neurocore.compiler.deployment import generate_riscv_driver

bases = [0x40000000, 0x40001000, 0x40002000, 0x40003000]
for i, base in enumerate(bases):
    d = generate_riscv_driver(
        f"sc_neuron_{i}",
        params={"v_th": 16},
        base_address=base,
        rtos="freertos",
    )
    with open(f"sc_neuron_{i}_riscv.h", "w") as f:
        f.write(d)
```

### 8.7 Zephyr Device Tree Integration

For Zephyr targets, the neuron peripheral should appear in the
Device Tree Source (DTS):

```dts
/ {
    soc {
        sc_neuron: sc-neuron@40000000 {
            compatible = "sc-neurocore,neuron";
            reg = <0x40000000 0x100>;
            interrupts = <5 1>;
            status = "okay";
        };
    };
};
```

### 8.8 Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| No spikes detected | Neuron not enabled | Call `enable()` after `reset()` |
| Wrong Q-format values | Fraction mismatch | Verify `FRAC` matches compiled neuron |
| Bus timeout | Wrong base address | Check SoC address map against linker script |
| ISR not firing | Interrupt not connected | Check DTS / PLIC configuration |
| Spike glitch on reset | Missing reset sequence | Use `reset()` which writes 0x02 then 0x01 |

---

## References

1. **RISC-V ISA specification:**
   Waterman, A. & Asanović, K. "The RISC-V Instruction Set Manual."
   Volume I: Unprivileged ISA, v20191213, 2019.

2. **PolarFire SoC FPGA architecture:**
   Microchip Technology. "PolarFire SoC FPGA Family Datasheet."
   DS0200, 2023.

3. **FreeRTOS real-time kernel:**
   Barry, R. "Mastering the FreeRTOS Real Time Kernel."
   v10.4.6, Real Time Engineers Ltd, 2022.

4. **Zephyr RTOS documentation:**
   Zephyr Project. "Zephyr Project Documentation."
   https://docs.zephyrproject.org/, 2024.

5. **AXI4-Lite protocol specification:**
   Arm Ltd. "AMBA AXI and ACE Protocol Specification."
   ARM IHI 0022E, 2013.

6. **Efinix Titanium FPGA:**
   Efinix Inc. "Titanium FPGA Family Datasheet."
   DSN-0019, 2023.

---

## Further Reading

- [SoC Integration Guide](soc_integration_guide.md) — AXI4-Lite, Wishbone, bus interfaces
- [Deployment Guide](deployment_guide.md) — Constraints, bitstream automation
- [Hardware Profiles Guide](hardware_profiles.md) — 175 platform profiles
- [Formal Verification Guide](formal_verification.md) — SVA, SymbiYosys
- [Pipeline & Adaptive Precision Guide](pipeline_adaptive_precision.md) — Pipeline stages
- [Network Compilation Guide](network_compilation.md) — BRAM arrays, weight ROM
