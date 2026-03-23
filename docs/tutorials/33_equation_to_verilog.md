<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->

# Tutorial 33: Equation-to-Verilog Compiler

SC-NeuroCore can compile arbitrary ODE strings into synthesizable Q8.8 fixed-point
Verilog RTL in a single function call. Write your neuron equations in Brian2-style
syntax, get both a Python simulation model and a Verilog module ready for FPGA.

## The Problem

Designing custom neuron models for FPGA requires:
1. Deriving fixed-point arithmetic for each equation
2. Writing Verilog with proper overflow handling
3. Verifying Python and Verilog produce identical results

The equation compiler automates all three steps.

## 1. One-Liner: ODE String to FPGA

```python
from sc_neurocore.compiler.equation_compiler import equation_to_fpga

# Define a simple LIF neuron as an ODE string
neuron, verilog = equation_to_fpga(
    equations="dv/dt = (-v + R*I) / tau : volt",
    threshold="v > 1.0",
    reset="v = 0.0",
    parameters={"R": 1.0, "tau": 20.0},
    module_name="custom_lif",
)

# `neuron` is a Python EquationNeuron — simulate it
for t in range(200):
    spike = neuron.step(I=0.8)
    if spike:
        print(f"  Spike at t={t}")

# `verilog` is synthesizable Q8.8 SystemVerilog
with open("custom_lif.sv", "w") as f:
    f.write(verilog)
print(f"Generated {len(verilog)} chars of Verilog")
```

## 2. Multi-Variable ODEs

The compiler handles coupled differential equations:

```python
# FitzHugh-Nagumo (2 variables)
neuron, verilog = equation_to_fpga(
    equations="""
    dv/dt = v - v**3/3 - w + I : volt
    dw/dt = 0.08 * (v + 0.7 - 0.8*w) : 1
    """,
    threshold="v > 1.0",
    reset="v = -1.0",
    parameters={},
    module_name="fitzhugh_nagumo",
)

# Simulate and plot
import numpy as np
voltages = []
for t in range(1000):
    neuron.step(I=0.5)
    voltages.append(neuron.state["v"])
```

```python
# Izhikevich (fast spiking)
neuron, verilog = equation_to_fpga(
    equations="""
    dv/dt = 0.04*v**2 + 5*v + 140 - u + I : volt
    du/dt = a * (b*v - u) : 1
    """,
    threshold="v >= 30.0",
    reset="v = c; u = u + d",
    parameters={"a": 0.1, "b": 0.2, "c": -65.0, "d": 2.0},
    module_name="izhikevich_fs",
)
```

## 3. The EquationNeuron Class

Build custom neurons from equations without the Verilog step:

```python
from sc_neurocore.neurons.equation_builder import from_equations

neuron = from_equations(
    equations="dv/dt = (-v + I) / tau : volt",
    threshold="v > 1.0",
    reset="v = 0.0",
    parameters={"tau": 10.0},
    dt=0.1,
)

# Same step()/reset() API as all 122 models
for t in range(500):
    spike = neuron.step(I=0.6)
```

## 4. Q8.8 Fixed-Point Arithmetic

The compiler converts floating-point equations to Q8.8 (16-bit signed,
8 fractional bits):

| Float value | Q8.8 representation | Range |
|-------------|---------------------|-------|
| 1.0 | 256 (0x0100) | |
| 0.5 | 128 (0x0080) | |
| -1.0 | -256 (0xFF00) | |
| Max | 127.996 | 32767 |
| Min | -128.0 | -32768 |
| Resolution | 1/256 = 0.00390625 | |

Multiplication: `(A * B) >> 8` with saturation.
The Verilog output includes explicit overflow clamping on every arithmetic operation.

## 5. AST-to-Verilog Expression Mapping

The compiler parses Python/Brian2 AST and emits Verilog:

| Python expression | Verilog output |
|---|---|
| `v + I` | `v_reg + I_in` |
| `v * 0.04` | `(v_reg * 16'd10) >>> 8` |
| `v ** 2` | `(v_reg * v_reg) >>> 8` |
| `v > 1.0` | `v_reg > 16'sd256` |
| `-v` | `(-v_reg)` |

## 6. Generate a Testbench

```python
from sc_neurocore.compiler.equation_compiler import generate_testbench

tb = generate_testbench(
    module_name="custom_lif",
    parameters={"R": 1.0, "tau": 20.0},
    n_steps=200,
    input_current=0.8,
)
with open("tb_custom_lif.sv", "w") as f:
    f.write(tb)
```

## 7. Full Pipeline: Equations to FPGA Bitstream

```bash
# 1. Generate Verilog from equations (Python)
python -c "
from sc_neurocore.compiler.equation_compiler import equation_to_fpga
_, v = equation_to_fpga('dv/dt = (-v + I) / 20 : volt',
    'v > 1.0', 'v = 0.0', {}, 'my_neuron')
open('my_neuron.sv', 'w').write(v)
"

# 2. Synthesize with Yosys
yosys -p "read_verilog -sv my_neuron.sv; synth_ice40 -top my_neuron; stat"

# 3. Place and route
nextpnr-ice40 --hx8k --json my_neuron.json --asc my_neuron.asc

# 4. Generate bitstream
icepack my_neuron.asc my_neuron.bin
```

## Supported ODE Features

| Feature | Supported | Example |
|---------|-----------|---------|
| Linear terms | Yes | `-v / tau` |
| Polynomial | Yes | `v**2`, `v**3` |
| Products | Yes | `a * b * v` |
| Addition/subtraction | Yes | `v - w + I` |
| Threshold comparison | Yes | `v > 1.0`, `v >= 30` |
| Multi-variable reset | Yes | `v = c; u = u + d` |
| Named parameters | Yes | `tau`, `a`, `b`, `c`, `d` |
| External input | Yes | `I` (injected per step) |

## Further Reading

- [Tutorial 09: Hardware Co-simulation](09_hardware_cosimulation.md) — verify Python vs Verilog
- [Tutorial 13: Fixed-Point Arithmetic](13_fixed_point_design.md) — Q8.8 details
- [API: Compiler](../api/compiler.md) — auto-generated API docs
- [Hardware Guide](../hardware/HARDWARE_GUIDE.md) — FPGA deployment
