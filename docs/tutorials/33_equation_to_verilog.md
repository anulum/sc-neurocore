<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SC-NeuroCore — Equation-to-Verilog compiler tutorial -->

# Tutorial 33: Equation-to-Verilog Compiler

SC-NeuroCore lowers validated `EquationNeuron` models to fixed-point Verilog.
The registered emitter owns state and clocking. The folded emitter exposes
state and selected parameters on ports so one processing element can be
time-multiplexed across a population.

The Python compiler is the maintained authority for this code-generation
surface. Historical Go, Julia, Mojo, and Rust-safety files were unwired,
non-executable generated stubs and have been removed; they were never valid
compiler implementations or benchmark peers.

## 1. Compile a registered neuron

```python
from sc_neurocore.compiler.equation_compiler import equation_to_fpga

neuron, verilog = equation_to_fpga(
    "dv/dt = (-v + R * I) / tau",
    threshold="v > 1.0",
    reset="v = 0.0",
    params={"R": 1.0, "tau": 20.0},
    init={"v": 0.0},
    dt=0.1,
    module_name="custom_lif",
)

for step in range(200):
    if neuron.step(I=25.0):
        print(f"spike at step {step}")

with open("custom_lif.v", "w", encoding="utf-8") as stream:
    stream.write(verilog)
```

The generated module has `clk`, active-low `rst_n`, fixed-point `I_t`,
`spike_out`, and one `<state>_out` port per state variable. Every emitted
registered and folded module starts with the repository's seven-line licence
header.

## 2. Choose an integration method

`equation_to_fpga()` builds the default simultaneous Euler model. Construct an
`EquationNeuron` directly when another maintained recurrence is required.

| `method` | Emitted recurrence | Constraint |
| --- | --- | --- |
| `euler` | Simultaneous forward Euler | Every derivative reads the committed pre-step state. |
| `gauss_seidel` | Sequential Euler | Later equations read already-updated earlier variables in declaration order. |
| `rk4` | Classical four-stage RK4 | All stages use the same fixed-point expression emitter. |
| `exp_euler` | Diagonal exponential Euler | Uses the symbolic diagonal Jacobian and the shared `exprel` LUT. |
| `map` | Direct discrete recurrence | Assigns `state[n+1] = f(state[n])` without a `dt` multiply. |

```python
from sc_neurocore.compiler.equation_compiler import compile_to_verilog
from sc_neurocore.neurons.equation_builder import EquationNeuron

neuron = EquationNeuron(
    equations={
        "v": "v - v * v * v / 3.0 - w + I",
        "w": "epsilon * (v + a - b * w)",
    },
    parameters={"a": 0.7, "b": 0.8, "epsilon": 0.08},
    state={"v": -1.0, "w": -0.5},
    threshold="v >= 1.0",
    detection="crossing",
    dt=0.1,
    method="rk4",
)

verilog = compile_to_verilog(neuron, module_name="sc_fhn_rk4")
```

Unsupported integration methods are rejected by `EquationNeuron` before
compilation.

## 3. Candidate, previous-state, and event semantics

Dynamics read the committed state at the beginning of an integration step.
Threshold and reset expressions read the integrated candidate. The reserved
`<state>_prev` alias reads the committed macro-boundary value.

This distinction supports wrapped maps and edge detection without conflating a
backward wrap with an upward threshold crossing:

```python
from sc_neurocore.neurons.equation_builder import EquationNeuron

period = 6.283185307179586
candidate = (
    "theta_prev + dt * ((1.0 - cos(theta_prev)) + "
    "(1.0 + cos(theta_prev)) * gain * I)"
)
theta = EquationNeuron(
    equations={
        "theta": (
            "(theta + dt * ((1.0 - cos(theta)) + "
            "(1.0 + cos(theta)) * gain * I)) % 6.283185307179586"
        )
    },
    parameters={"dt": 0.1, "gain": 1.0, "threshold": 3.141592653589793},
    state={"theta": 0.0},
    threshold=f"theta_prev < threshold <= ({candidate})",
    detection="crossing",
    dt=1.0,
    method="map",
)
```

A resetting model uses level-style commit logic even if its detection marker is
`crossing`. A non-resetting crossing model stores one threshold-history bit
and fires only on the inactive-to-active transition.

## 4. Fixed-point configuration

The default is signed Q8.8: 16 total bits, 8 fractional bits, range
`[-128, 127.99609375]`, and resolution `1/256`.

```python
verilog_q88 = compile_to_verilog(neuron, data_width=16, fraction=8)
verilog_q412 = compile_to_verilog(neuron, data_width=16, fraction=12)
verilog_q1616 = compile_to_verilog(neuron, data_width=32, fraction=16)
```

A non-zero `dt` that quantises to zero is rejected. Increase the fractional
width or choose a representable timestep. `dt=0.0` is a legal frozen-state
recurrence.

Both Verilog emitters currently require `signed=True`. Earlier unsigned output
mixed signed ports and expression arithmetic with unsigned next-state clamps;
that mode now fails closed instead of emitting an invalid UQ contract.

Overflow modes are explicit:

| Mode | Generated behaviour |
| --- | --- |
| `saturate` | Clamp candidate state to the selected word range. |
| `wrap` | Keep the low `data_width` bits. |
| `trap` | Keep the low bits and emit a simulation-only `$fatal` overflow check. |

Product rounding is explicit and signed:

| Mode | Generated integer behaviour |
| --- | --- |
| `truncate` | Arithmetic right shift; negative non-integral products round towards negative infinity. |
| `nearest` | Nearest code with exact half-way cases away from zero. |
| `bankers` | Nearest code with exact half-way cases to even. |
| `stochastic` | Rejected because neither public emitter owns a product-rounding LFSR. |

With `fraction=0`, multiplication results narrow directly without a rounding
bias. Stochastic model thresholds use their own explicit RNG contract; they do
not enable stochastic product rounding.

## 5. Expression lowering

| Equation form | Verilog role |
| --- | --- |
| `v + I` | State-register addition with `I_t`. |
| `v * 0.04` | Widened fixed-point multiply followed by the configured truncation. |
| `v ** 2` through `v ** 8` | Bounded integer-power multiply chain. |
| `v > 1.0` | Candidate-state comparison in threshold/reset context. |
| `a < b <= c` | Ordered conjunction retaining Python comparison order. |
| `x % 6.283185307179586` | Floored remainder for a finite positive literal divisor. |

Dynamic, non-finite, zero, negative, underflowed, or out-of-range modulo
divisors fail compilation. Unsupported AST nodes and function calls also fail
closed.

### Transcendental tables

The table geometry is explicit and shared with the integer C/Rust expression
lowerer:

| Family | Entries | Domain | Step |
| --- | ---: | --- | ---: |
| `exp`, `tanh`, `cosh`, `exprel`, `sigmoid`/`expit`, `sin`, `cos`, cube root | 256 | `[-16, 16)` | `1/8` |
| `log` | 256 | `[1/256, 8 + 1/256)` | `1/32` |
| `sqrt` | 16 | `[0, 7.5]` | `1/2` |

These geometries do not imply one model-independent error bound. Model-level
co-simulation defines the accepted state and event envelope.

## 6. Folded combinational datapaths

`compile_to_datapath()` emits the same arithmetic core without state
registers. State arrives on `<state>_reg` ports and leaves on
`<state>_next_out` ports.

```python
from sc_neurocore.compiler.equation_compiler import compile_to_datapath

pe = compile_to_datapath(
    neuron,
    module_name="sc_fhn_pe",
    param_ports=("a", "b", "epsilon"),
)
```

Named `param_ports` must exist in the model. Unlisted parameters remain module
parameters. A folded stochastic model receives an already-advanced
`rng_sample` port because the population scheduler owns one RNG state per
neuron.

Folded datapaths are combinational and do not accept multiplier pipeline
stages.

## 7. Registered pipelines and macro substeps

`pipeline_stages > 0` registers expression-multiply outputs. The registered
module exposes a constant `latency` port and holds recurrent state until the
pipeline has drained.

A model with `substeps > 1` advances state on each inner clock and evaluates a
single crossing at the macro boundary. This mode is restricted to non-resetting
crossing models without multiplier pipelining. Unsupported combinations raise
instead of emitting a recurrence that differs from the Python model.

Stochastic Poisson and escape-rate RTL currently requires a signed, unpipelined
registered datapath. The registered form owns a seeded 16-bit LFSR; the folded
form receives its sample from the scheduler.

## 8. Generate and compile a testbench

```python
from sc_neurocore.compiler.equation_compiler import generate_testbench

testbench = generate_testbench(
    neuron,
    module_name="sc_fhn_rk4",
    n_steps=200,
    input_current=0.8,
)
with open("tb_sc_fhn_rk4.v", "w", encoding="utf-8") as stream:
    stream.write(testbench)
```

Compile the real emitted files with Icarus Verilog:

```bash
iverilog -g2012 -o tb_sc_fhn sc_fhn_rk4.v tb_sc_fhn_rk4.v
vvp tb_sc_fhn
```

Repository co-simulation tests compare Python state/event sequences with
registered and folded RTL, including pipelines, macro substeps, stochastic
full-period streams, map semantics, and candidate-based resets:

```bash
python -m pytest \
  tests/test_bit_true_cosim.py \
  tests/test_cosim_emitters.py \
  tests/test_cosim_poisson.py \
  tests/test_folded_datapath.py \
  tests/test_pipeline_cosim.py
```

## 9. Compiler architecture

```text
verilog_compiler.py                 stable two-function facade
├── _verilog_registered_module.py  clocked state-owning module emission
└── _verilog_folded_datapath.py    combinational folded-PE emission
    both -> _verilog_neuron_core.py
            ├── _verilog_integrators.py -> verilog_expr_emitter.py
            └── verilog_expr_emitter.py
```

The dependency graph is acyclic. Architecture tests pin the two public
signatures, historical module identity, definition ownership, dependency
direction, and responsibility-specific size ceilings.

## 10. Measured local compiler timings

The committed `benchmarks/results/bench_verilog_compiler.json` artifact contains
25 rotating samples per workload, source hashes, output hashes, CPU affinity,
governor, and host load.

| Workload | Median compile time |
| --- | ---: |
| Euler registered | 0.114013 ms |
| Euler folded with parameter port | 0.301826 ms |
| RK4 registered | 0.401448 ms |
| Four-substep RK4 registered | 0.730636 ms |
| Escape-rate registered | 4.565140 ms |
| Square-root map registered | 0.260231 ms |
| Negative half-LSB nearest map registered | 0.136080 ms |

The run used all 12 logical CPUs as an allowed affinity mask, the `powersave`
governor, and a loaded workstation with another repository lane active. No core
was reserved. These measurements are local regression evidence, not
promotion-grade latency or throughput claims.

## 11. CLI and hardware profiles

The compiler CLI writes RTL and can invoke synthesis tools when installed:

```bash
sc-neurocore compile "dv/dt = -(v - E_L)/tau_m + I/C" \
  --threshold "v > -50" \
  --reset "v = -65" \
  --params "E_L=-65,tau_m=10,C=1" \
  --init "v=-65" \
  --target ice40 \
  --testbench \
  -o build/my_lif
```

Hardware profiles provide checked default widths, fractions, overflow modes,
and rounding modes. They are configuration inputs, not evidence that a design
meets timing, area, power, or certification requirements. Those claims require
a named synthesis/place-and-route flow and its retained reports.

## 12. Current boundaries

- The whole-neuron integer C/Rust generator mirrors only `euler` and `map`;
  it rejects other integrators.
- Registered and folded Verilog emission currently requires signed state.
- Folded datapaths do not support multiplier pipeline registers.
- Stochastic-threshold registered RTL is signed and unpipelined.
- Stochastic product rounding is rejected until the caller can provide and own
  a rounding RNG stream.
- A loaded-host compiler timing is not an FPGA timing, area, power, or energy
  measurement.
- Generated RTL must still pass the target toolchain, co-simulation envelope,
  and the project's hardware evidence gates.

## Further reading

- [Compiler API](../api/compiler.md)
- [Compiler Surface Policy](../api/compiler_surface.md)
- [Co-Simulation Guide](../guides/cosimulation_guide.md)
- [Precision Modes](../guides/precision_modes.md)
- [Hardware Guide](../hardware/HARDWARE_GUIDE.md)
