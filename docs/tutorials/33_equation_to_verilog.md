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

`exp_euler` differentiates each right-hand side symbolically. Every smooth
function of the vocabulary has its derivative in the same grammar (`tan`,
`arctan`, `arcsin`, `arccos`, `exp2`, `log2`, `log10`, `expm1`, `log1p` and
`erf` included, with their constant factors written as numbers). A dependence on
`abs`, `clip`, `min`, `max`, `floor`, `ceil`, `trunc` or `sign` is refused;
random draws do not depend on the state and stay as coefficients.

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
| `max(a, b, c)`, `min(a, b, c)` | Select chain over every argument (at least two), folded left. |
| `clip(x, lo, hi)` | Two selects; exactly three arguments. |
| `2 * pi * v` | `pi` is encoded as a literal unless a state variable or parameter is named `pi`. |

Dynamic, non-finite, zero, negative, underflowed, or out-of-range modulo
divisors fail compilation. Unsupported AST nodes and function calls also fail
closed, and so does a call whose arguments the Python model would not evaluate
as written: keyword arguments, `max`/`min` with one argument, `clip` without
both bounds, or a second argument to a one-argument function. The Verilog, C++
(HLS) and bit-true C/Rust lowerers share these rules
(`expr_lut_tables.FUNCTION_ARITY`, `check_call_arguments`, `NAMED_CONSTANTS`).
Before 2026-10-18 these calls were lowered silently: `max`/`min` used only
their first two arguments, and `clip(x)`/`max(x)` returned `x`.

### Transcendental tables

The table geometry is explicit and shared with the integer C/Rust expression
lowerer:

| Family | Entries | Domain | Step |
| --- | ---: | --- | ---: |
| `exp`, `tanh`, `cosh`, `exprel`, `sigmoid`/`expit`, `sin`, `cos`, cube root | 256 | `[-16, 16)` | `1/8` |
| `log` | 256 | `[1/256, 8 + 1/256)` | `1/32` |
| `sqrt` | 16 | `[0, 7.5]` | `1/2` |

Each table returns the function's value at the sample point at or below the
argument; there is no interpolation, and an argument outside the domain reads the
nearest endpoint. The approximation error therefore depends on the argument
range a model reaches. The bit-true co-simulation does not measure it, because
the C/Rust kernel reads the same tables; `sc_neurocore.compiler.lut_accuracy`
measures it against the Python model per function and format, the hardware
numeric contract reports it for every table a neuron uses, and
[the lut-v1 accuracy report](../validation/lut_accuracy_v1.md) lists it. Model-level co-simulation against the Python model defines the
accepted state and event envelope.

The index is `(x - lo) >> shift`, computed in a signed width that holds every
argument plus the table offset (`expr_lut_tables.lut_offset_width`): `data_width
+ 1` bits where that suffices, wider at formats whose integer range is narrower
than 16 (Q4.12, Q1.15, Q4.4, Q1.7). Before 2026-10-18 those formats wrapped the
offset, and both the RTL and the C kernel read wrong entries (for example
`exp(0) = 0` at Q4.12).

### Function lowering v2

`compile_to_verilog(..., function_lowering="v2")` (also `compile_to_datapath`,
`generate_bittrue_kernel_from_neuron` and `hardware_numeric_contract`) replaces
the tables with function units that reduce the argument and interpolate:

| Unit | Reduction and table |
| --- | --- |
| `exp` | `x·log2(e) = k + f`; `2^f` from a quadratic table on `[0, 1)`; shift by `k` |
| `log` | leading one of the input; `log2` of the mantissa on `[1, 2)`; times `ln 2` |
| `sqrt`, `x ** 0.5` | exact integer square root, rounded to nearest |
| `sin`, `cos` | argument in turns, reduced modulo one turn; quarter-wave table and quadrant symmetry |
| `sigmoid`/`expit`, `tanh` | table on `[0, 32)`, `σ(-x) = 1 - σ(x)`, `tanh x = 2σ(2x) - 1` |
| `cosh`, `sinh` | two exponentials, summed before one rounding |
| `exprel` | quartic series for `|x| < 1/8`, otherwise `(e^x - 1)/x` |
| `x ** (1/3)` | `2^(log2|x|/3)` with the sign of `x` |
| `exp2`, `log2`, `log10` | the `exp` and `log` reductions without the `log2(e)` and `ln 2` factors (`log10 = log2 · log10 2`) |
| `expm1`, `log1p` | the exponential minus an exact one; the logarithm of the exact integer `1 + x` |
| `tan` | sine and cosine of the reduced turn, divided and rounded once; saturated at the poles |
| `arctan` | quadratic table on `[0, 1]`, `arctan x = π/2 - arctan(1/x)` above 1 |
| `arcsin`, `arccos` | `arctan(x / sqrt(1 - x²))` with an exact square root; `arccos = π/2 - arcsin` |
| `erf` | table on `[0, 4)`, odd symmetry, 1 beyond |
| `floor`, `ceil`, `trunc`, `sign` | exact integer arithmetic, no table (also used by `lut-v1`) |

Each unit is written once as a 64-bit integer program
(`sc_neurocore.compiler.function_unit_programs`) and emitted as Verilog wires and
as C or Rust statements, so the RTL and the bit-true kernel are identical; both
equal the separately written integer reference `sc_neurocore.compiler.function_units`
(checked by Icarus, gcc with undefined-behaviour checks, and rustc with overflow
checks). The units accept words of at most 32 bits with
`data_width + fraction <= 61`; other formats are refused. `sinh`, `tan`,
`arctan`, `arcsin`, `arccos`, `exp2`, `log2`, `log10`, `expm1`, `log1p` and `erf`
exist only in `v2`; the exact `floor`, `ceil`, `trunc` and `sign` work in both
lowerings. These names are bound in the Python model to the NumPy functions of
the same name (`erf` to `scipy.special.erf`). `log` of a non-positive input is the most negative code, `sqrt` of a negative
input is 0, and `x ** (1/3)` of a negative input is `-(|x| ** (1/3))`; `log2`,
`log10` and `log1p` follow `log`, and `arcsin`/`arccos` of `|x| > 1` take their
values at `±1`. The Python model raises or returns NaN for these inputs.

At every code of the 16-bit formats every unit is correctly rounded (0.5 LSB),
except `cosh` and `sinh` (0.63 LSB) and `tan` (0.52 LSB). Near a pole of `tan` the
error follows the function's conditioning, about `(1 + tan² x)·2π·2^-30`: 2.4 LSB
at Q16.16 where `|tan x| <= 100`. At 32-bit formats the exponential family is
limited by the 30-bit internal fraction, a few LSB at the top of the range.
[The v2 accuracy report](../validation/function_units_v2_accuracy.md) lists every
unit and format. `lut-v1` remains the default and its RTL is unchanged; `v2` costs
more logic (64-bit intermediates and coefficient tables) and its synthesis cost is
not yet measured.

### Random draws

`rand()` (uniform on `[0, 1)`), `randn()` (standard normal) and `poisson(lam)`
(a Poisson count) may appear in equations, thresholds, reset rules and rate
expressions:

```python
from sc_neurocore.compiler.verilog_compiler import compile_to_verilog
from sc_neurocore.neurons.equation_builder import EquationNeuron

neuron = EquationNeuron(
    equations={"v": "-v + I + 0.5*randn()"},
    state={"v": 0.0},
    dt=0.1,
    threshold="v > 1.0",
    reset={"v": "rand() - 0.5"},
    rng_seed=2026,
    random_stream=3,
)
assert neuron.equations == {"v": "-v + I + 0.5*randn(0)"}
verilog = compile_to_verilog(neuron, "noisy_lif", 16, 8)
```

Every call draws one Philox4x32-10 block at counter `(site, step, stream)` with
the 64-bit `rng_seed` as key. The neuron numbers each bare call when it is
built (`rand()` becomes `rand(<site>)`, `poisson(lam)` becomes
`poisson(lam, <site>)`), so the Python model and the compiler read the same
site; a literal written by hand is kept and may not repeat.
`step` advances once per `EquationNeuron.step` and returns to 0 on `reset()`;
`random_stream` separates instances that share a seed. A draw is a function of
its counter only, so a site evaluated twice in one step returns the same number.

In RTL the module gains parameters `P_RNG_SEED` and `P_RNG_STREAM` (defaults
from the neuron) and a 64-bit `rng_step` register, cleared by reset and
advanced on every step. The bit-true C and Rust kernels carry `rng_step` in
their state and the constants `SC_RNG_SEED` and `SC_RNG_STREAM`. `rand()` is
`floor(rand·2^fraction)` of the model's value exactly; `randn()` is an integer
Box–Muller (`sqrt(-2 ln u1)·cos(2π u2)` with the v2 logarithm, square root and
quarter-wave cosine; near `u1 = 1`, where the radius is steep, `-ln u1` is
summed as a series), saturated to the word. Its code is within half an LSB plus
the transform's own error, `NORMAL_TRANSFORM_ERROR = 2^-24` (measured, largest
seen 3.1·10^-8), of the model's value clipped to the word, so at most
`0.5 + 2^(fraction-24)` LSB: about 0.504 LSB at a 16-bit fraction, but up to
32.5 LSB at Q3.29; a high fraction gains resolution, not accuracy. `poisson(lam)` inverts the distribution on word 0: the
count is the number of thresholds `ceil(2^32·CDF(k))` at or below the word, so
the model, the RTL and the kernel return the same count, the exact inversion of
a 32-bit uniform. Its mean must be a constant in `[0, 100]` in compiled
expressions (the thresholds, at most 168, are built with the unit); the Python
model also accepts a variable mean for `poisson()`. A variable mean compiles as
`poisson_grid(lam)`: the mean (any expression, evaluated as a word) is floored
to a multiple of 1/16 and clamped to `[0, 16]`, and word 0 is compared with the
exact thresholds of that grid mean, held in tables of 257 rows (47 thresholds,
packed three to a row); the model applies the same grid, so the count is the
exact inversion for the grid mean in the model, the RTL and both kernels. All of
them are built from the same integer programs in either function lowering, and
the RTL equals the kernel bit for bit.

A folded processing element (`compile_to_datapath`) takes the step counter and
stream on the inputs `rng_step` and `P_RNG_STREAM` (with the parameter
`P_RNG_SEED`), so the sequencer that streams a neuron's state also supplies
its counter; given the same values it draws exactly what the per-instance
module draws. In a pipelined module `rng_step` advances on the valid cycle, so
the draws stay fixed while the multiply stages fill and each logical step
draws what the unpipelined module draws; in a multi-substep module it advances
on the macro boundary, so every sub-step of a macro step draws the same
numbers, as the Python model does. The HLS C++ lowering refuses draws.

### Graded spikes

`spike_value` makes a spike carry a value, as the graded spikes of Loihi 2 and
the `graded` processes of Lava do:

```python
from sc_neurocore.neurons.equation_builder import EquationNeuron

neuron = EquationNeuron(
    equations={"v": "v + I"},
    state={"v": 0.0},
    dt=1.0,
    method="map",
    threshold="abs(v) > 0.5",
    reset={"v": "0.0"},
    spike_value="v",
)
```

On a spike the expression is evaluated where the reset rules are (the
integrated state before reset, with `<var>_prev` for the previous state) and
kept as `neuron.last_spike_value`; on other steps that value is 0. The
registered RTL adds the output `spike_value_out` (a word of the data format,
committed under the overflow policy like a reset value and registered with
`spike_out`), the folded processing element drives it combinationally, and the
bit-true C and Rust kernels keep it in `spike_value_out` of their state. A
graded spike needs a threshold or a stochastic spike mechanism, and a state
named `spike_value` is refused with it (its output would be `spike_value_out`).
In the multi-substep datapath the value is set on the macro boundary from the
state after the last sub-step. The C and Rust kernels evaluate the value and the
reset rules only on a spike, as the model does, so an expression undefined off
the spike (a reciprocal of a zero state) does not trap them.

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
neuron. A crossing-detection model without reset (rising-edge spikes) needs
the condition of the previous step: the PE takes it on `thr_prev_in` and
returns the new one on `thr_cond_out`, which the caller stores per neuron and
resets to `neuron.initial_threshold_active()`, as the module's `_thr_prev`
register does.

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
  tests/test_bit_true_cosim_bit_true_cosim.py \
  tests/test_bit_true_cosim_rust_matches_c.py \
  tests/test_cosim_emitters_co_simulation.py \
  tests/test_cosim_emitters_exp_euler_emitter.py \
  tests/test_cosim_emitters_macro_step_substep_emitter.py \
  tests/test_cosim_emitters_multi_precision.py \
  tests/test_cosim_emitters_rk4_emitter.py \
  tests/test_cosim_poisson.py \
  tests/test_folded_datapath.py \
  tests/test_pipeline_cosim.py \
  tests/test_lut_index_offset_width.py \
  tests/test_verilog_expr_emitter_calls.py \
  tests/test_function_lowering_v2_cosim.py
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

`--data-width` and `--fraction` set the format (also of the testbench),
`--method` the integrator, `--function-lowering lut-v1|v2` the transcendental
lowering, `--rng-seed` and `--random-stream` the key and stream of `rand()`,
`randn()` and `poisson()`, and `--spike-value` a graded spike output.

Hardware profiles provide checked default widths, fractions, overflow modes,
and rounding modes. They are configuration inputs, not evidence that a design
meets timing, area, power, or certification requirements. Those claims require
a named synthesis/place-and-route flow and its retained reports.

## 12. Current boundaries

- The whole-neuron integer C/Rust generator mirrors every integrator
  (`euler`, `map`, `rk4`, `exp_euler`, `gauss_seidel`) and the macro steps of
  sub-stepped crossing models; it does not mirror stochastic spike detection
  (the RTL's LFSR) and refuses the rounding modes the RTL alone supports.
- The integrator scalings (`dt`, the RK4 stage factors, `dt/6`) round under the
  format's rounding mode like every product; before 2026-10-19 the RTL
  truncated them while the kernel rounded the Euler `dt` product.
- Registered and folded Verilog emission currently requires signed state.
- The diffusion-noise symbol `xi` has no RTL or kernel lowering; an expression
  using it, the graded value included, is refused. `randn()` is the
  counter-based normal draw that compiles.
- Folded datapaths do not support multiplier pipeline registers.
- Stochastic-threshold registered RTL is signed and unpipelined.
- Stochastic product rounding is rejected until the caller can provide and own
  a rounding RNG stream.
- `rand()`, `randn()` and `poisson()` compile to registered modules (also
  pipelined or multi-substep), folded processing elements and the bit-true
  kernels; a compiled `poisson()` needs a constant mean, and
  `poisson_grid()` takes a variable one on the 1/16 grid. The folded
  network's neuron templates have no draws, and its sequencer does not yet
  assign step counters and streams.
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
