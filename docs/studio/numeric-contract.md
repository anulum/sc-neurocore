<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->

# Hardware Numeric Contract

The equation compiler turns a catalogue model into fixed-point RTL. Every
parameter, constant, initial state value, the time step and every numeric
literal is stored as `round(value * 2**fraction)` in a signed word of the
chosen width. Nothing in that encoding raises an error:

- a value outside the format's range wraps and can change sign. In Q8.8 the
  AdEx capacitance `C = 200` becomes `-56`;
- a non-zero value below the format's resolution becomes zero. In Q8.8 a
  conductance of `0.000381` disappears.

In either case the RTL describes a different neuron. The hardware numeric
contract shows what the RTL actually holds, so Studio can offer only the
formats that keep the model intact.

## What the contract states

`sc_neurocore.compiler.hardware_numeric_contract.hardware_numeric_contract`
takes the neuron the compiler lowers and one Q-format. It returns:

- **Every encoded value.** For each value it gives what the neuron declares,
  what the RTL word represents, and a status: `exact`, `rounded`,
  `underflows_to_zero` or `out_of_range`.
  - Values are read the way the Verilog emitter reads them.
  - A literal divisor is stored as its reciprocal.
  - A modulo period is stored as it stands.
  - Integer exponents and floor divisors are not stored.
  - RK4 adds its `dt/2` and `dt/6` scales.
  - Exponential Euler adds the literals of its Jacobian terms.
- **Whether the format is representable.** It is not when any value falls out
  of range, or when a parameter, constant, initial state, step, literal
  divisor or modulo period rounds to zero. It is also not representable when
  a divisor built only from parameters evaluates to zero in the datapath. The
  contract computes such a divisor the way the RTL does. For example,
  AlphaNeuron's `(1/tau_v - 1/tau_inh)**2` is 0.0025 in the model and 0 at
  Q8.8, so the RTL would divide by zero. A plain literal below the
  resolution, such as a `1e-12` floating-point round-off guard, is reported
  but does not block, because the fixed-point comparison it guards is exact.
- **The look-up tables the datapath uses.** Each table comes with its sample
  grid. Arguments outside the grid are clamped to the first or last entry.
- **Timing.** Studio compiles without pipeline stages, so every state register
  and the spike output take their next value on each rising clock edge.
- **Whether a generated bit-true C kernel mirrors the RTL.** Where one does,
  the contract includes its arithmetic statement. Where none does, it says
  why, for example that the method is not Euler or map. The mirror rests on a
  finite Icarus Verilog co-simulation, which checks the stimuli it runs; it is
  not a proof.

The contract does not state:

- the range the state reaches during a run, since a run can still saturate or
  wrap;
- the approximation error of the look-up tables;
- timing closure, place and route, or board behaviour.

## How Studio uses it

Studio compiles at Q8.8 and Q16.16. These are the word geometries that every
Studio stage handles: the RTL compile, the bit-true kernel and the
co-simulation harness. A model's detail (`GET /api/models/{name}`) carries:

- `compile_configuration.q_formats`: only the formats the model is
  representable in, smallest word first;
- `default_q_format`: the first of those, or `null` when there is none;
- `numeric_contracts`: the contract of every candidate format, including the
  refused ones and the reason each was refused.

The RTL configuration panel lists each refused format together with its
reason.

A compile request is refused in two cases:

- its format is not one Studio compiles at;
- the neuron, with the requested parameter overrides, step and integrator, is
  not representable in that format.

The contract is checked against the values the request actually uses, not
against the defaults. An accepted compile records the contract it was built
under in its evidence, as `configuration.numeric_contract`.

At the time of writing, 23 catalogue models compile only at Q16.16. One model,
`SCClippedRationalRecoveryMapNeuron`, has a clip bound of `1e6` that no Studio
format can hold, so it is offered no format.

## Co-simulation

Co-simulation compares the Icarus Verilog RTL with the generated bit-true C
kernel, both at the model's contract. This comparison is named
`rtl_vs_bittrue`. The floating-point model is not part of it.

Each run has two parts. First comes the requested constant current. Then comes
a fixed stress schedule, run in the same simulation:

1. the most negative input word, from a reset neuron;
2. the most positive input word;
3. a reset in the middle of the run, followed by the requested input again;
4. zero input.

The first two phases drive the saturating commit in both directions. The
report's `bit_exact` holds only when both traces agree. `stress` records the
schedule, its trace digests and its first mismatch, if any. The current must
lie inside the format's range, because an input word outside it would wrap.

The report does not cover a state landing exactly on the threshold, which
needs a stimulus specific to the model, or any stimulus beyond the two
schedules.

Studio offers co-simulation for an integrator only when a bit-true kernel
mirrors the model compiled with it. The kernel mirrors explicit Euler and
discrete maps. It does not mirror:

- stochastic spike detection, because the RTL draws from an LFSR that the
  kernel does not model;
- macro-step sub-stepping.

## What the stress schedule found in the RTL

The first stress runs showed the RTL and the kernel disagreeing whenever a
value left the word. The kernel carries sums at full precision and commits
them through the overflow policy. The RTL now does the same:

- **Map right-hand sides.** A discrete map's next state is held at twice the
  word width before it saturates. Before this fix it was wrapped at the word
  width, so the saturating clamp could never act.
- **Reset values.** A reset value is committed through the overflow policy,
  saturating or wrapping. It is no longer assigned directly.
- **Division.** Numerator and divisor keep twice the word width.
- **Comparisons.** Comparisons, including those inside `abs`, `clip`, `max`
  and `min`, compare the unwrapped values.
- **Look-up tables.** An argument below a table's grid now clamps to the
  first entry. The sign extension used to be unsigned, so such an argument
  read the last entry instead.
- **Truth values.** A comparison used as a number (`w * (u > 0)`) is `1.0` or
  `0.0` in the format, as it is in the model, not one least-significant bit.
  `and` and `or` are lowered over comparisons.

The kernel gained the constructs the RTL already had:

- conditional expressions;
- floor division by a power of two;
- crossing detection, whose edge tracker starts at the condition's value on
  the initial state, as in the RTL.

The kernel now multiplies as unsigned 64-bit words and never shifts a
negative value left, because both are undefined in C.

With these changes, all 67 configurations Studio offers for co-simulation
agree bit for bit under the stress schedule. Each configuration is a model,
an integrator and a format.

The compiler refuses an expression whose unwrapped sum could exceed twice the
word width, because the datapath would then no longer match the kernel
exactly.

## Which silicon operations a model has here

`GET /api/models/{name}/capabilities` (`sc-neurocore.studio.model-capabilities.v1`)
states, for one catalogue model, which operations this installation can run,
and gives every disabled one its reason. The model panel shows it as a short
list under the model's badges.

| Operation | Enabled when | Disabled with a reason when |
| --- | --- | --- |
| compile | the model has a canonical schema with an executable profile, and a Studio format holds it | there is no such schema, or no format holds the model |
| co-simulate | a generated bit-true C kernel mirrors the RTL for at least one offered integrator and format, and `iverilog`, `vvp` and a C compiler are installed | no integrator is mirrored (every RK profile today), or a tool is missing |
| synthesise | co-simulation is enabled and Yosys is installed; synthesis of a selected model runs only on RTL whose co-simulation was bit-exact | either is missing |
| place and route | synthesis is enabled and the target's place-and-route tool is installed; it reports the design's maximum frequency | per target: no tool in the Studio flow (Gowin, Xilinx) or the tool is not installed |
| formal | never, in the Studio | the Studio has no route that runs a formal job; the model's catalogue job, when it has one, is named with what it asserts and does not establish |

Co-simulation is listed per integrator and format, because a kernel can mirror
one format and not another. Every combination the matrix enables for a map
model and an adaptive Euler model is executed by the test suite and must be
bit-exact; an RK model compiles and has co-simulation disabled by name.
