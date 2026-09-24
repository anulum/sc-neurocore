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
  divisor or modulo period rounds to zero. A plain literal below the
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

At the time of writing, 23 catalogue models need Q16.16. One model,
`SCClippedRationalRecoveryMapNeuron`, has a clip bound of `1e6` that no Studio
format can hold, so it is offered no format.
