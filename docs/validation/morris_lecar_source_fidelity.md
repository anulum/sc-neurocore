<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
# Morris-Lecar source and implementation fidelity

`MorrisLecarNeuron` implements the two-state calcium-potassium conductance
oscillator attributed to Morris and Lecar (1981), DOI
`10.1016/S0006-3495(81)84782-0`, using the maintained type-II profile. The
continuous source identity is preserved; there is no separate SC recurrence to
delete, merge, or count as another literature model.

## Numerical specialization

The repository advances the continuous `(V, W)` equations with a fixed
`dt = 0.1 ms` candidate-first classical RK4 step. An event is an observed upward
crossing of `V >= 0`; the oscillator has no reset. RK4, the observation grid,
and the crossing convention are implementation choices and are not attributed
to the continuous paper.

## Runtime evidence

Python, the production Rust/PyO3 engine and NetworkRunner, standalone safety
Rust, Go, Julia, and executable Mojo carry the real RK4 recurrence. The stable
cross-libm contract is exact events and bounded complete state, not bitwise
floating-point identity. All maintained language kernels reproduce 0, 3, and 5
events at currents 0, 50, and 100 over 2,000 steps.

The source-hashed measured packet at
`benchmarks/results/local_python_2026-06-17_morris_lecar_rk4.json` executes
Python, production Rust, Go, Julia, and Mojo for 200,000 steps at current 100.
Every row records 476 events and the complete final states agree within
`4e-8`. Its timings are non-isolated local regression context; no production
speed or hardware measurement is claimed.

The compact receipt at
`src/sc_neurocore/neurons/reference_receipts/morris_lecar_1981.json` replays
3,000 steps at current 100, records the seven exact event indices, final `(V,W)`
state, and a binary trace digest. The independent feature receipt
`morris_lecar_driven_oscillation_doi.json` separately re-derives the RK4 feature
map.

## Hardware boundary

The committed `sc_morris_lecar.v` is exactly the current equation compiler's
signed-Q16.16 lowering. Icarus/VVP preserves all seven receipt events, the
object passes Yosys coarse synthesis, and `sc_morris_lecar.sby` passes depth-4
bounded reset-safety checking through public ports under an initial-reset plus
fixed `I=100` receipt protocol. The protocol is explicit because unconstrained
32-bit input cones do not strengthen the reset property and make the
transcendental BMC impractically slow.

This is the H2 terminal boundary for the present unit. It is not formal
equation equivalence, unrestricted real-number parity, timing closure, PPA,
target-device validation, FPGA/board execution, physical silicon, or broad
biological validation.
