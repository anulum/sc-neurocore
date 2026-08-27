<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
# FitzHugh-Nagumo source and implementation fidelity

`FitzHughNagumoNeuron` implements a transformed form of FitzHugh's (1961)
Bonhoeffer-van der Pol system, DOI `10.1016/S0006-3495(61)86902-6`. The source
identity is preserved; there is no separate SC recurrence to delete, merge, or
count as another literature model.

## Source transformation and numerical specialization

FitzHugh's equations (1)-(3) use `(x, y, z, c)`. Under
`v = -x`, `w = y`, `I = -z`, `tau = c t`, and `epsilon = 1/c^2`, they become

`dv/dtau = v - v^3/3 - w + I`

`dw/dtau = epsilon(v + a - bw)`.

The repository uses `a = 0.7`, `b = 0.8`, `epsilon = 0.08`, fixed `dt = 0.1`
classical RK4, and observes upward `v >= 1` crossings without reset. The
coordinate transformation follows the source equations; RK4, the parameter
profile, the observation grid, and the event convention are repository
specializations rather than claims about the paper's analogue solver.

## Runtime evidence

Python, production Rust/PyO3 and NetworkRunner, standalone safety Rust, Go,
Julia, and executable Mojo carry the same two-state RK4 recurrence. Python,
Rust, Julia, and Go are binary64-exact at the enrolled operation order. Mojo's
fused multiply-add lowering stays within `1e-9` over the measured trace while
preserving the complete event count and final-state contract.

The source-hashed packet at
`benchmarks/results/bench_fitzhugh_nagumo_simulate.json` executes Python,
production Rust, Julia, Go, and Mojo over 2,000,000 steps at current 0.5. Its
timings are non-isolated local regression context; no production-speed or
hardware measurement is claimed. Standalone safety Rust and the Go service
have separate native parity and failure-atomicity tests.

The compact receipt at
`src/sc_neurocore/neurons/reference_receipts/fitzhugh_nagumo_1961.json` replays
3,000 steps at current 0.5, records the eight exact event indices, final `(v,w)`
state, and a binary trace digest. The independent feature receipt
`fitzhugh_nagumo_driven_oscillation_doi.json` separately re-derives the RK4
feature map.

## Hardware boundary

The committed `sc_fitzhugh_nagumo.v` is exactly the current equation compiler's
signed-Q16.16 lowering. Icarus/VVP preserves all eight receipt events, the
object passes Yosys coarse synthesis, and `sc_fitzhugh_nagumo.sby` passes a
depth-4 bounded public-port reset-safety check.

This is the H2 terminal boundary for the present unit. It is not formal
real-number equation equivalence, unrestricted input parity, timing closure,
PPA, target-device validation, FPGA/board execution, physical silicon, or broad
biological validation.
