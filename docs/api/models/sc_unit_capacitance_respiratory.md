<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->

# SCUnitCapacitanceRespiratoryNeuron

`SCUnitCapacitanceRespiratoryNeuron` preserves the exact project recurrence
formerly exposed as `ButeraRespiratoryNeuron`. It is a count-neutral
compatibility identity and makes no Butera-paper attribution.

Its state is `(v, n, h_nap)`. Candidate-first RK4 at `dt=0.1 ms` advances the
historical unit-capacitance membrane equation and the two gates simultaneously.
The inactive tonic conductance keeps the former `e_syn=-10 mV` default
observable without changing the enrolled recurrence. Sampled upward crossings
of `-20 mV` emit events without resetting any state.

Python, Rust engine, Rust safety, Julia, Go, and Mojo retain the defaults and
recurrence under the SC name. Paired schemas, a frozen project-spec receipt,
measured behavior evidence, and a source-bound five-runtime benchmark establish
its independent custody. The benchmark records five events in Python, Rust,
Go, and Julia and four in Mojo over 20,000 steps at current 20; all five match
the independently frozen one-step state within `2e-12`. The explicit one-event
long-run envelope is attributed to cross-`libm` sensitivity and is not presented
as exact binary64 parity. No literature, fixed-point, RTL, synthesis, formal,
timing, PPA, device, board/HIL, or silicon claim is made.
