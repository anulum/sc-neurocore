<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
# SCChaoticMapNeuron

`SCChaoticMapNeuron` preserves SC-NeuroCore's original two-state,
sigmoid-gated engineering map. It is a separate project model, not an
implementation or modification of the one-state Aihara equations.

## Recurrence

Both states are updated simultaneously:

\[
x_{t+1}=\operatorname{clamp}\left(k_f x_t\sigma(x_t+\alpha)-y_t+I_t,-10,10\right),
\]

\[
y_{t+1}=\operatorname{clamp}\left(k_s y_t+\delta x_t,-10,10\right).
\]

The event output is an upward crossing of `x_threshold`. The default
parameters are `k_f=0.7`, `k_s=0.95`, `alpha=2.0`, `delta=0.05`, and
`x_threshold=0.5`.

## Provenance boundary

This recurrence is maintained as an experimental SC-NeuroCore design. It has
no claimed paper DOI and must not be cited as Aihara, Takabe, and Toyoda 1990.
The source-exact paper model is [`AiharaMapNeuron`](aihara_map.md).

## Current implementation status

The preserved recurrence has executable Python, modular Rust/PyO3, standalone
Rust safety, Julia, Go, and Mojo implementations. The five batch lanes return
both complete state traces, the edge-event trace, both final states, and the
event count. A 512-step mixed-drive parity test keeps Rust, Julia, and Go within
`1e-12` and Mojo within `1e-10` of Python, with exact events in every lane.

This establishes runtime parity without pretending the project model has a
paper-derived source oracle. Independent reference evidence, schema/RTL
co-simulation, formal work, and benchmark closure remain separate future
high-fidelity work.
