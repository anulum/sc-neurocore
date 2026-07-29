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

## Fidelity evidence

The preserved recurrence has executable Python, modular Rust/PyO3, standalone
Rust safety, Julia, Go, and Mojo implementations. The five batch lanes return
both complete state traces, the edge-event trace, both final states, and the
event count. A 512-step mixed-drive parity test keeps Rust, Julia, and Go within
`1e-12` and Mojo within `1e-10` of Python, with exact events in every lane.

The independent `sc-neurocore.sc-chaotic-map-project-spec.v1` receipt derives
the recurrence directly without importing the production model. Both paired
TOML/JSON schemas reproduce the hand states and events exactly.

Generated Q8.24 RTL is bit-exact to its quantized oracle across a 32-step
alternating-drive trace and preserves all 16 upward crossings. Against the
binary64 project recurrence, maximum observed errors stay below `0.009` for
`x` and `0.0016` for `y`. Yosys synthesis and depth-12 Z3 induction establish
the enrolled bounded-state and crossing-safety properties; they are not claims
of formal real-number equivalence, timing closure, device validation, or PPA.

The source- and binary-bound 200,000-step benchmark records 100,000 exact
events in all five maintained lanes. Its timings are local diagnostic evidence,
not a portable backend ranking.

## Native API documentation

- Python documents identity, simultaneous updates, event semantics, reset, and
  atomic batch behavior in the model and dispatcher docstrings.
- The production engine and standalone safety crate expose Rustdoc for public
  states, parameters, checked stepping, reset, and complete batch receipts.
- GoDoc, Julia docstrings, and Mojo module/ABI comments state the same project
  contract and failure boundary.
- The RTL header documents the signed Q8.24 interface, sigmoid LUT, simultaneous
  state contract, event output, and absence of publication attribution.
