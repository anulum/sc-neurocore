<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
# SCTriangularMcKeanNeuron

`SCTriangularMcKeanNeuron` preserves the former SC-NeuroCore recurrence as an
explicit, count-neutral project modification. It is not the catalogue-counted
McKean source identity and carries no McKean-paper attribution.

## Recurrence

```text
dv/dt = f(v) - w + I
dw/dt = epsilon * (v - gamma*w)

f(v) = -v       when v < a/2
       v-a      when a/2 <= v < (1+a)/2
       1-v      otherwise
```

The coupled state uses candidate-first RK4. An event is sampled only when `v`
crosses `v_peak` upward; there is no implicit reset. Defaults are `v=w=0`,
`a=0.25`, `epsilon=0.01`, `gamma=0.5`, `v_peak=0.8`, and `dt=0.1`.

## Runtime and evidence contract

Python, Rust, Julia, Go, and Mojo execute the same recurrence through
`sc_neurocore.accel.sc_triangular_mckean`. Runtime input is a finite
one-dimensional current trace. Candidate failure preserves the previous state.
The descriptor, reference receipt, focused tests, benchmark result, Rustdoc,
Julia docstrings, GoDoc, Mojo comments, RTL co-simulation, and bounded formal
harness are all enrolled under the SC identity. The complete 3,000-step
`<float64 v, float64 w, uint8 event>` trace is locked by SHA-256
`993226d5bf608aaf83f14e1e82a6b9df8278ccbc3326089551fd7bf2f19a8fca`;
the signed-Q32.32 RTL also passes Yosys coarse synthesis, establishing the
declared H2 terminal tier without a timing, PPA, device, or universal-equivalence
claim.

This scalar unit does not include a spatial medium, network topology, or
population statistics. Those are separate model surfaces.
