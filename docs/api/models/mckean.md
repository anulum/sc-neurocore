<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
# McKeanNeuron

`McKeanNeuron` is the catalogue-counted, source-bound space-clamped system from
Tonnelier's equations (1.3)-(1.6), following McKean's Nagumo caricature.

## Source equations

```text
dv/dt = -lambda*v + mu*H(v-a) - w + I
dw/dt = b*v
```

The numerical specialization declares the right-continuous convention
`H(0)=1`. Defaults are `v=w=0`, `a=0.25`, `lambda=mu=1`, `b=0.01`, and
`dt=0.1`; constraints include `mu > lambda*a`. The coupled state advances with
simultaneous RK4. An event is sampled only when `v` crosses `a` upward, and the
ODE has no spike reset.

Primary references:

- H. P. McKean (1970), *Nagumo's equation*, DOI `10.1016/0001-8708(70)90023-X`.
- A. Tonnelier (2003), *The McKean's Caricature of the FitzHugh--Nagumo Model I.
  The Space-Clamped System*, DOI `10.1137/S0036139901393500`. A 2002 preprint is
  preserved separately as HAL `hal-00393725`.

## Runtime and evidence contract

Python, Rust, Julia, Go, and Mojo execute complete arbitrary-current traces
through `sc_neurocore.accel.mckean`. Every runtime rejects non-finite input and
preserves state when a candidate leaves the enrolled safety envelope. A
deterministic independent-oracle receipt anchors trajectory parity; focused
tests cover the source equations, event semantics, schema parity, NetworkRunner,
native backends, signed-Q32.32 RTL co-simulation, Yosys coarse synthesis, and
bounded reset safety.

`SCTriangularMcKeanNeuron` separately preserves the prior three-branch project
recurrence and is count-neutral. Spatial diffusion, traveling waves, and network
statistics remain outside this scalar neuron unit.
