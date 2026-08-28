<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SC-NeuroCore — Cazelles source-faithful scalar map -->

# Cazelles map

`CazellesMapNeuron` implements the one-dimensional map introduced by Cazelles,
Courbage, and Rabinovich (2001). One `step()` call is one map iteration; no ODE
integrator or physical timestep is implied.

- Module: `sc_neurocore.neurons.models.cazelles_map`
- State: scalar `x`
- Primary source: [Cazelles, Courbage & Rabinovich (2001)](https://doi.org/10.1209/epl/i2001-00548-y)

## Published recurrence

The source defines

$$
x_{n+1}=f(x_n)+\alpha x_n^m,
$$

where $m\in\{1,2\}$ and $f$ is the four-branch map

$$
f(x)=
\begin{cases}
a_1+b_1x & x_0 < x < x_1,\\
a_2+b_2x & x_1 < x < x_2,\\
a_3+b_3x & x_2 < x < x_3,\\
a_4+b_4x & x_3 < x < x_4.
\end{cases}
$$

Figure 1 supplies the default uncoupled orbit:

| Quantity | Values |
| --- | --- |
| `alpha` | `0` |
| `x0..x4` | `0`, `0.4`, `0.6`, `0.7`, `1` |
| `a1..a4` | `0`, `1.5`, `-0.9`, `1.4` |
| `b1..b4` | `1.05`, `-1.25`, `1.5`, `-1` |

The paper writes strict inequalities, so it does not define values at exact
breakpoints. SC-NeuroCore makes that measure-zero boundary deterministic with
the right-continuous intervals `[x0,x1)`, `[x1,x2)`, `[x2,x3)`, and
`[x3,x4]`. Round-off no larger than eight binary64 ULP at a domain edge is
projected back to the exact edge; larger excursions fail atomically.

`current` is an additive maintained perturbation. `current=0` recovers the
published uncoupled recurrence. This extension is not attributed to equation
(1).

## Burst-cycle event

The source analyses burst phase through the minima that begin the slow regime.
The catalogue event therefore marks entry into that regime:

$$
s_{n+1}=\mathbf{1}[x_n\ge x_1 \land x_{n+1}<x_1].
$$

This is an explicit observation convention, not a paper-defined
action-potential threshold. It does not modify or reset `x`.

## Python use

```python
from sc_neurocore.neurons.models.cazelles_map import CazellesMapNeuron

neuron = CazellesMapNeuron()
trace, burst_entries = neuron.simulate(
    n_steps=600,
    current=0.0,
    backend="auto",
)
```

Invalid parameters, non-finite input, and candidates outside the configured
domain are rejected before state is mutated.

## Cross-runtime evidence

Python, Rust, Julia, and Go reproduce the complete binary64 reference orbit
exactly. The Figure-1 600-step orbit emits seven slow-regime-entry events, the
first at step 56. Mojo is checked per step to an eight-ULP bound; fused
multiply-add rounding is chaotically amplified over long trajectories, where
the measured 600-step count differs by one. The limitation is pinned in tests
instead of being hidden behind a long-horizon equality claim.

The independent source receipt is
`src/sc_neurocore/neurons/reference_receipts/cazelles_2001.json`. It records a
600-step trace digest derived without calling production model code. The
schema-runner reference validates the same complete feature set.

## Benchmark

The committed benchmark evidence is
`benchmarks/results/bench_cazelles_map.json`. It records the pinned
two-million-step Python, Rust, Julia, Go, and Mojo measurements together with
the parity boundary described above; it is evidence for this implementation,
not a general hardware-performance claim.

## RTL, co-simulation, and formal boundary

The hand implementation and paired TOML/JSON schemas are state- and
event-exact over the complete 600-step source orbit and visit all four
branches. Generated Q16.16 RTL is event-exact through the first 55 iterations,
including the first discontinuous-return approach, with maximum state error at
most `0.0062`.

Long fixed-point trajectories are an explicit excluded boundary: the chaotic
map amplifies quantisation, and the measured 600-step Q16.16 orbit emits two
events rather than seven. This does not weaken the bounded silicon claim; it
prevents it from being overstated.

The committed Q8.8 core:

- synthesises with Yosys and has a machine-readable cell report;
- passes the depth-4 SymbiYosys/Z3 reset-port safety job;
- carries no timing, PPA, board, device, or physical-silicon claim.

## Preserved historical recurrence

The former two-state clipped-logistic implementation was not deleted. It is
preserved with its historical trace digest unchanged under the explicit count-neutral identity
[`SCClippedLogisticBurstingMapNeuron`](sc_clipped_logistic_bursting_map.md),
without whole-model publication attribution.

## Focused reproduction

```bash
PYTHONPATH=src:bridge .venv/bin/pytest -q \
  tests/test_cazelles_map_backends.py \
  tests/test_cazelles_map_engine_binding.py \
  tests/test_cosim_cazelles_map.py \
  tests/test_reference_cazelles_map.py

cd hdl/formal/catalogue
sby -f sc_cazelles_map.sby
```
