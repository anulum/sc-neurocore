<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SC-NeuroCore — Courbage-Nekorkin-Vdovin map model -->

# Courbage–Nekorkin–Vdovin map

`CourageNekorkinMapNeuron` implements the discontinuous two-state map proposed
by Courbage, Nekorkin, and Vdovin (2007). One `step()` call advances one map
iteration; there is no ODE integrator or physical timestep.

- Module: `sc_neurocore.neurons.models.courage_nekorkin_map`
- Family: discontinuous map-based neuron
- State: fast membrane-like coordinate `x`, slow recovery coordinate `y`
- Primary source: [Courbage, Nekorkin & Vdovin (2007)](https://doi.org/10.1063/1.2795435), equations 3–6

The public class and schema identifiers retain the historical `Courage...`
spelling for compatibility. The publication and prose use the authors' surname
`Courbage`.

## Recurrence

The published autonomous map is

$$
\begin{aligned}
x_{n+1} &= x_n + F(x_n) - y_n - \beta H(x_n-d), \\
y_{n+1} &= y_n + \varepsilon(x_n-J),
\end{aligned}
$$

where

$$
F(x)=
\begin{cases}
-m_0x, & x\leq J_{\min},\\
m_1(x-a), & J_{\min}<x<J_{\max},\\
-m_0(x-1), & x\geq J_{\max},
\end{cases}
$$

$$
J_{\min}=\frac{am_1}{m_0+m_1}, \qquad
J_{\max}=\frac{m_0+am_1}{m_0+m_1}, \qquad
H(z)=\begin{cases}1,&z\geq0,\\0,&z<0.\end{cases}
$$

SC‑NeuroCore adds the API input `I_n` to the fast recurrence:

$$
x_{n+1}=x_n+F(x_n)-y_n-\beta H(x_n-d)+I_n.
$$

`I_n=0` therefore reproduces the published autonomous map. Both candidate
coordinates are computed from `(x_n, y_n)` and committed simultaneously.

## Figure-4 defaults and source boundary

| Name | Default | Meaning |
| --- | ---: | --- |
| `x` | `0.0` | initial fast coordinate |
| `y` | `0.0` | initial recovery coordinate |
| `m0` | `0.4` | magnitude of the outer-branch slope |
| `m1` | `0.65` | middle-branch slope |
| `a` | `0.2` | middle-branch offset and breakpoint parameter |
| `d` | `0.3` | Heaviside discontinuity |
| `j` | `0.13` | constant external-stimulus parameter `J` |
| `beta` | `0.25` | discontinuity jump magnitude |
| `eps` | `0.002` | recovery-coordinate scale separation |
| `x_threshold` | `0.3` | software event threshold |

The complete dynamical tuple is the paper's Figure 4 chaotic-attractor and
relaxation spike-bursting profile. The initial state `x=y=0` is the maintained
reproducibility protocol because the caption does not prescribe an initial
condition. The profile satisfies the paper's analysed-region conditions:

$$
0<J<d, \qquad J_{\min}<d<J_{\max}, \qquad m_0<1,
$$

with `J_min=0.1238095238` and `J_max=0.5047619048`.

## Event observable

The paper defines the map dynamics but not SC‑NeuroCore's binary event API. The
maintained observable emits one event on an upward crossing of
`x_threshold`:

$$
s_{n+1}=\mathbf{1}\!\left[
x_n<x_{\mathrm{threshold}}\ \land
x_{n+1}\geq x_{\mathrm{threshold}}
\right].
$$

The default event threshold equals `d`. The event does not reset either state.

## Python use

```python
from sc_neurocore.neurons.models.courage_nekorkin_map import (
    CourageNekorkinMapNeuron,
)

neuron = CourageNekorkinMapNeuron()
event = neuron.step(current=0.0)

trace, events = neuron.simulate(
    n_steps=2_000,
    current=0.0,
    backend="auto",
)
```

`simulate()` accepts `python`, `rust`, `julia`, `go`, `mojo`, or `auto`.
`auto` selects the compiled Rust path when available and otherwise uses the
Python implementation.

## Polyglot acceleration

Rust, Julia, Go, and Mojo reproduce the Python golden trace bit-for-bit. The
Mojo kernel preserves each binary64 product rounding with a non-inlined product
boundary, preventing FMA contraction from changing the chaotic orbit.

The committed benchmark artefact records 2,000,000 autonomous iterations,
five repeats, and a non-isolated loaded workstation. Every runtime records
88,435 events and zero full-trace state error. Median times in the recorded
run were 874.69 ms (Python), 15.94 ms (Rust), 15.67 ms (Julia), 11.87 ms (Go),
and 15.24 ms (Mojo). Reproduce the artefact with:

```bash
PYTHONPATH=src .venv/bin/python \
  benchmarks/bench_courage_nekorkin_map_simulate.py \
  --json benchmarks/results/bench_courage_nekorkin_map_simulate.json
```

The JSON result and `courage-nekorkin-map-five-backend-local-regression` gate
carry the measured timings, parity fields, workload, environment, and source
hashes. The numbers are regression evidence for that recorded environment,
not isolated-core release claims.

## Python-to-Verilog validation

The paired TOML and JSON schemas use `method="map"`, `dt=1.0`, and the exact
published branch ordering. Their `dt` value is the schema's iteration unit, not
physical time. The hand model and both schema runners agree exactly on every
state and event at the enrolled points.

Q16.16 RTL is event-exact on three bounded operating windows:

| Input | Iterations | Branch counts `(low, middle, high)` | Events | Maximum state error `(x, y)` |
| ---: | ---: | ---: | ---: | ---: |
| `-0.3` | `128` | `(128, 0, 0)` | `0` | `(<0.0016, <0.00066)` |
| `0.0` | `512` | `(512, 0, 0)` | `0` | `(<0.0038, <0.0016)` |
| `0.3` | `128` | `(1, 2, 125)` | `1` | `(<0.0020, <0.0008)` |

Q32.32 RTL is event-exact at `I=-0.3` over 128 iterations, `I=0` over 620,
and `I=0.3` over 128. The respective `(x,y)` error bounds are
`(<5e-8,<2e-8)`, `(<4.4e-4,<1.4e-6)`, and `(<1e-8,<5e-9)`.

### Declared Q16.16 boundary

The 620-iteration autonomous Q16.16 trajectory is deliberately excluded from
the parity band. Float64 emits one event, Q16.16 emits 25, and 26 event
positions differ. Q32.32 resolves the complete 620-step event vector. The
regression pins this boundary so bounded Q16.16 evidence cannot be read as
long-window fixed-point identity.

The class-specific evidence lives in
`tests/test_cosim_courage_nekorkin_map.py`.

## Independent reference trace

`courage_nekorkin_map_autonomous_doi.json` records a 2,000-iteration autonomous
Figure-4 trace. Its model-scoped test independently reimplements equations 3–5 rather
than calling the hand class or schema expressions. The committed features cover
event count, first event, and final/minimum/maximum/mean values for both state
coordinates. The production schema runner then validates the same artefact.

## Silicon and formal scope

The descriptor is science tier S5 and silicon tier H2:

- the paired schemas match the maintained hand implementation;
- generated Q16.16 and Q32.32 RTL satisfy the bounded contracts above;
- generated Q32.32 RTL is enrolled in the catalogue formal inventory and
  synthesises in Yosys;
- a port-only depth-4 SymbiYosys/Z3 job proves reset-spike safety.

The formal job is structural safety evidence. Coarse synthesis is not timing
closure, PPA evidence, board deployment, or physical silicon.

## Reproducing the evidence

```bash
PYTHONPATH=src:. .venv/bin/python -m pytest -q \
  tests/test_courage_nekorkin_map_backends.py \
  tests/test_cosim_courage_nekorkin_map.py \
  tests/test_reference_courage_nekorkin_map.py

PYTHONPATH=src:. .venv/bin/python tools/readiness_evidence_index.py --check

cd hdl/formal/catalogue
sby -f sc_courbage_nekorkin_map.sby
```

See the [co-simulation guide](../../guides/cosimulation_guide.md),
[reference-trace contract](../../validation/reference_traces.md), and
[model fidelity status](../model_fidelity_status.md) for the catalogue-wide
context.
