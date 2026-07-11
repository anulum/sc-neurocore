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

## Maintained defaults and source boundary

| Name | Default | Meaning |
| --- | ---: | --- |
| `x` | `0.0` | initial fast coordinate |
| `y` | `0.0` | initial recovery coordinate |
| `m0` | `0.0864` | magnitude of the outer-branch slope |
| `m1` | `0.65` | middle-branch slope |
| `a` | `0.2` | middle-branch offset and breakpoint parameter |
| `d` | `0.235` | Heaviside discontinuity |
| `j` | `0.2` | constant external-stimulus parameter `J` |
| `beta` | `0.085` | discontinuity jump magnitude |
| `eps` | `0.02` | recovery-coordinate scale separation |
| `x_threshold` | `0.235` | software event threshold |

The paper's Figure 1 uses `m0=0.0864`, `m1=0.65`, and `a=0.2` to illustrate
the admissible `B⁺` region. It does not publish the repository's complete
default tuple as one experimental operating point. The maintained
`d/J/β/ε` values are repository choices within the paper's analysed region:

$$
0<J<d, \qquad J_{\min}<d<J_{\max}, \qquad m_0<1,
$$

with `J_min=0.1765344921`, `J_max=0.2938620315`, and
`β₀=0.0762629006 < β=0.085 < β₁=0.0964680880` from equations 9 and 12.
This statement does not identify the defaults with the paper's Figure 4
bursting example, which uses a different parameter tuple.

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

The acceleration surface was completed in commit `63826b513` and is unchanged
by the schema-to-RTL enrolment. Rust, Julia, and Go reproduce the Python golden
trace bit-for-bit at the maintained test points. Mojo is checked per step
because fused multiply-add rounding is amplified by this sensitive map.

The committed benchmark artefact records 2,000,000 autonomous iterations,
five repeats, and a non-isolated loaded workstation. Python, Rust, Julia, and
Go each record 371,008 events; Mojo records 371,063 and is therefore not
presented as event-exact on that long trajectory. Reproduce the artefact with:

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
| `-0.3` | `30` | `(22, 1, 7)` | `1` | `(0.001969, 0.000163)` |
| `0.0` | `20` | `(13, 7, 0)` | `2` | `(0.013687, 0.000304)` |
| `0.3` | `30` | `(7, 1, 22)` | `1` | `(0.000491, 0.000051)` |

Q32.32 RTL is event-exact at `I=-0.3/0/0.3` over 30 iterations. Across that
set, the largest fast-coordinate error is `2.604e-5` and the largest recovery
error is `8.379e-7`.

### Declared Q16.16 boundary

The 30-iteration autonomous Q16.16 trajectory is deliberately excluded from
the parity band. Float64 emits four events, Q16.16 emits six, and six event
positions differ. Q32.32 resolves that same window at four events on both
paths. The regression pins this boundary so the bounded evidence cannot be
read as long-window fixed-point identity.

The class-specific evidence lives in
`tests/test_cosim_courage_nekorkin_map.py`.

## Independent reference trace

`courage_nekorkin_map_autonomous_doi.json` records a 30-iteration autonomous
trace. Its model-scoped test independently reimplements equations 3–5 rather
than calling the hand class or schema expressions. The committed features cover
event count, first event, and final/minimum/maximum/mean values for both state
coordinates. The production schema runner then validates the same artefact.

## Silicon and formal scope

The descriptor is science tier S5 and silicon tier H1:

- the paired schemas match the maintained hand implementation;
- generated Q16.16 and Q32.32 RTL satisfy the bounded contracts above;
- generated Q8.8 RTL is enrolled in the catalogue formal inventory;
- a port-only depth-4 SymbiYosys/Z3 job proves reset-spike safety.

The formal job is structural safety evidence. It does not replace the
behavioural trajectories and does not claim FPGA synthesis, timing closure, or
hardware deployment.

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
