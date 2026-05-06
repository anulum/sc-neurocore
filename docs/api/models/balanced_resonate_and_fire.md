<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996-2026 Miroslav Sotek. All rights reserved. -->
<!-- © Code 2020-2026 Miroslav Sotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->

# BalancedResonateAndFireNeuron

**Python:** `sc_neurocore.neurons.models.balanced_resonate_and_fire.BalancedResonateAndFireNeuron`

**Reference:** Higuchi, S., Kairat, S., Bohte, S. and Otte, S. (2024).
*Balanced Resonate-and-Fire Neurons*. Proceedings of ICML 2024, PMLR
235:18305-18323.

## Model

This is the Balanced Resonate-and-Fire (BRF) neuron from Algorithm 1 of
Higuchi et al. 2024, not the older Izhikevich 2001 RF neuron.

State:

```text
u = x + i y
q = refractory state
```

One timestep:

```text
b_t = p(omega) - b_offset - q_{t-1}
u_t = u_{t-1} + dt * ((b_t + i omega) * u_{t-1} + I_t)
theta_t = theta_c + q_{t-1}
z_t = Theta(Re(u_t) - theta_t)
q_t = gamma q_{t-1} + z_t
```

Divergence boundary:

```text
p(omega) = (-1 + sqrt(1 - (dt omega)^2)) / dt
```

The implementation rejects `dt * omega > 1` because the boundary is no longer
real. This follows the paper's discrete-time convergence condition.

## Design Boundary

Implemented:

- refractory period in the threshold;
- smooth reset through the damping term;
- divergence-bound damping;
- scalar deterministic neuron stepping;
- public import through `sc_neurocore.neurons` and `sc_neurocore.neurons.models`;
- `Population` construction using the BRF class;
- Python benchmark artefact in `benchmarks/results/`;
- Rust engine, PyO3, NetworkRunner, and safety mirror for the same scalar equations;
- Go, Julia, and Mojo research mirrors for the scalar update equation.

Not claimed:

- the full BRF-RSNN training recipe from the paper;
- benchmark parity with the ICML experiments;
- BHRF, the balanced harmonic RF variant.
- Universal DSL / schema execution for BRF.

Those require separate network-training integration and benchmark artefacts.
The generic equation DSL is intentionally not used for this model because its
threshold/reset contract evaluates the threshold after the state update, while
Algorithm 1 uses `q_{t-1}` in both `b_t` and `theta_t` and only then updates
`q_t = gamma q_{t-1} + z_t`. Encoding BRF as an ordinary ODE schema would
change that event ordering.

## Usage

```python
from sc_neurocore.neurons.models import BalancedResonateAndFireNeuron

neuron = BalancedResonateAndFireNeuron(omega=10.0, b_offset=1.0, dt=0.01)
spike = neuron.step(current=2.0)
snapshot = neuron.state()
```

## Verification

Focused tests cover:

- exact Algorithm 1 one-step update;
- divergence-bound formula;
- fail-fast invalid parameter handling;
- refractory threshold and smooth reset behaviour;
- deterministic traces;
- public import and `Population` wiring.

## Benchmark

Local benchmark command:

```bash
python benchmarks/bench_balanced_resonate_and_fire.py
```

Measured on Linux 6.17 / Python 3.12 / NumPy 2.2.6. The scalar rows use the
same 200,000-step `I=2`, `omega=10`, `b_offset=1` workload unless noted.

| Backend | Status | Time | Step time | Final `(x, y, q)` |
|---|---|---:|---:|---|
| Python | executed | 0.312 s | 1,559 ns | `(0.029363343430484898, 0.1955918095998993, 0.0)` |
| Rust PyO3 | executed | 0.0176 s | 87.9 ns | `(0.029363343430484898, 0.1955918095998993, 0.0)` |
| Go | executed | benchmark harness | 33.7 ns | harness reports `ns/op` |
| Julia | executed | 0.00410 s | 20.5 ns | `(0.029363343430484898, 0.1955918095998993, 0.0)` |
| Mojo | executed | 0.00116 s | 5.82 ns | `(0.029363343430484894, 0.1955918095998993, 0.0)` |

Additional Python workload rows:

| Workload | Time | Throughput |
|---|---:|---:|
| scalar Python BRF, 200,000 steps, `I=20`, `omega=20` | 0.318 s | 1.59 us/step |
| Python 256-neuron population, 2,000 steps | 0.798 s | 641,712 updates/s |

The benchmark JSON stores these as side-by-side comparison rows with backend
status, absolute timing, final states where the backend harness reports them,
and derived speedup values.

Benchmark JSON is written to
`benchmarks/results/bench_balanced_resonate_and_fire.json`.
