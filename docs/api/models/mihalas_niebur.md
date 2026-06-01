# MihalasNieburNeuron

**Module:** `sc_neurocore.neurons.models.mihalas_niebur`
**Reference:** Mihalas, S. & Niebur, E., Neural Comput. 21(3):704-718, 2009
**Family:** Generalised integrate-and-fire
**State variables:** `v`, `theta`, `i1`, `i2`

## Mathematical contract

The implementation advances the continuous four-state Mihalas-Niebur flow with a candidate-first fourth-order Runge-Kutta step. The spike reset is applied only after the candidate state is finite and the candidate membrane voltage crosses the adaptive threshold.

Membrane flow:

$$
\frac{dV}{dt} = \frac{-(V - V_{rest}) + I_1 + I_2 + I_{ext}}{\tau_v}
$$

Threshold flow:

$$
\frac{d\theta}{dt} = \frac{\theta_\infty - \theta + a(V - V_{rest})}{\tau_\theta}
$$

After-spike current flow:

$$
\frac{dI_1}{dt} = -\frac{I_1}{\tau_1},\qquad
\frac{dI_2}{dt} = -\frac{I_2}{\tau_2}
$$

Candidate-first RK4 update:

$$
y_{n+1}^{candidate} = y_n + \frac{\Delta t}{6}(k_1 + 2k_2 + 2k_3 + k_4)
$$

where `y = (v, theta, i1, i2)` and each `k` evaluates the same continuous vector field at the RK4 stage state.

Spike reset:

$$
V \leftarrow V_{reset} + b(V_{candidate} - V_{rest})
$$

$$
\theta \leftarrow \max(\theta_{candidate}, \theta_{reset}),\qquad
I_1 \leftarrow I_{1,candidate} + r_1,\qquad
I_2 \leftarrow I_{2,candidate} + r_2
$$

The `b` parameter preserves a controlled fraction of the candidate voltage excursion after a spike and is required for burst-like reset semantics.

## Fail-closed boundaries

`step(current)` returns `0` and preserves state when:

- `current` is non-finite.
- Any state or parameter is non-finite before the step.
- `tau_v`, `tau_theta`, `tau_1`, `tau_2`, or `dt` is non-positive.
- The RK4 candidate contains a non-finite value.

These boundaries are intentionally consistent across the Python reference, Rust engine, Go service, Julia mirror, and Rust safety surface.

## Cross-language surfaces

| Surface | Contract |
| --- | --- |
| Python | Authoritative reference class with candidate-first RK4 and fail-closed state preservation. |
| Rust engine | PyO3-backed engine class uses the same RK4 state transition and spike reset. |
| Go service | Service mirror uses the same scalar equations and reference-point tests. |
| Julia mirror | Scientific mirror uses the same scalar equations and fail-closed checks. |
| Rust safety | Standalone safety mirror validates the same state transition and reset semantics. |
| Mojo | Scalar kernel contract records the RK4 vector-field semantics for downstream kernel work. |

## Behavioural invariants

- A subthreshold default neuron with `current = 0.5` advances to `v = 0.04758125`, `theta = 1.0`, `i1 = 0.0`, and `i2 = 0.0` after one RK4 step.
- A neuron with `v = 0.99`, `b = 0.5`, `r1 = 1.25`, `r2 = -0.25`, and `current = 2.0` spikes and resets to `v = 0.5430570625`.
- Positive voltage-threshold coupling (`a > 0`) raises the adaptive threshold relative to the uncoupled case under the same starting state.
- Fast after-spike current decays faster than slow after-spike current when `tau_1 < tau_2`.
- Invalid inputs and invalid runtime parameters do not mutate state.

## Usage

```python
from sc_neurocore.neurons.models.mihalas_niebur import MihalasNieburNeuron

neuron = MihalasNieburNeuron(b=0.5, r1=1.25, r2=-0.25)
spikes = sum(neuron.step(2.0) for _ in range(1000))
print(spikes, neuron.v, neuron.theta, neuron.i1, neuron.i2)
```

```rust
use sc_neurocore_engine::neurons::MihalasNieburNeuron;

let mut neuron = MihalasNieburNeuron::new();
neuron.b = 0.5;
neuron.r1 = 1.25;
neuron.r2 = -0.25;
let spikes: i32 = (0..1000).map(|_| neuron.step(2.0)).sum();
println!("{spikes} {} {} {} {}", neuron.v, neuron.theta, neuron.i1, neuron.i2);
```

## Benchmark evidence

Generated locally on 2026-06-01 with:

```bash
PYTHONPATH=src .venv/bin/python benchmarks/bench_mihalas_niebur.py
```

Evidence file: `benchmarks/results/bench_mihalas_niebur.json`

| Backend | Steps/s | Wall ms for 100000 steps | Relative speed |
| --- | ---: | ---: | ---: |
| Python | 148913 | 671.53 | 1.00x |
| Rust PyO3 | 4470483 | 22.37 | 30.02x |

Parity evidence over 10000 steps: `max_abs_delta = 0.0`, `spikes_delta = 0`.

## Verification commands

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests/test_model_mihalas_niebur.py -q
(cd src/sc_neurocore/accel/go && go test ./services -run 'MihalasNiebur')
rustc --test src/sc_neurocore/accel/rust/safety/mihalas_niebur.rs -o /tmp/sc_neurocore_mihalas_niebur_safety && /tmp/sc_neurocore_mihalas_niebur_safety
cargo test --manifest-path engine/Cargo.toml mn_ --release
.venv/bin/python -m maturin develop --manifest-path engine/Cargo.toml --release
PYTHONPATH=src .venv/bin/python benchmarks/bench_mihalas_niebur.py
```
