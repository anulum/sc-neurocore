# GLIFNeuron

**Module:** `sc_neurocore.neurons.models.glif`
**Reference:** Teeter, C. et al., Nat. Commun. 9:709, 2018
**Family:** Allen Institute generalised leaky integrate-and-fire level 5
**State variables:** `v`, `theta`, `i_asc1`, `i_asc2`

## Mathematical contract

`GLIFNeuron` advances the continuous four-state GLIF5 flow with a candidate-first fourth-order Runge-Kutta step. The spike reset is applied only after the candidate state is finite and the candidate membrane voltage crosses the adaptive threshold.

Membrane flow:

$$
\frac{dV}{dt} = \frac{-(V - V_{rest}) + R I_{ext} + I_{asc1} + I_{asc2}}{\tau_m}
$$

Threshold flow:

$$
\frac{d\theta}{dt} = \frac{\theta_\infty - \theta + a_\theta(V - V_{rest})}{\tau_\theta}
$$

After-spike current flow:

$$
\frac{dI_{asc1}}{dt} = -\frac{I_{asc1}}{\tau_{asc1}},\qquad
\frac{dI_{asc2}}{dt} = -\frac{I_{asc2}}{\tau_{asc2}}
$$

Candidate-first RK4 update:

$$
y_{n+1}^{candidate} = y_n + \frac{\Delta t}{6}(k_1 + 2k_2 + 2k_3 + k_4)
$$

where `y = (v, theta, i_asc1, i_asc2)` and each `k` evaluates the same GLIF vector field at the RK4 stage state.

Spike reset:

$$
V \leftarrow V_{reset}
$$

$$
\theta \leftarrow \theta_{candidate} + \Delta_\theta,
\qquad I_{asc1} \leftarrow I_{asc1,candidate} + r_{asc1},
\qquad I_{asc2} \leftarrow I_{asc2,candidate} + r_{asc2}
$$

This preserves the Allen GLIF additive threshold jump while avoiding direct Euler drift in the membrane, threshold, and after-spike current states.

## Validation boundaries

The Python reference raises before mutation when:

- `current` is non-finite.
- Any state or scalar parameter is non-finite at runtime.
- `tau_m`, `tau_theta`, `tau_asc1`, `tau_asc2`, or `dt` is non-positive.
- `delta_theta` or `resistance` is negative.
- The RK4 candidate contains a non-finite value.

The non-throwing Go, Julia, Rust engine, and Rust safety mirrors preserve state and return no spike for the same invalid runtime boundaries.

## Cross-language surfaces

| Surface | Contract |
| --- | --- |
| Python | Authoritative reference class with candidate-first RK4 and explicit runtime validation. |
| Rust engine | PyO3-backed class uses the same RK4 state transition and additive spike reset. |
| Go service | Service mirror uses the same scalar equations and reference-point tests. |
| Julia mirror | Scientific mirror uses the same scalar equations and state-preserving invalid-input behaviour. |
| Rust safety | Standalone safety mirror validates the same state transition and reset semantics. |
| Mojo | Scalar kernel contract records the GLIF RK4 vector-field semantics for later kernel work. |

## Behavioural invariants

- A neuron starting from `v = -68.0`, `theta = -45.0`, `i_asc1 = 0.4`, `i_asc2 = -0.2`, with `current = 4.0`, advances to `v = -67.7924658728125` and `theta = -45.049541282631253` after one RK4 step.
- A neuron starting from `v = -51.0`, `theta = -50.5`, `delta_theta = 2.5`, `r_asc1 = 1.25`, `r_asc2 = -0.25`, with `current = 40.0`, spikes, resets voltage to `v_reset`, and updates threshold to `-47.9930331381625`.
- Fast after-spike current decays faster than slow after-spike current when `tau_asc1 < tau_asc2`.
- Positive voltage-threshold coupling raises the adaptive threshold relative to the uncoupled case under the same starting state.
- Invalid runtime inputs do not mutate state on non-throwing mirrors.

## Usage

```python
from sc_neurocore.neurons.models.glif import GLIFNeuron

neuron = GLIFNeuron(delta_theta=2.5, r_asc1=1.25, r_asc2=-0.25)
spikes = sum(neuron.step(40.0) for _ in range(1000))
print(spikes, neuron.v, neuron.theta, neuron.i_asc1, neuron.i_asc2)
```

```rust
use sc_neurocore_engine::neurons::GLIFNeuron;

let mut neuron = GLIFNeuron::new();
neuron.delta_theta = 2.5;
neuron.r_asc1 = 1.25;
neuron.r_asc2 = -0.25;
let spikes: i32 = (0..1000).map(|_| neuron.step(40.0)).sum();
println!("{spikes} {} {} {} {}", neuron.v, neuron.theta, neuron.i_asc1, neuron.i_asc2);
```

## Benchmark evidence

Generated locally on 2026-06-01 with:

```bash
PYTHONPATH=src .venv/bin/python benchmarks/bench_glif.py
```

Evidence file: `benchmarks/results/bench_glif.json`

| Backend | Steps/s | Wall ms for 100000 steps | Relative speed |
| --- | ---: | ---: | ---: |
| Python | 185536 | 538.98 | 1.00x |
| Rust PyO3 | 5149511 | 19.42 | 27.75x |

Parity evidence over 10000 steps: `max_abs_delta = 0.0`, `spikes_delta = 0`.

## Verification commands

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests/test_model_glif.py -q
(cd src/sc_neurocore/accel/go && go test ./services -run 'GLIF')
rustc --test src/sc_neurocore/accel/rust/safety/glif.rs -o /tmp/sc_neurocore_glif_safety && /tmp/sc_neurocore_glif_safety
cargo test --manifest-path engine/Cargo.toml glif_ --release
.venv/bin/python -m maturin develop --manifest-path engine/Cargo.toml --release
PYTHONPATH=src .venv/bin/python benchmarks/bench_glif.py
```
