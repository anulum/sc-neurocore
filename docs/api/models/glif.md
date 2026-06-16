# GLIFNeuron

**Module:** `sc_neurocore.neurons.models.glif`
**Rust engine:** `sc_neurocore_engine::neurons::GLIFNeuron`
**Polyglot `simulate` backends:** Rust engine (PyO3, bit-exact), Julia `GlifAccel`, Go c-shared (`accel/go/neurons/glif`), Mojo FFI (`accel/mojo/neurons/glif.mojo`); standalone Rust safety mirror `GLIFNeuron`
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
| Python | Authoritative reference class with candidate-first RK4 and explicit runtime validation; also the `simulate` reference loop. |
| Rust engine | PyO3-backed class uses the same RK4 state transition and additive spike reset, and exposes `py_glif_simulate` for the accelerated N-step path (bit-exact). |
| Julia | `GlifAccel.simulate_trace` mirrors the N-step RK4 recurrence (bit-exact, linear arithmetic). |
| Go | `accel/go/neurons/glif` builds a C-ABI shared library (`glif_simulate_c`) loaded via ctypes (bit-exact). |
| Mojo | `accel/mojo/neurons/glif.mojo` builds an FFI kernel (`glif_simulate_c`); validated non-amplifying within a ULP band (FMA fusion). |
| Rust safety | Standalone safety mirror validates the same single-`step` state transition and reset semantics. |

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

## Polyglot acceleration

A single `step` is trivial, but an N-step run is a sequential RK4 recurrence
with a discontinuous spike reset that does not vectorise, so a compiled inner
loop genuinely beats Python. `simulate(n_steps, current, backend="auto")`
dispatches across the polyglot chain and returns `(trace, spikes)`:

```python
from sc_neurocore.neurons.models.glif import GLIFNeuron

neuron = GLIFNeuron()
trace, spikes = neuron.simulate(2_000_000, current=30.0)   # auto → Rust
```

The Allen GLIF5 right-hand side is purely linear — additions, multiplications
and divisions, no transcendental functions — so every RK4 stage is exact
arithmetic. The Rust engine, Julia and Go backends therefore reproduce the
NumPy reference **bit-for-bit**: trace, spike count and the final
`(v, theta, i_asc1, i_asc2)` state all match exactly. Mojo fuses multiply-add;
on this platform it also matches bit-for-bit, but because FMA fusion is
compiler/version dependent it is validated as non-amplifying within a tight ULP
band (with identical spike counts) rather than asserted strictly exact. `auto`
selects Rust (the bit-exact backend shipped in the wheel).

### Measured throughput

2,000,000 RK4 steps, default tonic regime (`current=30.0`), median of 5
repeats. Non-isolated loaded workstation per
`BROADCAST_2026-06-04_benchmark_core_isolation` — functional/regression
evidence, not an isolated-core figure. Reproduce with
`python benchmarks/bench_glif_simulate.py`.

| Backend | Median (ms) | Speed-up vs Python | Whole-trace parity |
|---------|------------:|-------------------:|--------------------|
| python  | 2316.01 | 1.0× | reference |
| go      | 81.67 | 28.4× | bit-exact (0) |
| mojo    | 88.34 | 26.2× | 0 measured (FMA, ULP-validated) |
| rust (`auto`) | 89.08 | 26.0× | bit-exact (0) |
| julia   | 92.17 | 25.1× | bit-exact (0) |

Artefact: `benchmarks/results/bench_glif_simulate.json`.

## Verification commands

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests/test_model_glif.py -q
PYTHONPATH=src .venv/bin/python -m pytest tests/test_glif_backends.py -q
cargo test --manifest-path engine/Cargo.toml glif_ --release
(cd src/sc_neurocore/accel/go/neurons/glif && go build -buildmode=c-shared -o libglif.so glif.go)
(cd src/sc_neurocore/accel/mojo/neurons && mojo build --emit shared-lib -o libglif.so glif.mojo)
PYTHONPATH=src .venv/bin/python benchmarks/bench_glif_simulate.py --json benchmarks/results/bench_glif_simulate.json
```
