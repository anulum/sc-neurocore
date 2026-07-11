# MihalasNieburNeuron

**Module:** `sc_neurocore.neurons.models.mihalas_niebur`
**Rust engine:** `sc_neurocore_engine::neurons::MihalasNieburNeuron`
**Polyglot `simulate` backends:** Rust engine (PyO3, bit-exact), Julia `MihalasNieburAccel`, Go c-shared (`accel/go/neurons/mihalas_niebur`), Mojo FFI (`accel/mojo/neurons/mihalas_niebur.mojo`); standalone Rust safety mirror `MihalasNieburNeuron`
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
| Python | Authoritative reference class with candidate-first RK4 and fail-closed state preservation; also the `simulate` reference loop. |
| Rust engine | PyO3-backed engine class uses the same RK4 state transition and spike reset, and exposes `py_mihalas_niebur_simulate` for the accelerated N-step path (bit-exact). |
| Julia | `MihalasNieburAccel.simulate_trace` mirrors the N-step RK4 recurrence (bit-exact, linear arithmetic). |
| Go | `accel/go/neurons/mihalas_niebur` builds a C-ABI shared library (`mihalas_niebur_simulate_c`) loaded via ctypes (bit-exact). |
| Mojo | `accel/mojo/neurons/mihalas_niebur.mojo` builds an FFI kernel (`mihalas_niebur_simulate_c`); validated non-amplifying within a ULP band (FMA fusion). |
| Rust safety | Standalone safety mirror validates the same single-`step` state transition and reset semantics. |

## Behavioural invariants

- A subthreshold default neuron with `current = 0.5` advances to `v = 0.04758125`, `theta = 1.0`, `i1 = 0.0`, and `i2 = 0.0` after one RK4 step.
- A neuron with `v = 0.99`, `b = 0.5`, `r1 = 1.25`, `r2 = -0.25`, and `current = 2.0` spikes and resets to `v = 0.5430570625`.
- Positive voltage-threshold coupling (`a > 0`) raises the adaptive threshold relative to the uncoupled case under the same starting state.
- Fast after-spike current decays faster than slow after-spike current when `tau_1 < tau_2`.
- Invalid inputs and invalid runtime parameters do not mutate state.

## Hardware co-simulation

The paired TOML and JSON schemas reproduce the hand model's event decision and all four
post-step states exactly over a 1,600-step varied-current sequence containing 168 adaptive
resets. The emitted Q16.16 RTL has exact three-way spike-count parity at ten 1,000-step
operating points: 0/0/0/31/60/87/131/157/207/256 spikes at
`I=0/0.5/1/1.5/2/2.5/3.5/4/5/6`. The former 300-step `I=3` window is also exact at
36/36/36 after the compiler's candidate-reset/output correction.

One longer-window boundary remains explicit: at `I=3` over 1,000 steps the hand model and
schema runner report 111 spikes while Q16.16 RTL reports 112. Quantisation advances one
marginal `v >= theta` crossing between two evolving fixed-point states. The tests pin that
111/111/112 triplet directly; they do not hide it behind a general tolerance band or label
the boundary as exact parity.

The S5/H1 descriptor also registers the generated Q8.8 formal core and its depth-3
SymbiYosys/Z3 reset-spike safety proof. That bounded proof is structural safety evidence;
the Q16.16 operating set above remains the behavioural fidelity evidence.

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

## Polyglot acceleration

A single `step` is trivial, but an N-step run is a sequential RK4 recurrence
with a discontinuous spike reset that does not vectorise, so a compiled inner
loop genuinely beats Python. `simulate(n_steps, current, backend="auto")`
dispatches across the polyglot chain and returns `(trace, spikes)`:

```python
from sc_neurocore.neurons.models.mihalas_niebur import MihalasNieburNeuron

neuron = MihalasNieburNeuron()
trace, spikes = neuron.simulate(2_000_000, current=2.0)   # auto → Rust
```

The Mihalas-Niebur 2009 right-hand side is purely linear — additions,
multiplications and divisions, no transcendental functions — so every RK4 stage
is exact arithmetic. The Rust engine, Julia and Go backends therefore reproduce
the NumPy reference **bit-for-bit**: trace, spike count and the final
`(v, theta, i1, i2)` state all match exactly. Mojo fuses multiply-add; on this
platform it also matches bit-for-bit, but because FMA fusion is
compiler/version dependent it is validated as non-amplifying within a tight ULP
band (with identical spike counts) rather than asserted strictly exact. `auto`
selects Rust (the bit-exact backend shipped in the wheel).

### Measured throughput

2,000,000 RK4 steps, default tonic regime (`current=2.0`, 2857 spikes), median
of 5 repeats. Non-isolated loaded workstation per
`BROADCAST_2026-06-04_benchmark_core_isolation` — functional/regression
evidence, not an isolated-core figure. Reproduce with
`python benchmarks/bench_mihalas_niebur_simulate.py`.

| Backend | Median (ms) | Speed-up vs Python | Whole-trace parity |
|---------|------------:|-------------------:|--------------------|
| python  | 2381.79 | 1.0× | reference |
| rust (`auto`) | 85.72 | 27.8× | bit-exact (0) |
| go      | 86.21 | 27.6× | bit-exact (0) |
| mojo    | 92.63 | 25.7× | 0 measured (FMA, ULP-validated) |
| julia   | 95.83 | 24.9× | bit-exact (0) |

Artefact: `benchmarks/results/bench_mihalas_niebur_simulate.json`.

## Verification commands

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests/test_model_mihalas_niebur.py -q
PYTHONPATH=src .venv/bin/python -m pytest tests/test_mihalas_niebur_backends.py -q
PYTHONPATH=src .venv/bin/python -m pytest tests/test_cosimulation.py -k mihalas_niebur -q
cargo test --manifest-path engine/Cargo.toml mn_ --release
(cd hdl/formal/catalogue && sby -f sc_mihalasnieburneuron.sby)
(cd src/sc_neurocore/accel/go/neurons/mihalas_niebur && go build -buildmode=c-shared -o libmihalasniebur.so mihalas_niebur.go)
(cd src/sc_neurocore/accel/mojo/neurons && mojo build --emit shared-lib -o libmihalasniebur.so mihalas_niebur.mojo)
PYTHONPATH=src .venv/bin/python benchmarks/bench_mihalas_niebur_simulate.py --json benchmarks/results/bench_mihalas_niebur_simulate.json
```
