# SigmoidRateNeuron

`SigmoidRateNeuron` is a deterministic continuous-rate unit with one state,
`r`, and a branch-stable logistic target. It implements the reduced scalar
equation

\[
\tau \frac{dr}{dt} = -r + \sigma\!\left(\beta(I-\theta)\right),
\qquad
\sigma(x)=\frac{1}{1+e^{-x}}.
\]

Wilson and Cowan (1972) motivate the population-rate and sigmoid framework,
but their paper derives coupled excitatory and inhibitory population dynamics.
This class intentionally isolates a single relaxation-to-sigmoid motif. It is
not the complete Wilson-Cowan system; use `WilsonCowanUnit` for that family.

## Numerical contract

For constant input during one step, the target is fixed, so the maintained
finite-step update is the exact first-order relaxation

\[
r_{n+1}=e^{-\Delta t/\tau}r_n+
\left(1-e^{-\Delta t/\tau}\right)\sigma\!\left(\beta(I-\theta)\right).
\]

This update is a convex combination when `r` starts in `[0, 1]`, `tau > 0`,
and `dt > 0`. Large timesteps therefore relax directly toward the target
without forward-Euler overshoot.

The implementation validates the complete mutable contract before updating:

- `r`, `tau`, `beta`, `theta`, `dt`, and input must be finite;
- `r` must lie in `[0, 1]`;
- `tau` and `dt` must be positive;
- finite multiplication overflow in `beta * (I - theta)` saturates to the
  corresponding logistic limit;
- a non-saturating NaN argument fails before state mutation.

`reset()` restores only `r = 0`. It preserves `tau`, `beta`, `theta`, and `dt`.

## Parameters

| Field | Default | Meaning |
|---|---:|---|
| `r` | `0.0` | live normalised rate |
| `tau` | `10.0` | relaxation time constant |
| `beta` | `1.0` | logistic gain |
| `theta` | `0.0` | logistic midpoint |
| `dt` | `0.1` | simulation step |

## Scalar and batch use

```python
from sc_neurocore.neurons.models.sigmoid_rate import SigmoidRateNeuron

neuron = SigmoidRateNeuron(r=0.25, tau=10.0, beta=2.0, theta=1.0, dt=0.5)
value = neuron.step(3.0)

neuron.reset()
trace = neuron.simulate(512, current=3.0, backend="auto")
```

Supported batch names are `python`, `rust`, `julia`, `go`, and `mojo`.
An explicit unavailable backend raises `RuntimeError`; it is never replaced by
a silent Python surrogate. `auto` uses the order recorded by the current
benchmark evidence and falls through availability only. A successful batch
commits the final returned rate atomically.

## Executed backend evidence

The promotion contract uses
`r=0.25`, `tau=10`, `beta=2`, `theta=1`, `dt=0.5`, and `I=3`.
Every lane executes the real recurrence and transports the full parameter set.

| Surface | Public execution boundary | Result |
|---|---|---|
| Python | `SigmoidRateNeuron.simulate(..., backend="python")` | reference trace |
| Rust engine | modular `py_sigmoid_rate_simulate` PyO3 batch | complete trace and final rate |
| Rust safety | standalone `sigmoid_rate.rs` compiled and tested | 8 tests pass; reset preserves configuration |
| Julia | `SigmoidRateAccel.simulate_trace` | complete trace and final rate |
| Go | service plus generated `sigmoid_rate_simulate_c` shared library | atomic complete trace and final rate |
| Mojo | exported `sigmoid_rate_simulate_c` shared library | atomic complete trace and final rate |

Python, Rust, Julia, and Go are byte-identical on the enrolled 200,000-step
trajectory. Mojo's maximum absolute difference is
`3.0753177782116836e-14`, below the declared `5e-12` tolerance. A tolerance is
used because separate `exp` implementations are not promised bit-identical.

The batch is a rate trace, not a spike train. No positive-rate-as-spike count is
part of the contract.

## Fixed-point co-simulation

The paired TOML and JSON schemas preserve the exact-relaxation hand trajectory
within `5e-12` across a 256-step sign-changing input. The production equation
compiler emits a Q32.32 Verilog module from the same schema. Icarus Verilog
co-simulation reads the module's public `r_out` and `spike_out` ports over that
trajectory: every rate remains in `[0, 1]`, every event output remains zero, and
the measured maximum absolute rate difference from the hand model is
`0.014879114367180313`, below the declared `0.016` envelope.

The bound reflects the 0.125-argument spacing of the sigmoid and
exponential-relative lookup tables. This is generated-RTL co-simulation
evidence only. It does not claim bit identity for transcendental functions,
formal equivalence, synthesis, timing closure, device execution, or PPA.

## Local benchmark evidence

The committed artifact is
`benchmarks/results/local_python_2026-07-14_sigmoid_rate.json`. It measures five
200,000-step batches per backend after warm-up on one pinned logical CPU.

| Backend | Median ns/step | Trace mismatch count | Maximum absolute difference |
|---|---:|---:|---:|
| Python | 368.937860 | 0 | 0 |
| Rust | 234.925815 | 0 | 0 |
| Julia | 87.466740 | 0 | 0 |
| Go | 486.350110 | 0 | 0 |
| Mojo | 73.048140 | 0 | `3.0753177782116836e-14` |

These are same-host, non-exclusive local regression timings. The recorded load
average was high, no runtime cpuset shield was claimed, and the artifact makes
no production-speed, cross-host, hardware, or universal ranking claim.

## Pipeline boundary

`step()` returns a continuous `float`, not a binary event. Construction through
`Population` is supported, but a spiking event counter must not interpret every
positive rate as a spike. Rate-network scheduling and coupling remain the
caller's responsibility.

The four acceleration languages and the Rust engine remain floating-point
software execution surfaces. The separate generated Q32.32 surface carries
only the bounded co-simulation claim above.

## Files and reproduction

- Python reference: `src/sc_neurocore/neurons/models/sigmoid_rate.py`
- public dispatcher: `src/sc_neurocore/accel/sigmoid_rate.py`
- Rust engine: `engine/src/neurons/rate/sigmoid_rate.rs` and
  `engine/src/bindings/sigmoid_rate.rs`
- independent Rust safety: `src/sc_neurocore/accel/rust/safety/sigmoid_rate.rs`
- Julia: `src/sc_neurocore/accel/julia/neurons/sigmoid_rate.jl`
- Go: `src/sc_neurocore/accel/go/services/sigmoid_rate.go` and
  `src/sc_neurocore/accel/go/neurons/sigmoid_rate/sigmoid_rate.go`
- Mojo: `src/sc_neurocore/accel/mojo/kernels/sigmoid_rate.mojo`
- paired schemas:
  `src/sc_neurocore/neurons/model_schemas/sigmoid_rate.{toml,json}`
- generated-RTL co-simulation: `tests/test_cosim_sigmoid_rate.py`

```bash
PYTHONPATH=bridge:src:. .venv/bin/python -m pytest -q \
  tests/test_model_sigmoid_rate.py \
  tests/test_cosim_sigmoid_rate.py \
  tests/test_sigmoid_rate_backend_loading.py \
  tests/test_sigmoid_rate_backends.py \
  tests/test_bench_sigmoid_rate.py

taskset -c 4 env PYTHONPATH=bridge:src:. .venv/bin/python \
  benchmarks/bench_model_sigmoid_rate.py \
  --json benchmarks/results/local_python_2026-07-14_sigmoid_rate.json
```

## Reference

Wilson, H. R., and Cowan, J. D. (1972). *Excitatory and Inhibitory
Interactions in Localized Populations of Model Neurons*. Biophysical Journal,
12(1), 1-24. [doi:10.1016/S0006-3495(72)86068-5](https://doi.org/10.1016/S0006-3495(72)86068-5).
