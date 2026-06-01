# Prescott Neuron

## Scope

`PrescottNeuron` implements the reduced two-state Prescott et al. 2008 excitability model used to study Type I, Type II, and Type III transitions through the position of the slow potassium nullcline. The public state is membrane voltage `v` and slow recovery variable `w`.

The implementation is a reduced conductance model. It keeps the fast activation variable at instantaneous steady state and evolves the slow recovery gate explicitly as a state variable.

## Differential system

For external current `I`, the maintained equations are:

```text
dv/dt = -g_fast m_inf(v) (v - e_fast)
        -g_slow w (v - e_slow)
        -g_l (v - e_l)
        + I

dw/dt = phi (w_inf(v) - w) / tau_w
```

The activation curves are:

```text
m_inf(v) = 1 / (1 + exp(-(v + 20) / 15))
w_inf(v) = 1 / (1 + exp(-(v - beta_w) / gamma_w))
```

The spike output is an upward threshold-crossing indicator. A public `step(current)` call returns `1` only when the committed candidate voltage crosses `v_threshold` from below.

## Integration contract

The Python reference, Rust engine, Go service, Julia mirror, and standalone Rust safety surface now use a candidate-first RK4 update over the coupled `(v, w)` ODE. The prior direct Euler increment is not used by the maintained runtime path.

The public step contract is:

```python
from sc_neurocore.neurons.models.prescott import PrescottNeuron

neuron = PrescottNeuron()
spike = neuron.step(50.0)
state = (neuron.v, neuron.w)
```

Validation is fail-closed:

- Runtime current must be finite.
- Voltage must be finite.
- Recovery state `w` must remain finite and in `[0, 1]`.
- Conductances must be finite and non-negative.
- `gamma_w`, `tau_w`, and `dt` must be finite and positive.
- `phi` must be finite and non-negative.
- Non-finite derivatives or invalid RK4 candidates are rejected before committed state changes.

## Polyglot surfaces

| Surface | File | Status |
| --- | --- | --- |
| Python reference | `src/sc_neurocore/neurons/models/prescott.py` | Candidate-first RK4 source of truth |
| Rust engine / PyO3 | `engine/src/neurons/biophysical.rs`, `engine/src/pyo3_neurons.rs` | Candidate-first RK4 exposed through `get_state()` |
| Go service | `src/sc_neurocore/accel/go/services/prescott.go` | Candidate-first RK4 with state-preserving invalid-input handling |
| Julia mirror | `src/sc_neurocore/accel/julia/neurons/prescott.jl` | Candidate-first RK4 returning `-1` on invalid input or candidate |
| Mojo kernel note | `src/sc_neurocore/accel/mojo/kernels/prescott.mojo` | RK4 stage contract documented for future compiled kernel work |
| Rust safety surface | `src/sc_neurocore/accel/rust/safety/prescott.rs` | Standalone candidate-first RK4 safety reference |

## Current benchmark evidence

Benchmark command:

```bash
PYTHONPATH=src .venv/bin/python benchmarks/bench_prescott.py
```

Current local result from `benchmarks/results/bench_prescott.json`:

| Backend | Steps/s | Wall ms for 100,000 steps | Speedup |
| --- | ---: | ---: | ---: |
| Python reference | 187,521 | 533.27 | 1.00x |
| Rust PyO3 | 5,132,065 | 19.48 | 27.37x |

Parity evidence: the Python reference and Rust PyO3 class produced measured `max_abs_delta = 0.0` and `spikes_delta = 0` over 10,000 steps at `current = 50.0` on the current build.

## Test evidence

Module-specific evidence lives in `tests/test_model_prescott.py`. The suite checks:

- Deterministic RK4 reference separation from the former Euler candidate.
- Invalid parameter rejection and state preservation on invalid runtime current, corrupted state, and non-finite derivative paths.
- Slow oscillation, current response, beta-w excitability modulation, and recovery timescale behaviour.
- Population, network, and spike-count analysis wiring through the production boundary.

Focused verification on 2026-06-01:

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests/test_model_prescott.py -q
```

Evidence: 38 module-specific tests passed, and `src/sc_neurocore/neurons/models/prescott.py` reported 100% statement exercise in the focused run.
