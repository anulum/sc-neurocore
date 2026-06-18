# EnergyLIFNeuron

**Module:** `sc_neurocore.neurons.models.energy_lif`<br>
**Reference:** Fardet & Levina, Neural Comput. 32(12), 2020; Blouw et al. 2019<br>
**Family:** Integrate-and-fire with metabolic energy constraint<br>
**State variables:** membrane voltage `v`, metabolic energy reserve `epsilon`

`EnergyLIFNeuron` models a leaky integrate-and-fire neuron whose input gain and
spike eligibility are constrained by a slow metabolic reserve. High activity
depletes `epsilon`; low energy weakens current response and prevents spiking
until recovery restores enough reserve.

## Equations

The continuous state evolves under piecewise-constant current:

$$
\frac{dV}{dt} =
\frac{-(V - V_{rest}) + R I \epsilon(t)}{\tau_m}
$$

$$
\frac{d\epsilon}{dt} =
\frac{\epsilon_0 - \epsilon}{\tau_\epsilon}
$$

Because `epsilon(t)` relaxes independently, the implementation uses the exact
constant-current flow for the coupled `(V, epsilon)` candidate. With
`a_m = exp(-dt/tau_m)`, `a_e = exp(-dt/tau_e)`, and
`delta_e = epsilon - epsilon_0`:

$$
\epsilon_{next} =
\epsilon_0 + \delta_\epsilon a_e
$$

$$
V_{next} =
V_{rest} + (V - V_{rest})a_m +
\frac{RI}{\tau_m}
\left[
\epsilon_0\tau_m(1-a_m) +
\delta_\epsilon a_m
\frac{\exp((1/\tau_m - 1/\tau_\epsilon)dt)-1}
{1/\tau_m - 1/\tau_\epsilon}
\right]
$$

When `tau_m` and `tau_e` are equal, the implementation uses the continuous
limit for the final fraction:

$$
\delta_\epsilon a_m dt
$$

Spike handling is evaluated against the candidate:

$$
V_{next} \geq V_{threshold}
\quad \text{and} \quad
\epsilon_{next} > 0.1
$$

On spike:

$$
V \leftarrow V_{reset}
$$

$$
\epsilon \leftarrow \max(0, \epsilon_{next} - \alpha)
$$

Without a spike, the exact candidates are committed directly.

## Parameters

| Parameter | Default | Contract | Description |
|-----------|--------:|----------|-------------|
| `v` | -70.0 | finite, -200 to 100 | Membrane voltage. |
| `epsilon` | 1.0 | finite, 0 to `epsilon_0` | Metabolic reserve. |
| `v_rest` | -70.0 | finite | Resting voltage. |
| `v_reset` | -70.0 | finite, -200 to 100 | Reset voltage after spike. |
| `v_threshold` | -50.0 | finite, above rest and reset | Spike threshold. |
| `tau_m` | 10.0 | positive finite | Membrane time constant. |
| `tau_e` | 500.0 | positive finite | Energy recovery time constant. |
| `alpha` | 0.1 | finite, non-negative | Energy cost per spike. |
| `epsilon_0` | 1.0 | finite, non-negative | Resting energy reserve. |
| `resistance` | 1.0 | positive finite | Input-current scaling. |
| `dt` | 1.0 | positive finite, not above `tau_m` or `tau_e` | Integration timestep. |

## Runtime Contract

- Constructor validation rejects non-finite voltages, non-positive time
  constants, non-positive resistance or timestep, negative energy reserves,
  negative spike cost, overfilled energy reserve, and threshold/reset geometry
  that would be immediately spiking at rest.
- `step(current)` rejects non-finite runtime current before evaluating the exact
  flow.
- Runtime state is revalidated before every step, so corrupted state is rejected
  before mutation.
- Exact-flow candidates must remain finite, with voltage inside -200 to 100 and
  energy inside `[0, epsilon_0]`.
- Python raises `ValueError` on invalid contracts. Go, Julia, Mojo, and Rust
  mirrors return `-1` on invalid runtime input or state while preserving the
  previous state.

## Polyglot Surfaces

| Surface | Path | Contract |
|---------|------|----------|
| Python reference | `src/sc_neurocore/neurons/models/energy_lif.py` | Stateful exact-flow implementation with exceptions on invalid contracts. |
| Go service | `src/sc_neurocore/accel/go/services/energy_lif.go` | Stateful exact-flow mirror with invalid-state sentinel. |
| Julia mirror | `src/sc_neurocore/accel/julia/neurons/energy_lif.jl` | Stateful exact-flow mirror with invalid-state sentinel. |
| Mojo helper | `src/sc_neurocore/accel/mojo/kernels/energy_lif.mojo` | Stateless exact-flow helper functions for accelerator use. |
| Rust safety | `src/sc_neurocore/accel/rust/safety/energy_lif.rs` | Stateful exact-flow safety mirror with invalid-state sentinel. |

## Behaviour

At full energy, the membrane responds to `R * I`. As repeated spikes deplete
`epsilon`, the effective drive decreases and the spike gate eventually blocks
firing. During silence, `epsilon` recovers exponentially toward `epsilon_0`.

Three regimes are covered by module tests:

- Subthreshold drive: weak current does not cross threshold at full energy.
- Energy-gated spiking: sufficient current spikes while `epsilon > 0.1`.
- Metabolic silence: depleted energy blocks spikes until recovery.

## Analytical Properties

### Energy-Modulated Input Gain

The input term is scaled by the current metabolic reserve:

$$
R_{eff}(t) = R\epsilon(t)
$$

This creates a closed negative feedback loop:

| Energy state | Effective gain | Consequence |
|--------------|----------------|-------------|
| `epsilon = 1.0` | full `R * I` drive | The membrane follows the ordinary LIF response at full reserve. |
| `epsilon = 0.5` | half `R * I` drive | The same current produces a weaker depolarisation. |
| `epsilon <= 0.1` | low drive and spike gate closed | The neuron cannot emit a spike until recovery crosses the gate. |

The exact-flow implementation preserves this time-varying input gain inside the
membrane integral. It does not freeze `epsilon` at the start of the step and does
not approximate the recovery term with a raw Euler increment.

### Energy Recovery Dynamics

Without spikes, energy recovery follows the closed-form first-order relaxation:

$$
\epsilon(t) = \epsilon_0 - (\epsilon_0 - \epsilon_i)e^{-t/\tau_\epsilon}
$$

For the default `tau_e = 500 ms`, recovery is intentionally slower than the
default membrane relaxation:

| Initial reserve | Target reserve | Approximate recovery time |
|-----------------|----------------|---------------------------|
| 0.0 | 0.5 | 347 ms |
| 0.0 | 0.9 | 1,151 ms |
| 0.1 | 0.5 | 294 ms |

### Spike-Cost Balance

At a sustained operating point, average spike cost balances recovery:

$$
f\alpha \approx \frac{\epsilon_0 - \epsilon_{ss}}{\tau_\epsilon}
$$

For `epsilon_ss = 0.5`, `epsilon_0 = 1.0`, `tau_e = 500 ms`, and
`alpha = 0.1`, the balance gives approximately:

$$
f \approx \frac{1.0 - 0.5}{500 \cdot 0.1} = 0.01 \text{ spikes/ms}
$$

That corresponds to about 10 Hz under this simplified balance argument. The
actual emitted rate also depends on the voltage trajectory, reset timing,
current amplitude, and the hard `epsilon > 0.1` spike gate.

### Comparison With Standard LIF

| Feature | EnergyLIF | Standard LIF |
|---------|-----------|--------------|
| State variables | `v`, `epsilon` | `v` |
| Input gain | `R * epsilon * I` | `R * I` |
| Spike eligibility | voltage threshold and energy gate | voltage threshold |
| Spike cost | `epsilon -= alpha` after candidate recovery | none |
| Recovery process | slow metabolic relaxation | not represented |
| Main adaptation mechanism | intrinsic reserve depletion | absent unless another adaptation variable is added |

EnergyLIF therefore acts as a compact adaptation model with an explicit
metabolic reserve rather than a threshold or hyperpolarising adaptation current.

## Numerical Considerations

- The continuous two-state system is linear under constant input current because
  `epsilon(t)` is independent of voltage within a step.
- The implementation uses the exact coupled flow for `(v, epsilon)`, including
  the integral of recovering energy through the membrane equation.
- The branch for `tau_m == tau_e` uses the analytic limit
  `delta_e * exp(-dt/tau_m) * dt`, avoiding division by a near-zero rate
  difference.
- Spike reset and energy cost remain discontinuous events after the exact
  candidate is evaluated.
- Candidate-first validation prevents partial state mutation when a runtime
  current, corrupted state, or candidate value is invalid.
- The committed benchmark is local non-isolated evidence. Production benchmark
  claims require isolated CPU/core execution and recorded host-load context.

## Measured Benchmark Evidence

Benchmark artifact:
`benchmarks/results/local_python_2026-06-18_energy_lif_exact_flow.json`

Evidence class: `local_regression_non_isolated`. The artifact records local
functional and parity evidence only; it is not an isolated hardware-performance
claim and does not claim production speed.

Configuration:

| Field | Value |
|-------|------:|
| Steps per repeat | 200,000 |
| Repeats | 5 |
| Constant current | 50.0 |
| Expected spike parity | 2,550 spikes on every backend |

Measured timing on `aaarthuus`, Linux 6.17.0-35-generic, Python 3.12.3:

| Backend | Median ns/step | Min ns/step | Max ns/step | Spikes |
|---------|---------------:|------------:|------------:|-------:|
| Python | 1,692.761055 | 1,631.32507 | 1,947.13414 | 2,550 |
| Rust safety | 33.077485 | 32.78194 | 33.554295 | 2,550 |
| Go service | 42.83 | 42.08 | 43.72 | 2,550 |
| Julia mirror | 29.38143 | 28.853575 | 31.57255 | 2,550 |
| Mojo helper | 3.738814848475158 | 3.4707499435171485 | 3.9324749377556145 | 2,550 |

The benchmark gate
`energy-lif-exact-flow-multibackend-local-regression` requires numeric timing
rows for Python, Rust, Go, Julia, and Mojo, plus zero-tolerance spike-count
parity across all five backends.

## Verification

Focused checks for this hardening slice:

| Command | Coverage |
|---------|----------|
| `ruff check src/sc_neurocore/neurons/models/energy_lif.py tests/test_model_energy_lif.py benchmarks/bench_model_energy_lif.py` | Python lint for model, tests, and benchmark. |
| `mypy --strict src/sc_neurocore/neurons/models/energy_lif.py benchmarks/bench_model_energy_lif.py` | Strict typing for the Python model and benchmark harness. |
| `pytest tests/test_model_energy_lif.py -q` | Exact-flow math, invalid-state preservation, network, and analysis coverage. |
| `go test src/sc_neurocore/accel/go/services/energy_lif.go src/sc_neurocore/accel/go/services/energy_lif_test.go` | Go exact-flow candidate, spike-cost, invalid-state, and benchmark hook coverage. |
| `rustc --test src/sc_neurocore/accel/rust/safety/energy_lif.rs -o /tmp/energy_lif_safety_test && /tmp/energy_lif_safety_test` | Rust safety exact-flow and invalid-state coverage. |
| `julia --project=. -e '<EnergyLIF exact-flow smoke>'` | Julia exact-flow candidate smoke. |
| `mojo -I src/sc_neurocore/accel/mojo/kernels <EnergyLIF exact-flow smoke>` | Mojo helper smoke. |
| `python benchmarks/bench_model_energy_lif.py` | Regenerates the five-backend local benchmark artifact. |
| `python tools/benchmark_evidence_gate.py --manifest /tmp/energy_lif_gate.json --output /tmp/energy_lif_gate_report.json` | Validates required timing rows, source hashes, and spike parity. |

## Test Coverage

| Category | Coverage |
|----------|----------|
| Isolation | construction, binary output, subthreshold drive, spiking drive, energy depletion, recovery, non-negative reserve, reset |
| Validation | non-finite voltage/current rejection, non-positive scales, invalid threshold geometry, overfilled reserve, corrupted runtime-state preservation |
| Exact flow | exact candidate commit, separation from forward Euler, candidate-ordered spike energy cost |
| Network | population construction, Poisson-driven network spiking, recurrent projection wiring |
| Analysis | firing-rate and spike-count integration through the public analysis helpers |
| Native mirrors | Go exact-flow tests, Rust safety tests, Julia smoke, Mojo smoke |
| Benchmark | Python, Rust, Go, Julia, and Mojo rows with zero-tolerance spike-count parity |

## Usage

```python
from sc_neurocore.neurons.models.energy_lif import EnergyLIFNeuron

neuron = EnergyLIFNeuron()
spikes = [neuron.step(50.0) for _ in range(1000)]
print(sum(spikes), neuron.v, neuron.epsilon)
```

Invalid runtime current preserves state:

```python
neuron = EnergyLIFNeuron(v=-65.0, epsilon=0.5)
before = (neuron.v, neuron.epsilon)
try:
    neuron.step(float("nan"))
except ValueError:
    assert (neuron.v, neuron.epsilon) == before
```
