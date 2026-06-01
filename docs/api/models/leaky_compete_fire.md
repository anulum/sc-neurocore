# LeakyCompeteFireNeuron

**Module:** `sc_neurocore.neurons.models.leaky_compete_fire`
**Reference:** Oster, Douglas & Liu, Neural Computation 21(9), 2009
**Family:** Multi-unit winner-take-all circuit with lateral inhibition
**State variables:** `v` (one membrane potential per competing unit)

---

## Model contract

`LeakyCompeteFireNeuron` represents `n_units` competing leaky integrate-and-fire units with shared membrane timescale, threshold, and subtractive lateral inhibition. Each step returns one binary spike value per unit.

The maintained contract is vector-valued:

```python
spikes = neuron.step([i_0, i_1, ..., i_n])
```

A scalar current is broadcast to every unit for pipeline compatibility.

---

## Equations

Each unit follows the linear current-based membrane equation

$$\tau \frac{dV_i}{dt} = -V_i + I_i$$

For constant current over one timestep, SC-NeuroCore uses the exact first-order flow

$$V_i(t + \Delta t) = I_i + (V_i(t) - I_i)\exp(-\Delta t / \tau)$$

rather than a raw Euler increment. This keeps the membrane inside the relaxation envelope for large timesteps when the state and input are finite.

---

## Spike and inhibition pass

All membrane candidates are computed first from the old vector state. Spike detection then runs sequentially from unit `0` to `n_units - 1`:

$$\text{if } V_i \geq V_\theta:$$

$$\quad \text{spike}_i = 1$$

$$\quad V_i \leftarrow 0$$

$$\quad \forall j \neq i:\; V_j \leftarrow \max(0, V_j - w_{inh})$$

This preserves the published spiking WTA mechanism: a firing unit resets, and non-firing competitors receive subtractive lateral inhibition clamped at zero.

---

## Parameters

| Parameter | Default | Unit | Description |
|-----------|---------|------|-------------|
| `n_units` | 4 | count | Number of competing units |
| `v` | `[0.0] * n_units` | a.u. | Membrane vector |
| `tau` | 10.0 | ms | Shared membrane time constant |
| `v_threshold` | 1.0 | a.u. | Spike threshold |
| `w_inh` | 0.5 | a.u. | Lateral inhibition decrement per spike |
| `dt` | 1.0 | ms | Integration timestep |

`n_units` must be a positive integer. `tau` and `dt` must be finite and positive. `v_threshold`, `w_inh`, all voltages, and all currents must be finite, with `w_inh >= 0`.

---

## Numerical safety

The Python reference, Go service, Julia mirror, Mojo scalar helpers, and Rust safety surface now share these boundaries:

- validate vector length and finite state before integration;
- validate finite scalar or vector currents before integration;
- compute exact-relaxation candidates before mutation;
- reject non-finite candidates before committing any state;
- preserve state on invalid runtime input or corrupted runtime parameters.

---

## Winner-take-all behaviour

With asymmetric constant input, the highest-current unit reaches threshold most often and inhibits lower-drive units. With `w_inh = 0`, units behave as independent exact-relaxation LIF units. Larger non-negative `w_inh` values increase competitive suppression and reduce loser firing rates.

When multiple candidates cross threshold on the same step, lower-indexed units are processed first. This deterministic ordering is part of the implementation contract and is covered by module-specific tests.

---

## Pipeline compatibility

`Population` assumes scalar-valued `v` for automatic voltage synchronisation, so this multi-unit vector model remains an isolation and custom-workflow model rather than a default `Population` member. The module-specific tests explicitly check this boundary. Standalone usage and custom workflows can call `step(currents)` directly.

The native mirrors expose equivalent vector semantics where their language surface supports vectors. The Mojo file provides scalar exact-relaxation and spike helper primitives for per-unit kernels.

---

## Performance evidence

Fresh local benchmark evidence:

| Runtime | Command | Median |
|---------|---------|--------|
| Python | `PYTHONPATH=src .venv/bin/python benchmarks/bench_model_leaky_compete_fire.py` | 3638.31396 ns/step |

Benchmark artefact: `benchmarks/results/local_python_2026-06-01_leaky_compete_fire.json`.

The benchmark runs 5 repeats of 50,000 exact-relaxation WTA steps with currents `[5.0, 2.5, 1.25, 0.5]`, preserving the spike-and-inhibit path rather than measuring a silent subthreshold-only trace.
