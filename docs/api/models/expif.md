# ExpIFNeuron

`ExpIFNeuron` is the deterministic exponential integrate-and-fire model from
Fourcaud-Trocmé, Hansel, van Vreeswijk and Brunel (2003). The maintained model
has one voltage state plus an optional post-spike refractory remainder.

Primary source: N. Fourcaud-Trocmé et al., *Journal of Neuroscience* 23(37),
11628–11640, 2003,
[doi:10.1523/JNEUROSCI.23-37-11628.2003](https://doi.org/10.1523/JNEUROSCI.23-37-11628.2003).

## Equation and units

The source current balance is

\[
C\frac{dV}{dt} = -g_L(V-V_L)
  + g_L\Delta_T\exp\!\left(\frac{V-V_T}{\Delta_T}\right) + I.
\]

After division by \(g_L\), the implementation evaluates

\[
\tau\frac{dV}{dt} = -(V-V_{rest})
  + \Delta_T\exp\!\left(\frac{V-V_{rh}}{\Delta_T}\right)
  + I_{norm}, \qquad \tau=C/g_L.
\]

The public `current` argument is therefore \(I_{norm}=I/g_L\), expressed in
the same millivolt-equivalent scale as the other terms. It is not an unscaled
current in picoamperes.

`v_rh` is the soft threshold of the exponential current. `v_threshold` is a
separate finite numerical cutoff used to register a spike. Confusing the two
changes the response towards a leaky integrate-and-fire model; the source
specifically discusses this cutoff dependence.

## Maintained defaults

| Field | Default | Unit | Meaning |
|---|---:|---|---|
| `v` | −65.0 | mV | initial membrane voltage |
| `v_rest` | −65.0 | mV | leak reversal voltage |
| `v_reset` | −68.0 | mV | post-spike reset voltage |
| `v_threshold` | +30.0 | mV | finite numerical spike cutoff |
| `v_rh` | −59.9 | mV | soft exponential threshold |
| `delta_t` | 3.48 | mV | exponential slope factor |
| `tau` | 10.0 | ms | membrane time constant |
| `dt` | 0.02 | ms | RK4 macro-step |
| `refractory_period` | 0.0 | ms | reset hold loaded after a spike |
| `refractory_remaining` | 0.0 | ms | runtime reset-hold remainder |

The voltage, reset, soft threshold, slope factor and membrane time constant are
the source's fitted EIF values. The paper used a +30 mV finite cutoff for this
fit and a 1.7 ms refractory interval in its Wang–Buzsáki comparison protocol.
SC-NeuroCore keeps `refractory_period=0.0` as the deterministic schema-to-RTL
contract; set it to `1.7` to reproduce that particular protocol facet.

## Numerical contract

Each macro-step is candidate-first classical RK4:

```text
k1 = f(v)
k2 = f(v + dt*k1/2)
k3 = f(v + dt*k2/2)
k4 = f(v + dt*k3)
candidate = v + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)
```

RK4 stage voltages are bounded at `v_threshold` before evaluating the
exponential. This is an event-surface bound: it leaves every pre-cutoff stage
unchanged and avoids evaluating an irrelevant divergent post-event voltage. It
is not the former arbitrary exponential-argument clip.

If `candidate >= v_threshold`, the step emits `1`, stores `v_reset`, and loads
`refractory_period`. While `refractory_remaining > 0`, voltage stays at
`v_reset`, the remainder decreases by `dt`, and the step emits `0`. Otherwise
the step stores the candidate and emits `0`.

Construction and runtime checks fail closed for non-finite values, non-positive
scales, invalid threshold relationships, voltage at or above the cutoff between
steps, or refractory state outside its declared interval. Native C-ABI kernels
validate the complete run before writing their output buffers.

## Python API

```python
from sc_neurocore.neurons.models.expif import ExpIFNeuron

neuron = ExpIFNeuron()
trace, spikes = neuron.simulate(1_000, current=20.0, backend="auto")
assert spikes == 2
assert trace.shape == (1_000,)
```

`trace[t]` is the post-step voltage, including reset and refractory-held
samples. A successful run commits its final voltage and refractory remainder to
the instance. A rejected compiled run leaves both fields unchanged.

Accepted backend selectors are `auto`, `python`, `rust`, `julia`, `go`, and
`mojo`.

| Backend | Public contract |
|---|---|
| Python | complete state and parameter contract |
| Rust engine | factory-default contract; explicit rejection otherwise |
| Julia | complete state and parameter contract |
| Go C ABI | complete state and parameter contract |
| Mojo C ABI | complete state and parameter contract |

The Rust safety kernel is additionally maintained and tested as a fail-closed
native surface. `auto` follows the measured Julia → Go → Mojo → compatible Rust
→ Python order recorded by `benchmarks/bench_model_expif.py`. The Rust lane is
considered only when the instance matches its factory contract.

## Reproducibility and parity

The descriptor's reference configuration is 1,000 default-parameter steps at
`current=20.0`. It emits two spikes, ends at
`v=-59.00168087076545`, and hashes the float64 voltage trace to
`cd3fc0cd092a9924e7477426c3d8622510ea01c37ed51b83188b38f7e026a1c6`.

The enrolled 1,000-step event goldens are:

| Normalised current | 0 | 5 | 10 | 20 | 50 | 100 |
|---:|---:|---:|---:|---:|---:|---:|
| Spike count | 0 | 0 | 1 | 2 | 5 | 9 |

The hand recurrence, TOML schema and JSON schema have exact event parity and a
float64 state envelope of at most `2e-10`. The generated RTL is enrolled at
Q32.32 with the same event counts over this operating set. Q16.16 is not
claimed: the steep source exponential crosses its active fixed-point range and
does not preserve these event counts.

Evidence anchors:

- `tests/test_expif_backends.py` — executable Python/Rust/Julia/Go/Mojo parity,
  complete ABI state, dispatch and rejection paths;
- `tests/test_cosim_exp_if.py` — independent source recurrence, paired schemas
  and Q32.32 RTL event parity;
- `tests/test_reference_exp_if.py` — DOI-bound driven reference trace;
- `src/sc_neurocore/neurons/model_descriptors/ExpIFNeuron.toml` — reproducible
  digest and dual-axis readiness facets;
- `hdl/formal/catalogue/sc_exp_if.sby` — bounded formal safety job.

## Benchmark evidence

`benchmarks/bench_model_expif.py` measures the public `simulate()` path for all
five backends under one workload. It records source hashes, event and trace
parity, CPU affinity, host load, governor state and runtime versions. The
committed result is
`benchmarks/results/local_python_2026-06-16_expif_rk4.json`; its timings are a
local regression record, not a general throughput or hardware-isolation claim.

## Scope limits

- The model represents exponential spike initiation in a point-neuron voltage
  equation; it does not reproduce ionic-channel waveforms.
- The zero-refractory default is chosen for deterministic co-simulation and is
  not the source paper's entire fitted comparison protocol.
- Fixed-point validation currently proves Q32.32 event parity only at the
  declared operating points and horizon.
- Higher silicon readiness requires separate synthesis, timing and hardware
  evidence.
