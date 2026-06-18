# SFANeuron

**Module:** `sc_neurocore.neurons.models.sfa`
**Reference:** Benda & Herz 2003
**Family:** Integrate-and-fire with spike-frequency adaptation
**State variables:** `v` (membrane voltage), `g_sfa` (adaptation conductance)

`SFANeuron` models spike-frequency adaptation with a membrane voltage and a
slow potassium-like adaptation conductance. The adaptation conductance
increases after each spike and decays between spikes, so repeated firing
lengthens the inter-spike interval.

---

## Equations

### Membrane voltage

$$
\tau_m \frac{dV}{dt}
= -(V - V_{\text{rest}})
- g_{\text{sfa}}(V - E_K)
+ R I
$$

where:

- `V` is membrane voltage.
- `g_sfa` is the adaptation conductance.
- `E_K` is the potassium reversal potential.
- `R I` is the resistance-scaled external drive.

### Adaptation conductance

$$
\frac{dg_{\text{sfa}}}{dt}
= -\frac{g_{\text{sfa}}}{\tau_{\text{sfa}}}
$$

### Spike and reset

The implementation evaluates the spike condition against the RK4 voltage
candidate:

$$
V_{\text{candidate}} \geq V_\theta
$$

If the candidate crosses threshold:

$$
V \leftarrow V_{\text{reset}}, \qquad
g_{\text{sfa}} \leftarrow g_{\text{sfa,candidate}} + \Delta g
$$

The reset does not discard the adaptation candidate. This preserves the
between-spike conductance decay and then applies the spike-triggered increment.

---

## Implementation Contract

The production step is candidate-first:

```python
def step(self, current: float) -> int:
    current = self._finite(current, "current")
    v, g_sfa = self._validated_state()
    v_next, g_next = self._rk4_candidate(v, g_sfa, current)
    if v_next >= self.v_threshold:
        self.v = self.v_reset
        self.g_sfa = g_next + self.delta_g
        return 1
    self.v = v_next
    self.g_sfa = g_next
    return 0
```

The full `(v, g_sfa)` ODE is advanced with a fourth-order Runge-Kutta
candidate. No state is mutated until the candidate passes validation.

### Fail-closed checks

The Python reference rejects invalid runtime state before mutation:

- voltage state and voltage parameters must be finite;
- `v` and `v_reset` must stay inside the safety envelope `[-200, 100]`;
- `g_sfa` must be finite, non-negative, and below the conductance envelope;
- `tau_m`, `tau_sfa`, `resistance`, and `dt` must be finite and positive;
- `delta_g` must be finite, non-negative, and below the conductance envelope;
- `current` must be finite;
- RK4 candidates must remain finite and inside their envelopes;
- post-spike `g_sfa_candidate + delta_g` must remain finite and bounded.

The Go, Julia, and Rust mirrors preserve previous state on invalid input and
return an invalid-step signal. The Mojo mirror is stateless and returns invalid
candidates or `-1` for malformed input.

---

## Parameters

| Parameter | Default | Unit | Description |
|-----------|---------|------|-------------|
| `v` | -70.0 | mV | Membrane voltage |
| `g_sfa` | 0.0 | a.u. | Adaptation conductance |
| `v_rest` | -70.0 | mV | Resting potential |
| `v_reset` | -70.0 | mV | Post-spike reset voltage |
| `v_threshold` | -50.0 | mV | Spike threshold |
| `tau_m` | 10.0 | ms | Membrane time constant |
| `tau_sfa` | 200.0 | ms | Adaptation decay time constant |
| `delta_g` | 0.5 | a.u. | Spike-triggered adaptation increment |
| `e_k` | -80.0 | mV | Potassium reversal potential |
| `resistance` | 1.0 | MOhm | Input resistance |
| `dt` | 1.0 | ms | Integration timestep |

---

## Adaptation Behaviour

The adaptation current is:

$$
I_{\text{sfa}} = g_{\text{sfa}}(V - E_K)
$$

During depolarisation, `V` is above `E_K`, so the adaptation current opposes
the external drive. A spike adds `delta_g`, and subsequent steps decay
`g_sfa` through the RK4 conductance candidate. The result is spike-frequency
adaptation: early inter-spike intervals are shorter than late inter-spike
intervals.

### Measured single-neuron dynamics

Measured locally on 2026-06-18 with the RK4 implementation, 10,000 steps, and
default parameters.

| Current | Spikes | Early mean ISI | Late mean ISI | Final `g_sfa` | Final `v` |
|---------|-------:|---------------:|--------------:|--------------:|----------:|
| 0.0 | 0 | - | - | 0.000000 | -70.000000 |
| 10.0 | 0 | - | - | 0.000000 | -60.000000 |
| 20.0 | 0 | - | - | 0.000000 | -50.000000 |
| 30.0 | 54 | 168.2 | 188.0 | 0.441392 | -52.544444 |
| 50.0 | 123 | 52.8 | 83.0 | 1.338653 | -54.798522 |
| 100.0 | 292 | 4.4 | 35.0 | 2.910047 | -52.190676 |

The `current=50.0` row shows the adaptation signature directly: late ISIs are
longer than early ISIs after `g_sfa` accumulates.

### No-adaptation boundary

Setting `delta_g=0.0` removes spike-triggered adaptation. The test suite checks
that the resulting spike train has a near-constant late ISI coefficient of
variation.

### Timescale controls

- Smaller `tau_sfa` makes adaptation decay faster and permits more spikes.
- Larger `tau_sfa` keeps adaptation active longer and suppresses later spikes.
- Larger `delta_g` increases the post-spike adaptation jump and reduces firing.

---

## Analytical Properties

When `g_sfa = 0` and no spike reset occurs, the continuous subthreshold
steady-state voltage is:

$$
V_{ss} = V_{\text{rest}} + R I
$$

The corresponding continuous rheobase is:

$$
I_{\text{rheo}}
= \frac{V_\theta - V_{\text{rest}}}{R}
= 20
$$

With the discrete RK4 step and strict threshold comparison, `current=20.0`
approaches the threshold without producing spikes in the 10,000-step measured
run. The benchmark and tests therefore record discrete spike behavior rather
than promoting a continuous-limit spike count.

At nonzero `g_sfa`, the effective current required to reach threshold rises:

$$
I_{\text{eff}}
= \frac{
V_\theta - V_{\text{rest}}
+ g_{\text{sfa}}(V_\theta - E_K)
}{R}
$$

This expression explains why the neuron slows after repeated spikes.

---

## Numerical Considerations

- The production path no longer uses a raw forward-Euler membrane increment.
- The adaptation decay is integrated through the same RK4 candidate as voltage,
  so voltage-dependent adaptation current and conductance decay share one
  timestep contract.
- `dt=1.0 ms` remains conservative for the default `tau_m=10.0 ms` and
  `tau_sfa=200.0 ms` settings.
- The implementation rejects non-finite RK4 stages and candidates before
  mutation.
- The safety envelope prevents runaway voltage or adaptation conductance from
  silently entering persistent state.

---

## Polyglot Surfaces

The maintained mirrors implement the same equations and spike/reset contract:

| Surface | File | Runtime contract |
|---------|------|------------------|
| Python | `src/sc_neurocore/neurons/models/sfa.py` | Stateful reference, raises on invalid input |
| Go | `src/sc_neurocore/accel/go/services/sfa.go` | Stateful service, returns `-1` on invalid input |
| Julia | `src/sc_neurocore/accel/julia/neurons/sfa.jl` | Stateful mirror, returns `-1` on invalid input |
| Mojo | `src/sc_neurocore/accel/mojo/kernels/sfa.mojo` | Stateless candidate helpers and spike flag |
| Rust safety | `src/sc_neurocore/accel/rust/safety/sfa.rs` | Stateful safety mirror, returns `-1` on invalid input |

The shared derivative contract is:

$$
\dot V = \frac{-(V - V_{\text{rest}}) - g_{\text{sfa}}(V - E_K) + R I}{\tau_m}
$$

$$
\dot g_{\text{sfa}} = -g_{\text{sfa}}/\tau_{\text{sfa}}
$$

---

## Local Measured Performance

Measured on `aaarthuus` on 2026-06-18 with
`benchmarks/results/local_python_2026-06-18_sfa_rk4.json`. This is a local,
non-isolated regression artifact and is not a production speed claim.

| Backend | Median ns/step | Min ns/step | Max ns/step | Spikes |
|---------|---------------:|------------:|------------:|-------:|
| Python | 3311.596240 | 3228.061555 | 3684.090205 | 2412 |
| Rust safety | 36.169245 | 36.090090 | 36.760775 | 2412 |
| Go service | 42.130000 | 41.480000 | 43.600000 | 2412 |
| Julia kernel | 39.652655 | 39.126480 | 41.637055 | 2412 |
| Mojo kernel | 35.560530 | 35.380395 | 35.795900 | 2412 |

All measured mirrors emitted exactly 2,412 spikes over 200,000 steps at
`current=50.0`, giving zero-tolerance spike parity across the maintained
polyglot surfaces.

---

## Verification

| Category | Tests | What is verified |
|----------|------:|-----------------|
| Isolation | 5 | defaults, binary output, state evolution, finite long run, reset |
| Adaptation | 5 | ISI lengthening, spike increment, RK4 candidate decay, adaptation opposition, accumulation |
| f-I curve | 4 | subthreshold silence, suprathreshold firing, rate increase, zero-current silence |
| Parameters | 6 | adaptation timescale, increment strength, no-adaptation regularity, dt stability |
| Validation | 15 | finite voltages, non-negative adaptation, positive scales, finite current before mutation |
| Determinism | 1 | deterministic 300-step trace |
| Network | 2 | population construction and Python network spiking |
| Analysis | 2 | spike-count consistency |
| Native mirrors | 2 | Go and Rust RK4 candidate/native fail-closed tests |
| **Python total** | **67** | **Passed on 2026-06-18** |

Focused verification commands used for this hardening slice:

```bash
ruff check src/sc_neurocore/neurons/models/sfa.py tests/test_model_sfa.py benchmarks/bench_model_sfa.py
mypy --strict src/sc_neurocore/neurons/models/sfa.py benchmarks/bench_model_sfa.py
pytest tests/test_model_sfa.py -q
go test src/sc_neurocore/accel/go/services/sfa.go src/sc_neurocore/accel/go/services/sfa_test.go
rustc --test src/sc_neurocore/accel/rust/safety/sfa.rs -o /tmp/sfa_safety_test && /tmp/sfa_safety_test
python tools/benchmark_evidence_gate.py --manifest /tmp/sfa_gate.json --output /tmp/sfa_gate_report.json
```

The Mojo and Julia mirrors were also executed with one-step smoke checks, and
the full five-backend benchmark regenerated
`benchmarks/results/local_python_2026-06-18_sfa_rk4.json`.

---

## Usage

### Basic step loop

```python
from sc_neurocore.neurons.models.sfa import SFANeuron

neuron = SFANeuron()
spikes = 0
for _ in range(10_000):
    spikes += neuron.step(50.0)

print(spikes, neuron.v, neuron.g_sfa)
```

### Reset

```python
neuron.reset()
assert neuron.v == neuron.v_rest
assert neuron.g_sfa == 0.0
```

### Removing adaptation

```python
regular = SFANeuron(delta_g=0.0)
```

With `delta_g=0.0`, the adaptation conductance still decays if it is nonzero,
but spikes no longer add new adaptation.

---

## Technical Reference

| Method | Signature | Returns | Description |
|--------|-----------|---------|-------------|
| `step` | `step(current) -> int` | `0` or `1` | Advance one RK4 timestep and return a spike flag |
| `reset` | `reset() -> None` | - | Restore `v = v_rest` and `g_sfa = 0.0` |

The current benchmark evidence is
`benchmarks/results/local_python_2026-06-18_sfa_rk4.json`. Use that artifact
for regression comparison; do not promote the loaded-workstation timings to
production throughput claims.
