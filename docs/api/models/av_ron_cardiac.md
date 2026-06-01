# AvRonCardiacNeuron

**Module:** `sc_neurocore.neurons.models.av_ron_cardiac`
**Rust:** `sc_neurocore_engine::neurons::biophysical::AvRonCardiacNeuron`
**Reference:** Av-Ron, Parnas & Segel, *Biological Cybernetics* 69(2), 1993.
**Family:** Conductance-based cardiac ganglion Type III burster.
**State variables:** `v`, `h`, `n`, `s`.

## Contract implemented

`AvRonCardiacNeuron` implements the four-state cardiac ganglion burster with:

1. instantaneous sodium activation `m_inf(v)`,
2. voltage-dependent `h`, `n`, and `s` gate relaxation,
3. candidate-first RK4 integration over `(v, h, n, s)`,
4. fail-closed rejection of invalid current, parameters, gates, derivatives, or candidate states.

The model no longer mutates gates before voltage acceptance. Each step computes a full RK4 candidate from the old state and commits only when all state values are finite and gates remain inside `[0, 1]`.

## Equations

```text
dV/dt = -I_Na - I_K - I_s - I_L + I_ext
I_Na = g_Na * m_inf(v)^3 * h * (v - E_Na)
I_K  = g_K  * n^4        * (v - E_K)
I_s  = g_s  * s          * (v - E_s)
I_L  = g_L               * (v - E_L)
```

Boltzmann steady states:

```text
m_inf = 1 / (1 + exp(-(v + 40) / 7))
h_inf = 1 / (1 + exp( (v + 45) / 5))
n_inf = 1 / (1 + exp(-(v + 40) / 15))
s_inf = 1 / (1 + exp( (v + 35) / 3))
```

Gate time constants:

```text
tau_h = 1 + 12   / (1 + exp((v + 50) / 8))
tau_n = 1 + 8    / (1 + exp((v + 35) / 8))
tau_s = 200 + 1000 / (1 + exp((v + 30) / 5))
```

Gate ODEs:

```text
dh/dt = (h_inf - h) / tau_h
dn/dt = (n_inf - n) / tau_n
ds/dt = (s_inf - s) / tau_s
```

Spike output is an upward threshold crossing:

```text
spike = v_next >= v_threshold and v_old < v_threshold
```

## Defaults

| Parameter | Default | Role |
| --- | ---: | --- |
| `v` | -60.0 | membrane voltage |
| `h` | 0.6 | sodium inactivation gate |
| `n` | 0.3 | potassium activation gate |
| `s` | 0.5 | slow inactivation gate |
| `g_na` | 80.0 | sodium conductance |
| `g_k` | 40.0 | potassium conductance |
| `g_s` | 20.0 | slow current conductance |
| `g_l` | 0.1 | leak conductance |
| `e_na` | 40.0 | sodium reversal |
| `e_k` | -80.0 | potassium reversal |
| `e_s` | -25.0 | slow-current reversal |
| `e_l` | -60.0 | leak reversal |
| `dt` | 0.02 | timestep |
| `v_threshold` | -20.0 | spike detection threshold |

## Backend surfaces

| Surface | File | Contract |
| --- | --- | --- |
| Python reference | `src/sc_neurocore/neurons/models/av_ron_cardiac.py` | candidate-first RK4 |
| Rust engine | `engine/src/neurons/biophysical.rs` | candidate-first RK4 |
| PyO3 | `sc_neurocore_engine.AvRonCardiacNeuron` | Rust engine exposure |
| Go service | `src/sc_neurocore/accel/go/services/av_ron_cardiac.go` | candidate-first RK4 |
| Julia kernel | `src/sc_neurocore/accel/julia/neurons/av_ron_cardiac.jl` | candidate-first RK4 |
| Rust safety mirror | `src/sc_neurocore/accel/rust/safety/av_ron_cardiac.rs` | fail-closed RK4 mirror |
| Mojo contract | `src/sc_neurocore/accel/mojo/kernels/av_ron_cardiac.mojo` | kernel promotion contract |

## Behavioural tests

The dedicated model test `tests/test_model_av_ron_cardiac.py` checks:

- RK4 reference point and separation from the former Euler candidate,
- upward-only threshold crossing semantics,
- gate boundedness during nominal burst drive,
- invalid current preservation,
- invalid parameter preservation,
- corrupted gate preservation,
- nonfinite candidate preservation,
- reset semantics,
- Boltzmann activation/inactivation monotonicity.

## Benchmark

Benchmark script: `benchmarks/bench_av_ron_cardiac.py`

Current result artefact: `benchmarks/results/bench_av_ron_cardiac.json`

Run:

```bash
PYTHONPATH=src .venv/bin/python benchmarks/bench_av_ron_cardiac.py
```

The benchmark records Python reference throughput, Rust PyO3 throughput when available, and Python/Rust parity over `v`, `h`, `n`, `s`, and spike count.

## Minimal use

```python
from sc_neurocore.neurons.models.av_ron_cardiac import AvRonCardiacNeuron

neuron = AvRonCardiacNeuron()
spike = neuron.step(2.0)
state = {"v": neuron.v, "h": neuron.h, "n": neuron.n, "s": neuron.s}
```
