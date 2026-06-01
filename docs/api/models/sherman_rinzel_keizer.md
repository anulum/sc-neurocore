# ShermanRinzelKeizerNeuron

**Module:** `sc_neurocore.neurons.models.sherman_rinzel_keizer`
**Reference:** Sherman, Rinzel & Keizer 1988
**Family:** reduced pancreatic beta-cell conductance model
**State:** `v`, `n`, `s`

## Current implementation contract

`ShermanRinzelKeizerNeuron` advances the published reduced beta-cell burster
ODE with candidate-first fourth-order Runge-Kutta integration. The runtime
rejects non-finite currents, non-finite state, gates outside `[0, 1]`, invalid
conductances, invalid time constants, and RK4 candidates that leave the finite
voltage/gate envelope before mutating state.

The continuous state is not reset on a threshold crossing. `step(current)`
returns `1` only when the RK4 candidate crosses `v_threshold` from below;
otherwise it returns `0`.

## Equations

\[
\frac{dV}{dt} = -I_{Ca} - I_K - I_s + I_{ext}
\]

\[
I_{Ca} = g_{Ca}m_\infty(V)(V-E_{Ca}),\quad
I_K = g_K n(V-E_K),\quad
I_s = g_s s(V-E_K)
\]

\[
m_\infty(V)=\frac{1}{1+e^{-(V+20)/12}},\quad
n_\infty(V)=\frac{1}{1+e^{-(V+16)/5}},\quad
s_\infty(V)=\frac{1}{1+e^{-(V+35)/10}}
\]

\[
\frac{dn}{dt}=\frac{n_\infty(V)-n}{9.09},\quad
\frac{ds}{dt}=\frac{s_\infty(V)-s}{\tau_s}
\]

The implementation evaluates the sigmoid arguments with bounded exponentials
for numerical safety while preserving the same limiting values.

## Parameters

| Parameter | Default | Meaning |
| --- | ---: | --- |
| `v` | `-50.0` | membrane voltage |
| `n` | `0.1` | fast potassium activation gate |
| `s` | `0.1` | slow potassium gate |
| `g_ca` | `3.6` | calcium conductance |
| `g_k` | `10.0` | potassium conductance |
| `g_s` | `4.0` | slow potassium conductance |
| `e_ca` | `25.0` | calcium reversal potential |
| `e_k` | `-75.0` | potassium reversal potential |
| `tau_s` | `5000.0` | slow-gate time constant |
| `dt` | `0.5` | integration timestep |
| `v_threshold` | `-20.0` | threshold-crossing detector |

## Backend surfaces

| Surface | Status |
| --- | --- |
| Python reference | candidate-first RK4 with fail-closed validation |
| Go service | same RK4 state contract and invalid-state preservation |
| Julia mirror | same RK4 state contract and validation helper |
| Rust safety | same RK4 state contract and module tests |
| Mojo kernel file | contract notes for promoting the scalar RK4 path |

## Behavioural verification

The module-specific tests check:

| Contract | Evidence |
| --- | --- |
| RK4 reference point | first `current=5.0` step matches independent RK4 calculation |
| Former Euler separation | voltage and fast gate differ materially from the old raw increment |
| Bounded gates | `n` and `s` stay inside `[0, 1]` under sustained drive |
| Timescale separation | fast gate moves more than the slow gate over short windows |
| Current signs | calcium current is inward at rest; potassium currents are outward |
| Fail-closed paths | invalid current, parameters, gate state, and candidate preserve state |
| Public wiring | `Population`, `Network`, and `spike_count` use the public model surface |

Go and Rust safety tests use the same subthreshold RK4 reference point:

| Quantity after one step from defaults, `current=5.0` | Value |
| --- | ---: |
| `v` | `-54.24952703064663` |
| `n` | `0.09468731121669713` |
| `s` | `0.10000523900468992` |

## Benchmark

Reproducible command:

```bash
PYTHONPATH=src .venv/bin/python benchmarks/bench_sherman_rinzel_keizer.py
```

Result artifact: `benchmarks/results/bench_sherman_rinzel_keizer.json`.

Measured locally on 2026-06-01 for `80,000` steps at `current=5.0`:

| Backend | Steps/s | Wall seconds | Spikes |
| --- | ---: | ---: | ---: |
| Python | 46331 | 1.726702 | 1738 |
| Rust safety | 6637052 | 0.012054 | 1738 |

The committed JSON also records Python/Rust state parity through maximum
absolute state delta and spike-count delta. Update this table when the
benchmark JSON changes.

## Minimal use

```python
from sc_neurocore.neurons.models.sherman_rinzel_keizer import ShermanRinzelKeizerNeuron

neuron = ShermanRinzelKeizerNeuron()
spike = neuron.step(5.0)
state = neuron.v, neuron.n, neuron.s
```
