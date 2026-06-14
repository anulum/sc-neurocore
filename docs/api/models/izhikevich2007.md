<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996-2026 Miroslav Sotek. All rights reserved. -->
<!-- © Code 2020-2026 Miroslav Sotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SC-NeuroCore model documentation -->

# Izhikevich2007Neuron

**Module:** `sc_neurocore.neurons.models.izhikevich2007`
**Reference:** Izhikevich 2007, *Dynamical Systems in Neuroscience*
**Family:** Biophysical quadratic integrate-and-fire
**State variables:** `v` (mV), `u` (pA)

## Equations

The model follows the NeuroML 2 `izhikevich2007Cell` parameterisation:

$$C \frac{dv}{dt} = k(v - v_r)(v - v_t) - u + I$$

$$\frac{du}{dt} = a(b(v - v_r) - u)$$

Spike and reset:

$$v \geq v_{peak} \Rightarrow v \leftarrow c,\quad u \leftarrow u + d$$

Units use the NeuroML base convention used by the importer: `C` in pF,
`k` in nS/mV, voltages in mV, `a` in 1/ms, `b` in nS, and current terms in pA.

## Parameters

| Parameter | Default | Unit | Description |
|-----------|--------:|------|-------------|
| `C` | 100.0 | pF | Membrane capacitance |
| `k` | 0.7 | nS/mV | Quadratic gain |
| `vr` | -60.0 | mV | Resting membrane voltage |
| `vt` | -40.0 | mV | Instantaneous threshold voltage |
| `vpeak` | 35.0 | mV | Spike detection voltage |
| `a` | 0.03 | 1/ms | Recovery time-scale |
| `b` | -2.0 | nS | Recovery-voltage coupling |
| `c` | -50.0 | mV | Reset voltage |
| `d` | 100.0 | pA | Recovery reset increment |
| `v0` | `vr` | mV | Initial membrane voltage |
| `dt` | 0.1 | ms | Integration step |
| `integrator` | `rk4` | - | `rk4` or `euler` |

## Polyglot acceleration

`step` runs one RK4 update, but `simulate(n_steps, current, backend=...)` is a
sequential recurrence (each step depends on the previous) that does not
vectorise — a compiled inner loop genuinely beats Python. The kernel carries a
full polyglot chain over the **RK4** integrator (the production default;
`simulate` raises for the `euler` integrator, which stays on `step()`):

```python
from sc_neurocore.neurons.models.izhikevich2007 import Izhikevich2007Neuron

neuron = Izhikevich2007Neuron()
trace, spikes = neuron.simulate(2_000_000, current=300.0)          # auto -> Rust
trace, spikes = neuron.simulate(2_000_000, 300.0, backend="go")   # force a backend
```

`backend` accepts `"auto" | "rust" | "julia" | "go" | "mojo" | "python"`. `auto`
prefers Rust (it ships in the `sc_neurocore_engine` wheel). `trace[t]` is `v`
after step `t` (reset to `c` on a spiking step); `spikes` counts the steps that
reached `vpeak`. The Rust backend is a dedicated `Izhikevich2007Rk4` integrator
in the engine — distinct from the dimensionless 2003 `IzhikevichRk4`.

The NeuroML right-hand side `k (v-vr)(v-vt)/C` is **exact arithmetic** — products,
a sum and a division, no transcendental functions and no `pow` — so **Rust,
Julia and Go reproduce the NumPy trace bit-for-bit**, verified over a 60,000-step
tonic-spiking run. Mojo's release build fuses some RK4 multiply-adds into FMAs;
the hard `v >= vpeak -> v = c` reset re-anchors the trajectory on every spike, so
the gap does not amplify and the spike counts always match. The residual gap
depends on the firing rate: ~5e-12 at strong drive (frequent resets), up to
~4e-8 in a sparse-firing regime (long inter-spike intervals). Mojo is validated
on that band plus exact spike counts, not whole-trace equality.

### Measured backends

Reproduce with `python benchmarks/bench_izhikevich2007_simulate.py --json
benchmarks/results/bench_izhikevich2007_simulate.json`. Workload: 2,000,000 RK4
steps, default parameters, current = 300 pA (tonic spiking), median of 5 repeats.
**Non-isolated** (loaded workstation, Python 3.12 / NumPy 2.3) —
functional/regression evidence, not isolated-core release numbers.

| backend | median (ms) | speedup vs NumPy | parity Δ vs NumPy |
|---|---:|---:|---:|
| python (NumPy) | 1773.10 | 1.00× | 0 |
| mojo | 73.12 | 24.25× | 8.78e-12 (FMA band at I=300) |
| rust | 88.98 | 19.93× | 0 (bit-exact) |
| go | 92.86 | 19.09× | 0 (bit-exact) |
| julia | 93.48 | 18.97× | 0 (bit-exact) |

Mojo is fastest in raw throughput, but it is not bit-exact, so `auto` selects
Rust — the fastest backend that is both bit-exact and ships in the wheel. The
four derivative stages per step keep the absolute per-step cost high, so the
speedups settle around 19–24×.

## NeuroML Import

`<izhikevichCell>` remains mapped to `SCIzhikevichNeuron`, the dimensionless
2003 formulation. `<izhikevich2007Cell>` maps to `Izhikevich2007Neuron`, so
the biophysical 2007 parameters are preserved instead of being converted into
the 2003 parameter set.

## API

::: sc_neurocore.neurons.models.izhikevich2007.Izhikevich2007Neuron
