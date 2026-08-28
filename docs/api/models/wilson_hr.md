<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
# WilsonHRNeuron

**Module:** `sc_neurocore.neurons.models.wilson_hr`
**Rust engine:** `sc_neurocore_engine::neurons::simple_spiking::WilsonHRNeuron`
**Reference:** Wilson, H. R. (1999), *Simplified Dynamics of Human and
Mammalian Neocortical Neurons*, DOI `10.1006/jtbi.1999.1002`.

`WilsonHRNeuron` implements the continuous two-state polynomial cortical model:

```text
C*dV/dt = -(17.81 + 47.71*V + 32.63*V^2)*(V - 0.55)
          - 26*R*(V + 0.92) + I
dR/dt   = (-R + 1.35*V + 1.03) / tau_R
```

The source parameters and defaults are `C=0.8` and `tau_R=1.9 ms`. At zero
input the stated initial rest voltage `V=-0.70` gives `R=0.085`. Wilson's spikes are continuous
limit-cycle trajectories: neither state is reset. The repository reports an
event only when the sampled voltage crosses `v_peak=0` upward. That event level,
fixed `dt=0.05`, and classical RK4 are numerical observation conventions, not
additional source equations.

The former unit-capacitance, `v>=0.4`, hard-reset recurrence remains available
under the explicit project identity
[`SCResettingWilsonHRNeuron`](sc_resetting_wilson_hr.md).

## Runtime contract

Python, production Rust, Rust safety, Julia, Go, and Mojo advance all RK4 stages
in the same order. Invalid state, current, scale, intermediate stage, candidate,
or returned batch fails explicitly. A rejected batch leaves the public Python
state unchanged. Rust, Julia, and Go are enrolled for exact binary64 parity;
Mojo carries a `1e-9` complete-trace bound and an exact event-count gate.

```python
from sc_neurocore.neurons.models.wilson_hr import WilsonHRNeuron

neuron = WilsonHRNeuron()
trace, spikes = neuron.simulate(5_000, current=0.1, backend="auto")
```

`trace` contains the continuous post-step voltage; spike samples are not replaced
by a reset sentinel.

## Evidence

- The paired `wilson_hr.{toml,json}` schemas reproduce source capacitance,
  simultaneous RK4, continuous state, and upward-crossing observation.
- `wilson_hr_driven_spiking_doi` independently re-derives 5,000 steps at
  `I=0.1`: 46 crossings, first at step 15, plus complete voltage/recovery feature
  statistics.
- Q16.16 co-simulation compares hand model, schema runner, and emitted RTL at
  silent and periodic operating points.
- The generated Q8.8 catalogue RTL and port-only harness carry a bounded
  SymbiYosys safety job. This is not claimed as universal real-number
  equivalence, timing closure, PPA, or physical-silicon evidence.
- `benchmarks/results/bench_wilson_hr_simulate.json` records an executed,
  source-hash-bound five-runtime packet. Its timings are non-isolated local
  regression evidence and are not production speed claims.
- The public workflow tests exercise real `Population`, `Projection`, `Network`,
  `SpikeMonitor`, and spike-statistics surfaces.

The model is a reduced polynomial dynamical system, not a detailed ionic-gate
replacement for Hodgkin-Huxley. Its current uses Wilson's scaled source units;
values inherited from hard-reset point-neuron examples are not interchangeable.
