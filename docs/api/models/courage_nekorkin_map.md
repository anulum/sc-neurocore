# CourageNekorkinMapNeuron

**Module:** `sc_neurocore.neurons.models.courage_nekorkin_map`
**Reference:** Courbage M., Nekorkin V.I. & Vdovin L.V. (2007), *Chaotic oscillations
in a map-based model of neural activity*, **Chaos 17:043109** ([arXiv:0712.2097](https://arxiv.org/abs/0712.2097)).
**Family:** Map-based (discrete-time, discontinuous piecewise-linear Lorenz-type)
**State variables:** `x` (fast, membrane-like), `y` (slow, recovery-like)

The Courbage-Nekorkin-Vdovin (CNV) map is a two-dimensional discrete-time neuron
built from a discrete FitzHugh-Nagumo system combined with a one-dimensional
Lorenz-type map, with a Heaviside discontinuity at `x = d` that sets the
excitation threshold. With a single parameter region it reproduces a rich gallery
of biological firing modes — chaotic spiking-bursting, subthreshold oscillations,
tonic spiking, phasic spikes and bursts — at a fraction of the cost of an
ODE-based conductance model. That makes it a natural building block for
large-scale spiking-network simulation, where per-neuron cost dominates.

---

## Equations

The map `f : (x, y) → (x̄, ȳ)` is (Courbage et al. 2007, eqs. 3–5):

$$\bar{x} = x + F(x) - y - \beta\,H(x - d) + I$$

$$\bar{y} = y + \varepsilon\,(x - J)$$

with the piecewise-linear field and the Heaviside step

$$F(x) = \begin{cases} -m_0\,x & x \le J_{\min} \\ m_1\,(x - a) & J_{\min} < x < J_{\max} \\ -m_0\,(x - 1) & x \ge J_{\max} \end{cases}
\qquad
H(z) = \begin{cases} 1 & z \ge 0 \\ 0 & z < 0 \end{cases}$$

The continuity breakpoints of `F` are fixed by the parameters:

$$J_{\min} = \frac{a\,m_1}{m_0 + m_1}, \qquad J_{\max} = \frac{m_0 + a\,m_1}{m_0 + m_1}$$

`x` is the membrane potential, `y` the recovery variable (outward ionic currents),
`I` (`current`) an injected external stimulus, `J` a constant external drive, and
`d` the discontinuity line that determines the excitation threshold of the
spiking-bursting oscillations. `I = 0` reproduces the published autonomous map.

The model carries **no clip**: it stays bounded by its own invariant attractor
inside the parameter region below, not by an artificial saturation.

### Parameter region (eq. 6)

The published analysis holds on

$$0 < J < d, \qquad J_{\min} < d < J_{\max}, \qquad m_0 < 1.$$

The chaotic-attractor region `B⁺` additionally requires
`β₀ < β < β₁` with `β₀ = F(J_max) − F(J_min)` and
`β₁ = min{q(J_max − d), q(d − J_min)}`, where `q = 1 + m₁`.

### Spike detection

A spike is the upward crossing of `x_threshold` (defaulting to the discontinuity
`d`): `x_prev < x_threshold` **and** `x̄ ≥ x_threshold`. This counts each
threshold excursion once.

---

## Parameters

| Parameter | Default | Description |
|-----------|--------:|-------------|
| `x` | 0.0 | Fast variable (membrane potential) |
| `y` | 0.0 | Slow variable (recovery / outward currents) |
| `m0` | 0.0864 | Slope of the outer (resting / spike-peak) branches of `F` |
| `m1` | 0.65 | Slope of the middle (regenerative) branch of `F`; `q = 1 + m1` |
| `a` | 0.2 | Offset of the middle branch; sets `J_min`, `J_max` |
| `d` | 0.235 | Heaviside discontinuity — the excitation threshold |
| `j` | 0.2 | Constant external drive `J` |
| `beta` | 0.085 | Reset jump applied through `H(x − d)` |
| `eps` | 0.02 | Recovery time-scale `ε` (slow variable) |
| `x_threshold` | 0.235 | Spike-detection threshold (defaults to `d`) |

The defaults are the published values: `m0 = 0.0864`, `m1 = 0.65`, `a = 0.2` are
the figure-1 parameters; `d`, `J`, `β`, `ε` sit inside the `B⁺` invariant-region
triangle and the chaotic-spiking regime (Table, p20). With these,
`J_min ≈ 0.17653`, `J_max ≈ 0.29386`, so `J_min < d < J_max` and `0 < J < d` hold.

---

## Firing regimes

The same map reproduces the published gallery of neural activity by moving inside
the parameter region (Courbage et al. 2007, Table p20):

| Regime | Condition (in addition to the region constraints) |
|--------|----------------------------------------------------|
| Chaotic spiking (default) | `J > J_min`, `ε ≪ 1`, `F(J_min) > F(d) − β`, `F(J_max) > F(d)` |
| Chaotic spiking-bursting | invariant-region inequalities (eq. 26) |
| Subthreshold oscillations | `J > J_min`, `ε < m₀`, `m₀ > m₁²/4`, `ε > max(m₀²/4, m₁²/4)` |
| Tonic spiking | `J > J_min`, `ε ≪ 1`, `F(J_min) < F(d) − β` |
| Phasic spikes / bursts | `J < J_min`, `ε ≪ 1` (excitable rest state, response to a pulse) |

The default parameters produce a bounded chaotic attractor. Verified numerically
on a 20 000-step autonomous run from `(x, y) = (0, 0)`: ~3 700 spikes, trajectory
bounded in `x ∈ [0, 0.28]` (no clip-pegging), inter-spike intervals spanning
in-burst (≤ 3) and inter-burst (≥ 8) gaps, and sensitive dependence on initial
conditions (a `10⁻⁹` perturbation grows to `10⁻²` within 2 000 steps).

---

## Polyglot acceleration

A single `step` is trivial, but an N-step run is a sequential recurrence that does
not vectorise, so a compiled inner loop genuinely beats Python.
`simulate(n_steps, current, backend="auto")` dispatches across the polyglot chain
and returns `(trace, spikes)`:

```python
from sc_neurocore.neurons.models.courage_nekorkin_map import CourageNekorkinMapNeuron

neuron = CourageNekorkinMapNeuron()
trace, spikes = neuron.simulate(20_000, current=0.0)   # auto → Rust
```

The map is exact floating-point arithmetic (additions, multiplications, one
division for the breakpoints, and a piecewise/Heaviside branch — no transcendental
functions), so **Rust, Julia and Go reproduce the NumPy reference bit-for-bit**
even though the dynamics are chaotic. Mojo's release build contracts a
multiply-add into a fused multiply-add: the trajectory is bit-exact for the first
~100 steps, after which a single ULP appears and the chaotic map amplifies it into
a whole-trace gap — so the Mojo backend is validated per-step and on spike counts,
and `auto` selects Rust (the fastest bit-exact backend, shipped in the wheel).

### Measured throughput

2 000 000 steps, default chaotic regime, median of 5 repeats. Non-isolated loaded
workstation (Intel i5-11600K) per `BROADCAST_2026-06-04_benchmark_core_isolation`
— functional/regression evidence, not an isolated-core figure. Reproduce with
`python benchmarks/bench_courage_nekorkin_map_simulate.py`.

| Backend | Median (ms) | Speed-up vs Python | Whole-trace parity |
|---------|------------:|-------------------:|--------------------|
| python  | 618.49 | 1.0× | reference |
| rust (`auto`) | 15.97 | 38.7× | bit-exact (0) |
| julia   | 17.49 | 35.4× | bit-exact (0) |
| go      | 15.88 | 39.0× | bit-exact (0) |
| mojo    | 15.42 | 40.1× | 1.97×10⁻¹ (chaotic FMA, by design) |

Artefact: `benchmarks/results/bench_courage_nekorkin_map_simulate.json`.

---

## Usage

### Single neuron, threshold crossings

```python
from sc_neurocore.neurons.models.courage_nekorkin_map import CourageNekorkinMapNeuron

neuron = CourageNekorkinMapNeuron()
spikes = [t for t in range(10_000) if neuron.step() == 1]
print(f"{len(spikes)} spikes")
```

### Switching firing regime

```python
# Excitable / phasic regime: drive J below J_min (~0.1765)
excitable = CourageNekorkinMapNeuron(j=0.12)
_, n_excitable = excitable.simulate(20_000)

# Default chaotic spiking-bursting
_, n_default = CourageNekorkinMapNeuron().simulate(20_000)
assert n_excitable < n_default
```

### Coupled network

```python
from sc_neurocore.network.population import Population
from sc_neurocore.network.projection import Projection
from sc_neurocore.network.network import Network
from sc_neurocore.network.monitor import SpikeMonitor
from sc_neurocore.network.stimulus import PoissonInput
from sc_neurocore.neurons.models.courage_nekorkin_map import CourageNekorkinMapNeuron

pop = Population(CourageNekorkinMapNeuron, n=50, label="excitable")
drive = PoissonInput(n=50, rate_hz=500.0, weight=0.3, dt=0.001, seed=42)
mon = SpikeMonitor(pop)
net = Network(pop, drive, mon)
net.run(duration=2.0, dt=0.001, backend="python")
print(mon.count)
```

---

## Applications

- **Large-scale spiking-network simulation.** Map-based neurons replace ODE
  integration with a few arithmetic operations per step, so cortical-column and
  whole-network models run at a fraction of the conductance-model cost while
  retaining spiking, bursting and subthreshold dynamics.
- **Chaotic spiking-bursting studies.** The bounded chaotic attractor models the
  irregular spiking-bursting seen in hippocampal pyramidal and thalamic cells.
- **Subthreshold-oscillation regimes** relevant to inferior-olive neurons and
  olivo-cerebellar timing.
- **Synchronisation and wave propagation** in coupled excitable lattices.
- **Neuromorphic / FPGA targets.** With no transcendental functions and a single
  division for the breakpoints, the per-step kernel maps directly onto
  fixed-point hardware.

---

## Implementation notes

| Surface | File |
|---------|------|
| Python reference + backend dispatch | `src/sc_neurocore/neurons/models/courage_nekorkin_map.py` |
| Rust engine struct + `simulate` (PyO3) | `engine/src/neurons/maps.rs`, `engine/src/lib.rs` |
| Rust fail-closed safety mirror | `src/sc_neurocore/accel/rust/safety/courage_nekorkin_map.rs` |
| Julia backend | `src/sc_neurocore/accel/julia/neurons/courage_nekorkin_map.jl` |
| Go backend (c-shared) | `src/sc_neurocore/accel/go/neurons/courage_nekorkin_map/courage_nekorkin_map.go` |
| Mojo backend (FFI) | `src/sc_neurocore/accel/mojo/neurons/courage_nekorkin_map.mojo` |
| Tests | `tests/test_model_courage_nekorkin_map.py`, `tests/test_courage_nekorkin_map_backends.py` |
| Benchmark | `benchmarks/bench_courage_nekorkin_map_simulate.py` |

All five backends share an identical operation order; the bit-exact backends are
verified against the NumPy reference across several currents and parameter regimes
in `tests/test_courage_nekorkin_map_backends.py`.

---

## Citations

1. Courbage M., Nekorkin V.I., Vdovin L.V. (2007). Chaotic oscillations in a
   map-based model of neural activity. *Chaos* 17:043109.
   DOI: [10.1063/1.2795435](https://doi.org/10.1063/1.2795435);
   [arXiv:0712.2097](https://arxiv.org/abs/0712.2097).
2. Courbage M., Nekorkin V.I. (2010). Map based models in neurodynamics.
   *Int. J. Bifurcat. Chaos* 20(6):1631–1651.
   DOI: [10.1142/S0218127410026733](https://doi.org/10.1142/S0218127410026733).
3. Rulkov N.F. (2002). Modeling of spiking-bursting neural behavior using
   two-dimensional map. *Phys. Rev. E* 65:041922.
   DOI: [10.1103/PhysRevE.65.041922](https://doi.org/10.1103/PhysRevE.65.041922).
