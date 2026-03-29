# AkidaNeuron

**Module:** `sc_neurocore.neurons.models.akida_neuron`
**Reference:** BrainChip Akida 2021
**Family:** Hardware (event-domain)
**State variables:** `v` (integer membrane potential), `_rank` (event counter), `_spiked` (fired flag)

## Equations

$$V \mathrel{+}= w \cdot \mu^{\text{rank}}$$

Spike when $V \geq \theta$ (fires at most **once** per presentation).
No leak between events. Integer arithmetic.

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `v` | 0 | Membrane potential (integer) |
| `threshold` | 100 | Spike threshold (integer) |
| `modulation` | 0.75 | Rank-order decay factor $\mu$ |

## Behaviour

- **Event-domain:** No clock-driven updates. Membrane accumulates only
  when an input spike arrives with nonzero weight.
- **Rank-order coding:** First events contribute most (weight × μ⁰ = full).
  Later events are attenuated (weight × μⁿ). This implements temporal
  coding where the order of arriving spikes carries information.
- **First-to-spike:** Fires at most once per presentation. The `_spiked`
  flag prevents re-firing. Call `reset()` between presentations.

## Network Usage

```python
from sc_neurocore.neurons.models.akida_neuron import AkidaNeuron
from sc_neurocore.network.population import Population
from sc_neurocore.network.stimulus import PoissonInput
from sc_neurocore.network.network import Network
from sc_neurocore.network.monitor import SpikeMonitor

pop = Population(AkidaNeuron, n=10, label="akida")
drive = PoissonInput(n=10, rate_hz=1000.0, weight=30.0, dt=0.001, seed=42)
mon = SpikeMonitor(pop)
net = Network(pop, drive, mon)
net.run(duration=0.1, dt=0.001)
# Each neuron fires at most once → max 10 spikes
```

**Drive requirements:** Integer weight × modulation^rank must accumulate
to threshold (default 100). Weight=30 at rate=1000 Hz reaches threshold
in ~6 events.

## Infrastructure Pipeline

```
AkidaNeuron
├── step(weight: int) → int {0,1}
├── reset() → v=0, rank=0, spiked=False
├── In Population: 1 instance per neuron, scalar current (cast to int)
│   └── Return value: 0 or 1 (fires at most once per presentation)
├── In Network: compatible with PoissonInput (weight as integer current)
│   ├── PoissonInput (weight=30, rate=1000Hz)
│   ├── SpikeMonitor
│   └── Must call reset() between presentations
├── Analysis: spike_count (binary train, max 1 spike per neuron)
├── SC encoding: NOT applicable (event-domain, not rate-coded)
├── Verilog: NOT directly compilable (integer accumulator, rank counter)
│   Could be implemented as custom HDL with counter + shift register
└── Rust NetworkRunner: NOT supported (non-standard step() signature)
```

## Wiring Plan

```
PoissonInput(weight=30, rate=1000Hz)
    ↓ integer weight per event
Population(AkidaNeuron, n=N)
    ↓ binary spike vector (max N spikes total — first-to-spike)
SpikeMonitor → spike_trains
    ├── Each neuron fires at most ONCE
    ├── Spike latency = number of events to reach threshold
    └── Earlier spike = stronger input match (rank-order coding)
```

## Performance

| Metric | Python (NumPy) | Rust engine |
|--------|---------------|-------------|
| Isolation (single neuron, with reset) | 2.43 Msteps/s | N/A (non-standard interface) |
| Network (100 neurons, 100ms) | 794 Kneuron-steps/s | N/A |
| Max spikes (100 neurons) | 100 (one per neuron) | — |

Measured on AMD EPYC / Python 3.12. Event-domain model — speed depends
on input event rate, not clock cycles.

## Test Coverage

| Category | Tests | What is verified |
|----------|------:|-----------------|
| Isolation | 8 | construction, step binary, spikes under drive, fires-only-once, rank modulation, zero weight, reset, integer state |
| Network | 3 | Population creation, spike production, first-to-spike property (each neuron ≤ 1 spike) |
| Analysis | 1 | spike_count on binary train |
| **Total** | **12** | |

See `tests/test_model_akida_neuron.py`.
