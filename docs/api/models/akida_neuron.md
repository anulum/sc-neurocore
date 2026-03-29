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

## Test Coverage

See `tests/test_model_akida_neuron.py` (12 tests):
isolation, rank-order modulation, first-to-spike property,
network integration, analysis toolkit.
