# Tutorial 72: Neural Energy Accounting

Map every spike in your SNN to a real energy cost on real hardware.
Unlike Tutorial 41 (pre-silicon estimation based on architecture), this
tutorial accounts for *actual spike activity* — dead neurons cost zero,
busy neurons dominate the energy budget.

## Why Per-Spike Accounting

Architecture-level energy estimation says "this network uses 5 mW."
Per-spike accounting says "layer 1 accounts for 73% of energy because
neuron 42 fires 10× more than average." This enables targeted
optimisation: silence the expensive neurons, not the whole network.

## Quick Start

```python
from sc_neurocore.energy_accounting import EnergyAccountant
import numpy as np

acc = EnergyAccountant("loihi2")

report = acc.account(
    layer_names=["hidden", "output"],
    layer_sizes=[(784, 256), (256, 10)],
    spike_counts=[5000, 200],
    n_timesteps=100,
)

print(report.summary())
# Energy Accounting (loihi2):
#   hidden: 5000 spikes × 2.1 pJ/spike = 10.5 nJ (98.1%)
#   output:  200 spikes × 1.0 pJ/spike =  0.2 nJ (1.9%)
#   Total: 10.7 nJ per inference
#   Dominant layer: hidden (98.1%)
```

## Per-Spike Energy on Different Chips

Energy per spike varies dramatically across hardware:

```python
for chip in ["loihi2", "spinnaker2", "akida", "brainscales2", "fpga_ice40"]:
    acc = EnergyAccountant(chip)
    report = acc.account(["layer"], [(128, 64)], [1000], 100)
    print(f"{chip:15s}: {report.energy_per_spike_pj:>6.1f} pJ/spike, "
          f"total={report.total_energy_nj:>8.1f} nJ")

# loihi2         :    2.1 pJ/spike, total=     2.1 nJ
# spinnaker2     :   10.0 pJ/spike, total=    10.0 nJ
# akida          :    3.5 pJ/spike, total=     3.5 nJ
# brainscales2   :    0.3 pJ/spike, total=     0.3 nJ (analog!)
# fpga_ice40     :   15.0 pJ/spike, total=    15.0 nJ
```

BrainScaleS-2 is analog — spikes cost almost nothing. Loihi 2 is
digital but highly optimised. FPGA is more expensive per spike but
offers custom neuron models.

## Energy Breakdown by Operation

Each spike triggers several operations, each with a cost:

| Operation | Loihi 2 | SpiNNaker2 | FPGA (45nm) |
|-----------|---------|-----------|-------------|
| Membrane update | 0.5 pJ | 2.0 pJ | 3.0 pJ |
| Spike generation | 0.1 pJ | 0.5 pJ | 1.0 pJ |
| Synapse lookup | 0.8 pJ | 5.0 pJ | 8.0 pJ |
| Weight multiply | 0.5 pJ | 2.0 pJ | 2.0 pJ |
| AER routing | 0.2 pJ | 0.5 pJ | 1.0 pJ |
| **Total per spike** | **2.1 pJ** | **10.0 pJ** | **15.0 pJ** |

These are literature values, not measurements from our hardware.

## Optimisation Targets

The accounting report tells you where to focus:

```python
# If hidden layer is 98% of energy:
# → Prune hidden layer (Tutorial 44)
# → Increase threshold (reduce spike rate)
# → Use homeostasis (Tutorial 68) to regulate activity

# If dominant neuron is responsible for >10% of total:
# → That neuron may be over-connected or have low threshold
# → Check with Architecture Doctor (Tutorial 56)
```

## Integration with Studio

The Training Monitor shows per-layer spike rates. Multiply by the
chip's energy-per-spike to get real-time energy estimates during
training.

## References

- Horowitz (2014). "Computing's Energy Problem." ISSCC Keynote.
- Davies et al. (2021). "Advancing Neuromorphic Computing with Loihi 2."
  Intel White Paper.
