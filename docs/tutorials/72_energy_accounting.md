# Tutorial 72: Neural Energy Accounting

Per-spike energy cost mapped to real hardware.

```python
from sc_neurocore.energy_accounting import EnergyAccountant

acc = EnergyAccountant("loihi2")
report = acc.account(
    layer_names=["hidden", "output"],
    layer_sizes=[(784, 256), (256, 10)],
    spike_counts=[5000, 200],
    n_timesteps=100,
)
print(report.summary())
print(f"Dominant layer: {report.dominant_layer}")
print(f"Energy per spike: {report.energy_per_spike_pj:.1f} pJ")
```
