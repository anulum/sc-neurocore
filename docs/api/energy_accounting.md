# Energy Accounting

Per-spike, per-synapse, per-layer energy accounting mapped to hardware.

- `EnergyAccountant` — Track energy consumption of every operation (spike generation, synaptic transmission, membrane update) using hardware-calibrated cost models. Reports in picojoules per inference.

Supported hardware targets: 45nm CMOS, 28nm, 7nm, Loihi, SpiNNaker.

```python
from sc_neurocore.energy_accounting import EnergyAccountant

accountant = EnergyAccountant(technology="7nm")
accountant.track(model, inputs)
print(f"Total: {accountant.total_pj:.1f} pJ")
```

See [Tutorial 72: Energy Accounting](../tutorials/72_energy_accounting.md).

::: sc_neurocore.energy_accounting
    options:
      show_root_heading: true
