# Spintronic Mapper

Maps SC networks onto magnetic tunnel junction (MTJ) arrays. Models
spin-torque switching (STT/SOT), TMR calculation, and MuMax3 co-simulation.
`SpintronicDeviceConfig` carries explicit write-path and parallel-state
resistance parameters, so switching energy and cell resistance are derived from
the selected device contract rather than a fixed generic MTJ constant.

## Quick Start

```python
from sc_neurocore.spintronic.spintronic_mapper import (
    SpintronicMapper, MagneticDomainSim, SpinTorqueModel,
)
```

::: sc_neurocore.spintronic.spintronic_mapper
