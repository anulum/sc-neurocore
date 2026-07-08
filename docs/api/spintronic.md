# Spintronic Mapper

Maps SC networks onto magnetic tunnel junction (MTJ) arrays. Models
spin-torque switching (STT/SOT), TMR calculation, and MuMax3 co-simulation.
`SpintronicDeviceConfig` carries explicit write-path and parallel-state
resistance parameters, so switching energy and cell resistance are derived from
the selected device contract rather than a fixed generic MTJ constant.

## Quick Start

```python
from sc_neurocore.spintronic import (
    SpintronicMapper, SpintronicTech, SpintronicDeviceConfig,
)
```

The package facade exports the documented mapper surface, including
`SpintronicMapper`, device/material models, MuMax3 script/output helpers,
racetrack and skyrmion utilities, multi-level-cell and write-verify helpers,
aging/radiation/defect models, and the Verilog generator. The submodule path
remains available for compatibility.

::: sc_neurocore.spintronic.spintronic_mapper
