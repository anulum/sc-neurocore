# Memristor Mapper

Memristive crossbar array mapper with non-linear conductance,
device aging simulation, and power/area estimation.

## Quick Start

```python
from sc_neurocore.memristor import (
    CrossbarArray, ConductanceModel, AgingSimulator, CrossbarEstimator,
)
```

The package facade exports the documented mapper surface, including
`MemristorMapper`, `MemristorTechnology`, `CrossbarTopology`,
`CompensationStrategy`, Monte Carlo reports, variability injection helpers,
and the SystemVerilog emitter. The submodule path remains available for
compatibility.

::: sc_neurocore.memristor.memristor_mapper
