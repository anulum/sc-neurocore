# ASIC Flow

Multi-PDK ASIC generation pipeline. Supports Sky130, GF12LP, TSMC N5.
Includes floor-planning, static timing analysis, power estimation,
DRC checking, and formal verification linkage.

## Quick Start

```python
from sc_neurocore.asic_flow.asic_flow import (
    PDKConfig, FloorplanGenerator, TimingAnalyzer, PowerEstimator,
)
```

::: sc_neurocore.asic_flow.asic_flow
