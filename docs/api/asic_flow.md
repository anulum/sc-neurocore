# ASIC Flow

Multi-PDK ASIC generation pipeline. The open-source path targets Sky130 and
GF180MCU through Yosys/OpenROAD-compatible script generation, PDK path
resolution, SC-aware synthesis optimisation, timing constraints, power-grid
analysis, DRC/LVS script generation, and formal verification linkage.

Commercial PDK entries are templates only; they require user-provided Liberty,
LEF, technology LEF, DRC, and LVS decks.

## Quick Start

```python
from sc_neurocore.asic_flow.asic_flow import (
    PDKConfig,
    PDKType,
    OpenSourcePDKResolver,
    ASICFlowGenerator,
    DesignParams,
)

pdk = PDKConfig.from_pdk_type(PDKType.SKY130)
resolution = OpenSourcePDKResolver.resolve(pdk, pdk_root="/path/to/pdks")

flow = ASICFlowGenerator().generate(resolution.pdk, DesignParams())
```

::: sc_neurocore.asic_flow.asic_flow
