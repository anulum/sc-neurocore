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

## 2026-04-30 one-command bundle helper

The Python API now includes `generate_asic_flow_bundle(...)` for the
roadmap "one-command ASIC flow" path. It writes the complete Yosys/OpenROAD
deck set plus `asic_flow_manifest.json` into a chosen output directory:

```python
from sc_neurocore.asic_flow.asic_flow import DesignParams, generate_asic_flow_bundle

bundle = generate_asic_flow_bundle(
    "build/asic/sky130_demo",
    pdk_type="sky130",
    design=DesignParams(top_module="sc_neurocore_top", rtl_files=["rtl/top.sv"]),
    pdk_root="/opt/pdks",
    n_neurons=32,
    n_synapses=512,
    bitstream_width=256,
    n_aer_ports=8,
)

print(bundle.manifest_path)
print(bundle.estimate.dynamic_power_mw)
```

The helper does not run external EDA tools. The manifest explicitly records
`external_eda_executed: false` and `physical_ppa_claim_allowed: false` until a
real Yosys/OpenROAD run, exact OpenROAD binary or container digest, and PDK
revision are attached as evidence. When `require_pdk_files=True`, missing
Liberty/LEF/tech LEF paths are listed as blockers instead of being hidden.

The package facade also exposes the helper for stable application imports:

```python
from sc_neurocore.asic_flow import ASICFlowBundle, generate_asic_flow_bundle
```

`tests/test_asic_flow/test_asic_flow_package_api.py` locks the package-level
exports and verifies that the facade generates a manifest-bearing
`ASICFlowBundle` without running external EDA tools.
