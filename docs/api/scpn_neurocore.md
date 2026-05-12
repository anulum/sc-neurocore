<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SC-NeuroCore — SCPN NeuroCore bridge API -->

# SCPN NeuroCore Bridge API

`scpn_neurocore` is the canonical Python namespace for SC-NeuroCore bridge
surfaces consumed by SCPN repositories. It is intentionally separate from the
core `sc_neurocore` simulation package: `sc_neurocore` hosts the neuromorphic
engine, while `scpn_neurocore` hosts source-facing artifacts and datastream
packets for cross-repository SCPN workflows.

The previous unseparated namespace spelling is no longer an active tracked
package. Current consumers must import the bridge through:

```python
from scpn_neurocore.bridge import (
    load_connectome,
    load_live_stream,
    load_power_grid,
    load_tokamak_data,
)
```

Bridge-only imports are kept lightweight. Importing `scpn_neurocore.bridge`
does not eagerly import datastream codecs or optional engine modules. Datastream
exports from `scpn_neurocore` are resolved lazily when requested.

## Artifact Contract

The bridge loaders return `QPUBridgeArtifact` objects. Each artifact carries:

- `K_nm`: finite, symmetric, non-negative coupling matrix with zero diagonal.
- `omega`: finite natural-frequency vector matching `K_nm.shape[0]`.
- `theta0`: optional finite initial phase vector.
- `layer_assignments`: per-oscillator integer layer labels.
- `source_mode`, `source_name`, `normalization`, and `extraction_method`.
- `source_timestamp` or `replay_id`.
- stable SHA-256 hashes for numerical arrays and the full artifact payload.

Publication-safe modes are `recorded`, `replay`, `curated`, and `derived`.
Smoke-test modes are `synthetic`, `simulation`, and `fixture`; they are useful
for interface health checks but are not publication evidence.

## Datastream Contract

`scpn_neurocore.datastream` builds auditable bridge packets containing waveform
samples, AER spike rasters, telemetry summaries, optional QPU artifact hashes,
and optional optimiser observations. The schema version is:

```text
scpn_neurocore.datastream.v1
```

This bridge packet is distinct from the internal 16-layer
`sc_neurocore.scpn.datastream` JSON contract, whose schema remains
`sc-neurocore.scpn.datastream.v1`.

## API Reference

::: scpn_neurocore.bridge
    options:
      show_root_heading: true

::: scpn_neurocore.datastream
    options:
      show_root_heading: true
