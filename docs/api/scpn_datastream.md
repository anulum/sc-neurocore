<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SC-NeuroCore — SCPN datastream JSON contract -->

# SCPN Datastream JSON Contract

**Module:** `sc_neurocore.scpn.datastream`
**Purpose:** Deterministic JSON exchange format for SCPN cross-repository tests.

The SCPN datastream carries a canonical 16-layer payload:

- `schema_version`: must equal `sc-neurocore.scpn.datastream.v1`
- `dt_s`: positive finite timestep
- `seed`: integer generator seed
- `probabilities`: 2-D finite numeric array with values in `[0, 1]`
- `spike_train`: 2-D binary numeric array matching `probabilities`
- `omega_rad_s`: finite numeric vector with 16 entries
- `knm`: finite symmetric `16 x 16` numeric matrix with zero diagonal

`read_scpn_datastream()` parses UTF-8 JSON and rejects non-object roots.
`SCPNDatastream.from_json_dict()` validates the schema and required fields
before converting arrays to numpy. The loader rejects missing fields,
non-numeric array values, non-finite values, fractional spike entries, shape
mismatches, and unsupported schema versions.

```python
from sc_neurocore.scpn import generate_scpn_datastream, read_scpn_datastream, write_scpn_datastream

stream = generate_scpn_datastream(n_steps=32, dt_s=0.01, seed=1729)
write_scpn_datastream("scpn_stream.json", stream)
loaded = read_scpn_datastream("scpn_stream.json")
```
