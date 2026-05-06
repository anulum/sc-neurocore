<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996-2026 Miroslav Sotek. All rights reserved. -->
<!-- © Code 2020-2026 Miroslav Sotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->

# Stochastic Photonic Co-Design Loop

**Module:** `sc_neurocore.bridges.photonic_codesign`

The stochastic photonic co-design loop connects four existing SC-NeuroCore
surfaces:

1. deterministic LFSR-backed SC bitstream generation;
2. photonic NoC compilation through `SCToPhotonic`;
3. optical pulse/netlist compilation through `PhotonicCompiler`;
4. representative FDTD smoke simulation for the highest-transition channel.

It does not claim foundry sign-off. The exported layout manifest is a PDA
handoff record and explicitly lists the external PDK, DRC, LVS, and FDTD
sign-off inputs that remain necessary before fabrication claims.

## Public API

```python
from sc_neurocore.bridges import (
    PhotonicCoDesignConfig,
    StochasticPhotonicCoDesignLoop,
)

loop = StochasticPhotonicCoDesignLoop(
    PhotonicCoDesignConfig(bitstream_length=1024, run_fdtd=True)
)
report = loop.compile(adjacency, probabilities=[0.25, 0.5, 0.75])
```

`report.feasible` is true only when:

- every SC channel density is within the configured two-sided Hoeffding bound;
- all routed optical paths pass detector sensitivity;
- worst optical power margin is above `min_power_margin_db`;
- WDM crosstalk is below `max_crosstalk_db`;
- the representative FDTD pulse has non-zero field energy when FDTD is enabled.

When any condition fails, the report keeps the evidence and returns explicit
`blockers` rather than hiding the failure.

## Evidence Fields

`PhotonicCoDesignReport.to_json()` records:

- compact per-channel bitstream evidence: target probability, measured
  probability, popcount, density error, transition count, packed word count;
- photonic design size: nodes, waveguides, MZI gates, WDM channels, area;
- optical compiler outputs: modulator count, mean optical power, phase coverage,
  representative FDTD energy;
- power-budget and WDM crosstalk analyses;
- stochastic correlation coefficient matrix for all encoded SC channels;
- PDA handoff manifest with MZI cells, waveguide routes, WDM assignments, and
  missing external sign-off requirements.

Packed bit words are retained in memory on `BitstreamEvidence.packed_words` for
exact replay. The JSON export intentionally stores only compact evidence so
routine reports do not become large binary artefacts.

## Physical Boundary

The loop is suitable for pre-layout feasibility, toolchain handoff, and
regression testing. A fabrication or publication claim still requires:

- foundry PDK layer map and calibrated component models;
- DRC and LVS decks;
- full 2D/3D photonic simulation for the exported layout;
- measured laser, modulator, detector, and thermal-tuning parameters;
- lab measurement data after fabrication.

Until those artefacts exist, the correct status is co-design readiness, not
fabrication validation.
