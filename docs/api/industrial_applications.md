<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- (C) Concepts 1996-2026 Miroslav Sotek. All rights reserved. -->
<!-- (C) Code 2020-2026 Miroslav Sotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SC-NeuroCore documentation -->

# Industrial Applications

Industrial application profiles map SC-NeuroCore modules to domain-specific hazards, standards, and evidence categories. They are readiness gates, not deployment approvals: a profile is marked ready only when the supplied evidence bag contains every mandatory category for that domain. This is the commercial bridge between research prototypes and buyer-facing diligence: it turns potential use cases into explicit evidence checklists and missing-evidence reports.

```python
from sc_neurocore.industrial_applications import (
    IndustrialDomain,
    assess_industrial_readiness,
)
from sc_neurocore.safety_cert import EvidenceBag, EvidenceItem

bag = EvidenceBag()
bag.add(EvidenceItem("design.md", "design", "system architecture"))
bag.add(EvidenceItem("tests.xml", "test", "targeted verification"))
bag.add(EvidenceItem("fmeda.md", "analysis", "failure analysis"))
bag.add(EvidenceItem("report.md", "report", "safety report"))

assessment = assess_industrial_readiness(IndustrialDomain.INDUSTRIAL_CONTROL, bag)
assert assessment.ready
```

Built-in profiles cover aerospace, automotive, medical HIL research, rail, and
industrial-control condition monitoring. The profile data deliberately records
hazards and missing evidence instead of making unsupported field-deployment
claims.

## Buyer-facing interpretation

Use this API to turn a potential application into a diligence checklist. A
`ready=false` result is useful: it names the exact missing evidence before a
pilot or commercial deployment discussion. A `ready=true` result means the
local evidence bag satisfies the profile contract; it does not replace
independent certification, target-hardware qualification, cybersecurity review,
or domain-authority acceptance.

::: sc_neurocore.industrial_applications
