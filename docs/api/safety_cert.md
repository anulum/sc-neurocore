# Safety Certification

IEC 61508, ISO 26262, and FDA Class III safety certification stack.
Automated FMEDA analysis, reliability prediction (MIL-HDBK-217F),
formal property linking to SymbiYosys proofs, and compliance auditing.

## Quick Start

```python
from sc_neurocore.safety_cert.safety_cert import (
    SafetyRequirement, RequirementTracer, FMEDAAnalyzer,
    FormalPropertyLink, ReliabilityPredictor, CertificationAuditor,
)
```

::: sc_neurocore.safety_cert.safety_cert
