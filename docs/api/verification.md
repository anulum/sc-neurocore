<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- (C) Concepts 1996-2026 Miroslav Sotek. All rights reserved. -->
<!-- (C) Code 2020-2026 Miroslav Sotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SC-NeuroCore documentation -->

# Verification

Formal and functional verification utilities for SNN designs.

- Temporal property checking: verify that SNN outputs satisfy temporal logic specifications
- Equivalence checking: verify Python simulation matches Verilog RTL bit-for-bit
- Coverage metrics: track which neuron states and transitions have been exercised

## Formal SNN Verification Standard

```python
from sc_neurocore.verification import (
    SNNVerificationEvidence,
    VerificationClaimStatus,
    VerificationEvidenceKind,
    VerificationLevel,
    assess_snn_verification_standard,
)

report = assess_snn_verification_standard(
    [
        SNNVerificationEvidence(
            evidence_id="temporal",
            level=VerificationLevel.TEMPORAL_PROPERTIES,
            kind=VerificationEvidenceKind.TEMPORAL_RESULT,
            status=VerificationClaimStatus.PASS,
            description="bounded temporal properties",
        ),
    ]
)
assert not report.passed  # external formal proof and other mandatory evidence are missing
```

The publication-grade profile requires bounded temporal evidence, interval
probability/state bounds, implementation-equivalence evidence, and an external
formal proof log. Missing evidence is reported explicitly; the standard does not
claim unbounded semantic correctness without a passing external proof artefact.

## Generated Code Safety Screen

`CodeSafetyVerifier` performs a conservative AST pass over generated Python
snippets before they are accepted by higher-level verification workflows. It
rejects parse failures, relative imports, direct imports of file/process/network
and dynamic-execution modules such as `os`, `pathlib`, `socket`, `subprocess`,
`importlib`, and `ctypes`, and AST-visible calls to file mutation, process,
network, reflection, dynamic import, and dynamic execution helpers.

The screen is intentionally fail-closed for visible escape routes such as
`open(...)`, `Path(...).write_text(...)`, `socket.socket()`, `eval(...)`,
`__builtins__.eval(...)`, `__builtins__['eval'](...)`, and
`getattr(__builtins__, 'eval')`. Pure local helper calls and allowed scientific
imports such as NumPy remain permitted when no blocked import or call is visible.
This is a preflight screen, not a sandbox or proof of semantic safety.

::: sc_neurocore.verification
    options:
      show_root_heading: true
