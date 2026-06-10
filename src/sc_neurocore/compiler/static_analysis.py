# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Static analysis tools for fixed-point ODE compilation

"""Static analysis utilities for the equation compiler.

Provides five capabilities that no other neuromorphic compiler offers:

1. **Guard-bit auto-computation** — determine how many extra MSBs are needed
   in intermediate accumulators to prevent silent overflow.

2. **Formal overflow proof** — use interval arithmetic on the ODE AST to
   statically prove that no overflow occurs at a given precision.

3. **SystemVerilog Assertion (SVA) generation** — emit formal verification
   properties for safety-critical certification (DO-254 / IEC 61508).

4. **Pipeline stage analysis** — compute critical path depth and required
   pipeline stages for high-frequency targets.

5. **Power estimation** — switching-activity-based dynamic/static power
   model from generated Verilog without synthesis.
"""

from __future__ import annotations

from .guard_bits import (
    compute_guard_bits,
    compute_guard_bits_multi,
)
from .overflow_proof import (
    FixedPointEnvelopeProof,
    Interval,
    OverflowProofResult,
    prove_fixed_point_envelope,
    prove_no_overflow,
)
from .pipeline_analysis import (
    critical_path_depth,
    pipeline_analysis,
    pipeline_stages_needed,
)
from .power_estimator import (
    PowerEstimate,
    estimate_power,
)
from .sva_gen import (
    generate_sva,
)

__all__ = [
    "compute_guard_bits",
    "compute_guard_bits_multi",
    "FixedPointEnvelopeProof",
    "Interval",
    "OverflowProofResult",
    "prove_fixed_point_envelope",
    "prove_no_overflow",
    "critical_path_depth",
    "pipeline_analysis",
    "pipeline_stages_needed",
    "PowerEstimate",
    "estimate_power",
    "generate_sva",
]
