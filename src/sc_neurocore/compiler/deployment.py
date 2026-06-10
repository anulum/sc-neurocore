# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Deployment utilities facade

"""Deployment utilities for compiled neuron modules.

Nine capabilities provided via modular sub-packages:

1. **Resource estimation** — estimate LUT/FF/DSP/BRAM without synthesis
2. **Constraint file gen** — auto-generate SDC/XDC timing constraints
3. **Host driver gen** — auto-generate C/Python drivers for bus wrappers
4. **Cocotb testbench gen** — generate Python-based verification testbenches
5. **SymbiYosys formal** — one-command bounded model checking scripts
6. **RISC-V driver gen** — bare-metal, FreeRTOS, and Zephyr RTOS drivers
7. **SLR placement** — multi-die PBLOCK constraints for Versal/Agilex
8. **Certification evidence** — DO-254, IEC 61508, ISO 26262 XML traceability
9. **Multi-target compare** — compile to N targets, generate comparison table
"""

from __future__ import annotations

from .certification_gen import (
    CertificationItem,
    generate_certification_evidence,
)
from .cocotb_gen import (
    generate_cocotb_testbench,
)
from .constraint_gen import (
    generate_constraints,
)
from .host_driver_gen import (
    generate_host_driver,
)
from .multi_target import (
    CompilationResult,
    compile_multi_target,
    format_comparison_table,
)
from .resource_estimator import (
    ResourceEstimate,
    estimate_resources,
)
from .riscv_driver import (
    generate_riscv_driver,
)
from .sby_formal import (
    generate_sby_script,
)
from .slr_placement import (
    SLRPlacement,
    generate_slr_constraints,
)

__all__ = [
    "CertificationItem",
    "generate_certification_evidence",
    "generate_cocotb_testbench",
    "generate_constraints",
    "generate_host_driver",
    "CompilationResult",
    "compile_multi_target",
    "format_comparison_table",
    "ResourceEstimate",
    "estimate_resources",
    "generate_riscv_driver",
    "generate_sby_script",
    "SLRPlacement",
    "generate_slr_constraints",
]
