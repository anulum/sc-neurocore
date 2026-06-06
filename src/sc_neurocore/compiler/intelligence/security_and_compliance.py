# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Security and compliance facade

"""Security analysis, IP protection, and compliance facade."""

from __future__ import annotations

from .bitstream_encryption import (
    generate_bitstream_encryption,
)
from .carbon_footprint import (
    CarbonEstimate,
    estimate_carbon_footprint,
)
from .checksum import (
    embed_model_checksum,
)
from .ip_obfuscation import (
    ObfuscationResult,
    obfuscate_ip,
)
from .license_compliance import (
    LicenseCheck,
    check_license_compliance,
)
from .pqc_protection import (
    PQCProtection,
    protect_ip_pqc,
)
from .sbom_gen import (
    SBOM,
    generate_sbom,
)
from .side_channel_lint import (
    SideChannelFinding,
    lint_side_channels,
)
from .supply_chain_risk import (
    SupplyChainRisk,
    score_supply_chain_risk,
)
from .trojan_lint import (
    TrojanLintResult,
    lint_hardware_trojans,
)
from .watermark import (
    WatermarkResult,
    embed_watermark,
)

__all__ = [
    "CarbonEstimate",
    "LicenseCheck",
    "ObfuscationResult",
    "PQCProtection",
    "SBOM",
    "SideChannelFinding",
    "SupplyChainRisk",
    "TrojanLintResult",
    "WatermarkResult",
    "check_license_compliance",
    "embed_model_checksum",
    "embed_watermark",
    "estimate_carbon_footprint",
    "generate_bitstream_encryption",
    "generate_sbom",
    "lint_hardware_trojans",
    "lint_side_channels",
    "obfuscate_ip",
    "protect_ip_pqc",
    "score_supply_chain_risk",
]
