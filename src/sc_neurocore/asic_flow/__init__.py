# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — sc_neurocore.asic_flow -- Multi-PDK ASIC generation and verification flow

"""Package facade for one-command ASIC flow bundle generation.

Tier: industrial.
"""

from sc_neurocore.asic_flow.asic_flow import ASICFlowBundle, generate_asic_flow_bundle

__tier__ = "industrial"

__all__ = ["ASICFlowBundle", "generate_asic_flow_bundle"]
