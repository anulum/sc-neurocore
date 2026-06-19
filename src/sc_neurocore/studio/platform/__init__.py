# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Studio platform contracts

"""Platform contracts for SC-NeuroCore Studio."""

from sc_neurocore.studio.platform.capabilities import (
    CapabilityDescriptor,
    CapabilityHealth,
    CapabilityRegistry,
    CapabilityRequirement,
    CapabilityStatus,
    EvidenceClass,
    build_default_studio_capability_registry,
)

__all__ = [
    "CapabilityDescriptor",
    "CapabilityHealth",
    "CapabilityRegistry",
    "CapabilityRequirement",
    "CapabilityStatus",
    "EvidenceClass",
    "build_default_studio_capability_registry",
]
