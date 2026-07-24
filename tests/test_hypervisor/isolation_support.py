# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_isolation.py

from __future__ import annotations

"""Verify tenant address isolation and its historical import surface."""
import ast
from pathlib import Path
from sc_neurocore.hypervisor import hypervisor as compatibility_surface
from sc_neurocore.hypervisor import isolation
from sc_neurocore.hypervisor.isolation import (
    BitstreamFirewall,
    FirewallRule,
    verify_isolation,
)

__all__ = [
    "ast",
    "Path",
    "compatibility_surface",
    "isolation",
    "BitstreamFirewall",
    "FirewallRule",
    "verify_isolation",
]
