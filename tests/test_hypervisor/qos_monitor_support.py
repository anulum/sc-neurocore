# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_qos_monitor.py

from __future__ import annotations

"""Verify throughput metering, SLA detection, and definition ownership."""
import ast
from pathlib import Path
import pytest
from sc_neurocore.hypervisor import hypervisor as compatibility_surface
from sc_neurocore.hypervisor import qos_monitor as qos_owner
from sc_neurocore.hypervisor.qos_monitor import BandwidthMeter, SLAMonitor
from sc_neurocore.hypervisor.tenant import Tenant, TenantPriority
def _tenant(
    tid: str = "t0", name: str = "test", prio: TenantPriority = TenantPriority.NORMAL
) -> Tenant:
    return Tenant(tenant_id=tid, name=name, priority=prio)

__all__ = ['ast', 'Path', 'pytest', 'compatibility_surface', 'qos_owner', 'BandwidthMeter', 'SLAMonitor', 'Tenant', 'TenantPriority', '_tenant']
