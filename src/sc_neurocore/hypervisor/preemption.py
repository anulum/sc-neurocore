# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Hypervisor tenant preemption

"""Capture victim state and hand a hardware region to a preempting tenant."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from sc_neurocore.hypervisor.region import HWRegion
from sc_neurocore.hypervisor.tenant import Tenant, TenantState


@dataclass
class PreemptionEvent:
    """Record of a preemption event."""

    victim_id: str
    preemptor_id: str
    cycle: int
    state_saved: bool


class PreemptionManager:
    """Handles preemption with state checkpoint/restore."""

    def __init__(self) -> None:
        self.events: List[PreemptionEvent] = []
        self.saved_states: Dict[str, TenantState] = {}

    def preempt(
        self,
        victim: Tenant,
        preemptor: Tenant,
        region: HWRegion,
        cycle: int,
    ) -> PreemptionEvent:
        """Preempt victim and give region to preemptor."""
        state_saved = False
        if victim.state is not None:
            victim.state.compute_checksum()
            self.saved_states[victim.tenant_id] = victim.state
            state_saved = True

        victim.active = False
        victim.region_id = None
        region.tenant_id = preemptor.tenant_id
        preemptor.region_id = region.region_id
        preemptor.active = True

        evt = PreemptionEvent(victim.tenant_id, preemptor.tenant_id, cycle, state_saved)
        self.events.append(evt)
        return evt

    def restore_preempted(self, tenant: Tenant) -> bool:
        """Restore a previously preempted tenant's state."""
        if tenant.tenant_id not in self.saved_states:
            return False
        tenant.state = self.saved_states.pop(tenant.tenant_id)
        return True
