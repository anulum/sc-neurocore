# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Compiler-service request fixtures

"""Shared construction fixtures for compiler-service contracts."""

from __future__ import annotations

from sc_neurocore.compiler_service import CompilerServiceRequest, DigitalTwinSyncContract
from sc_neurocore.optimizer.sc_optimizer import HardwareBudget, LayerProfile
from sc_neurocore.optimizer.surrogate_sc_optimizer import TargetHardwareProfile


def _target(name: str = "pynq-z2") -> TargetHardwareProfile:
    return TargetHardwareProfile(
        name=name,
        budget=HardwareBudget(max_luts=12_000, max_power_mw=2_500.0, max_latency_cycles=1024),
    )


def _request(*changed_fields: str) -> CompilerServiceRequest:
    return CompilerServiceRequest(
        request_id="req-001",
        target=_target(),
        network=(
            LayerProfile("input", mac_count=128),
            LayerProfile("hidden", mac_count=256, is_critical_path=True),
        ),
        changed_fields=changed_fields or ("weights",),
        twin_sync=DigitalTwinSyncContract(session_id="twin-a", twin_nodes=2),
        evidence_payload={"results": []},
    )
