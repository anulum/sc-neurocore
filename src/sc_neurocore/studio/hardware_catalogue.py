# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Registered hardware catalogue for Studio

"""Project hardware registrations without promoting them to physical evidence."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

from sc_neurocore.hardware.device import DEVICE_CATALOG, DeviceFamily
from sc_neurocore.nir_bridge.hardware_targets import (
    NeuromorphicHardwareProfile,
    available_hardware_profiles,
)
from sc_neurocore.nir_bridge.neuromorphic_adapters import SUPPORTED_ADAPTER_TARGETS
from sc_neurocore.studio.synthesis_profiles import SynthesisTargetProfile, synthesis_target_profiles

_CORE_NAMES = {
    DeviceFamily.LOIHI: "Loihi",
    DeviceFamily.LOIHI2: "Loihi 2",
    DeviceFamily.SPINNAKER: "SpiNNaker",
    DeviceFamily.SPINNAKER2: "SpiNNaker2",
    DeviceFamily.BRAINSCALES: "BrainScaleS",
    DeviceFamily.BRAINSCALES2: "BrainScaleS-2",
    DeviceFamily.FPGA_GENERIC: "Generic FPGA planning profile",
    DeviceFamily.AKIDA: "Akida",
}


class HardwareCatalogueEntry(BaseModel):
    """One registered target and the exact software paths it declares.

    A NIR capability manifest describes intended lowering constraints, not
    conformance to a vendor SDK. Physical evidence is deliberately separate
    from registration; this catalogue does not import measurement receipts.
    """

    model_config = ConfigDict(frozen=True)

    target_id: str
    name: str
    kind: Literal["neuromorphic", "fpga", "board"]
    registration_sources: tuple[str, ...]
    planning_profile: bool = False
    nir_profile: NeuromorphicHardwareProfile | None = None
    adapter_package: bool = False
    synthesis_profile: SynthesisTargetProfile | None = None
    driver: str | None = None
    requirements: tuple[str, ...]
    limitations: tuple[str, ...]
    evidence_level: Literal["registration"] = "registration"
    physical_evidence: tuple[str, ...] = ()


class HardwareCatalogue(BaseModel):
    """A deterministic catalogue of registrations, with no hardware probing."""

    model_config = ConfigDict(frozen=True)

    schema_version: Literal["studio.hardware-catalogue.v1"] = "studio.hardware-catalogue.v1"
    entries: tuple[HardwareCatalogueEntry, ...]
    evidence_scope: str = (
        "Registration only. Tool availability and source-bound run evidence are "
        "separate; no physical measurement receipts are attached here."
    )


def hardware_catalogue() -> HardwareCatalogue:
    """Read current public registries without opening hardware or running tools.

    Returns
    -------
    HardwareCatalogue
        Device planning profiles, NIR manifests, synthesis configurations and
        the existing PYNQ driver. Identity matching is exact: BrainScaleS-2
        and the separately declared BrainScaleS-3 manifest remain distinct.
        Device planning constants are not published as measured specifications.
        This function performs no vendor SDK calls, tool probes or writes.
    """
    entries: dict[str, HardwareCatalogueEntry] = {}
    nir_profiles = {profile.target_id: profile for profile in available_hardware_profiles()}
    for family in DEVICE_CATALOG:
        key = family.name.lower()
        entries[key] = HardwareCatalogueEntry(
            target_id=key,
            name=_CORE_NAMES.get(family, family.name.replace("_", " ").title()),
            kind="fpga" if family is DeviceFamily.FPGA_GENERIC else "neuromorphic",
            registration_sources=("sc_neurocore.hardware.DEVICE_CATALOG",),
            planning_profile=True,
            requirements=("SC-NeuroCore hardware resource-planning API",),
            limitations=(
                "Planning assumptions do not establish vendor execution or measured power.",
            ),
        )
    for key, profile in nir_profiles.items():
        existing = entries.get(key)
        sources = () if existing is None else existing.registration_sources
        entries[key] = HardwareCatalogueEntry(
            target_id=key,
            name=profile.display_name,
            kind="neuromorphic",
            registration_sources=(*sources, "sc_neurocore.nir_bridge.available_hardware_profiles"),
            planning_profile=existing is not None,
            nir_profile=profile,
            adapter_package=key in SUPPORTED_ADAPTER_TARGETS,
            requirements=(
                "NIR lowering contract; vendor software and device access for execution",
            ),
            limitations=(
                "Capability manifests and adapter packages do not invoke a vendor runtime.",
                *profile.notes,
            ),
        )
    for target_profile in synthesis_target_profiles():
        entries["fpga:" + target_profile.target] = HardwareCatalogueEntry(
            target_id="fpga:" + target_profile.target,
            name=target_profile.capacity_device,
            kind="fpga",
            registration_sources=(
                "sc_neurocore.studio.synthesis_profiles.synthesis_target_profiles",
            ),
            synthesis_profile=target_profile,
            requirements=("Yosys",)
            + (() if target_profile.routing_tool is None else (target_profile.routing_tool,)),
            limitations=(
                "Capacity is a reference-device bound, not a board or timing verdict.",
                "An installed tool is not evidence that this design was compiled or measured.",
            ),
        )
    entries["board:pynq_z2"] = HardwareCatalogueEntry(
        target_id="board:pynq_z2",
        name="PYNQ-Z2",
        kind="board",
        registration_sources=("sc_neurocore.drivers.sc_neurocore_driver.SC_NeuroCore_Driver",),
        driver="sc_neurocore.drivers.sc_neurocore_driver.SC_NeuroCore_Driver",
        requirements=("PYNQ runtime on the target board", "Matching bitstream and SCPN Layer 1 IP"),
        limitations=(
            "Driver registration is not board detection or a successful hardware run.",
            "EMULATION mode is software execution and does not attest physical behaviour.",
        ),
    )
    return HardwareCatalogue(entries=tuple(entries[key] for key in sorted(entries)))
