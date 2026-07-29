# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Studio capability registry

"""Capability registry for SC-NeuroCore Studio.

The registry is intentionally small and deterministic. It gives Studio a
contract for exposing features without turning UI panels into the source of
truth for capability, evidence, or requirement state.
"""

from __future__ import annotations

import shutil
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum

ExecutableResolver = Callable[[str], str | None]


class CapabilityStatus(str, Enum):
    """Runtime status for a Studio capability."""

    STABLE = "stable"
    EXPERIMENTAL = "experimental"
    DEGRADED = "degraded"
    UNAVAILABLE = "unavailable"


class EvidenceClass(str, Enum):
    """Evidence class attached to a Studio capability."""

    STATIC_INVENTORY = "static_inventory"
    CONTRACT_TEST = "contract_test"
    LOCAL_REGRESSION = "local_regression"
    ISOLATED_BENCHMARK = "isolated_benchmark"
    FORMAL_BUNDLE = "formal_bundle"
    PHYSICAL_MEASUREMENT = "physical_measurement"


@dataclass(frozen=True, slots=True)
class CapabilityRequirement:
    """Requirement needed for a capability to be usable."""

    name: str
    available: bool
    detail: str

    def to_public_dict(self) -> dict[str, object]:
        """Return a public, non-secret representation."""

        return {
            "name": self.name,
            "available": self.available,
            "detail": self.detail,
        }


@dataclass(frozen=True, slots=True)
class CapabilityDescriptor:
    """Static descriptor for a Studio capability."""

    capability_id: str
    title: str
    summary: str
    status: CapabilityStatus
    requirements: tuple[CapabilityRequirement, ...]
    evidence: tuple[EvidenceClass, ...]
    ui_placement: str
    docs_path: str | None


@dataclass(frozen=True, slots=True)
class CapabilityHealth:
    """Runtime health projection for a Studio capability."""

    capability_id: str
    title: str
    summary: str
    status: CapabilityStatus
    healthy: bool
    message: str
    requirements: tuple[CapabilityRequirement, ...]
    evidence: tuple[EvidenceClass, ...]
    ui_placement: str
    docs_path: str | None

    def to_public_dict(self) -> dict[str, object]:
        """Return a public, non-secret API representation."""

        return {
            "capability_id": self.capability_id,
            "title": self.title,
            "summary": self.summary,
            "status": self.status.value,
            "healthy": self.healthy,
            "message": self.message,
            "requirements": [requirement.to_public_dict() for requirement in self.requirements],
            "evidence": [evidence.value for evidence in self.evidence],
            "ui_placement": self.ui_placement,
            "docs_path": self.docs_path,
        }


class CapabilityRegistry:
    """In-memory registry for Studio capabilities."""

    def __init__(self) -> None:
        self._descriptors: dict[str, CapabilityDescriptor] = {}

    def register(self, descriptor: CapabilityDescriptor) -> None:
        """Register a capability descriptor.

        Raises
        ------
        ValueError
            If another capability already uses the same stable ID.
        """

        if descriptor.capability_id in self._descriptors:
            raise ValueError(f"Capability {descriptor.capability_id!r} is already registered.")
        self._descriptors[descriptor.capability_id] = descriptor

    def health(self, capability_id: str) -> CapabilityHealth:
        """Return runtime health for one capability."""

        descriptor = self._descriptors[capability_id]
        unavailable = [item for item in descriptor.requirements if not item.available]
        if unavailable:
            status = CapabilityStatus.UNAVAILABLE
            healthy = False
            message = "One or more capability requirements are unavailable."
        else:
            status = descriptor.status
            healthy = descriptor.status != CapabilityStatus.UNAVAILABLE
            message = "Capability is available." if healthy else "Capability is unavailable."

        return CapabilityHealth(
            capability_id=descriptor.capability_id,
            title=descriptor.title,
            summary=descriptor.summary,
            status=status,
            healthy=healthy,
            message=message,
            requirements=descriptor.requirements,
            evidence=descriptor.evidence,
            ui_placement=descriptor.ui_placement,
            docs_path=descriptor.docs_path,
        )

    def health_all(self) -> list[CapabilityHealth]:
        """Return runtime health for all capabilities sorted by stable ID."""

        return [self.health(capability_id) for capability_id in sorted(self._descriptors)]


def _executable_requirement(
    executable: str,
    *,
    resolver: ExecutableResolver,
) -> CapabilityRequirement:
    """Build a path-free runtime requirement for one external executable."""

    available = resolver(executable) is not None
    return CapabilityRequirement(
        executable,
        available,
        "executable found on PATH" if available else "executable not found on PATH",
    )


def build_default_studio_capability_registry(
    *,
    executable_resolver: ExecutableResolver = shutil.which,
) -> CapabilityRegistry:
    """Build the default capability registry for the current Studio backend."""

    registry = CapabilityRegistry()
    registry.register(
        CapabilityDescriptor(
            capability_id="studio.api",
            title="Studio API",
            summary="FastAPI backend for Visual SNN Design Studio workflows.",
            status=CapabilityStatus.STABLE,
            requirements=(CapabilityRequirement("fastapi", True, "create_app importable"),),
            evidence=(EvidenceClass.CONTRACT_TEST,),
            ui_placement="Admin",
            docs_path="docs/studio/index.md",
        )
    )
    registry.register(
        CapabilityDescriptor(
            capability_id="studio.capability_registry",
            title="Capability Registry",
            summary="Typed inventory for Studio capabilities, requirements, and evidence.",
            status=CapabilityStatus.STABLE,
            requirements=(CapabilityRequirement("studio.platform", True, "registry active"),),
            evidence=(EvidenceClass.CONTRACT_TEST,),
            ui_placement="Admin",
            docs_path="docs/studio/index.md",
        )
    )
    registry.register(
        CapabilityDescriptor(
            capability_id="studio.network_canvas",
            title="Network Canvas",
            summary="Graph validation, simulation, and NIR import/export surfaces.",
            status=CapabilityStatus.EXPERIMENTAL,
            requirements=(
                CapabilityRequirement("sc_neurocore.studio.network_graph", True, "importable"),
            ),
            evidence=(EvidenceClass.CONTRACT_TEST,),
            ui_placement="Build",
            docs_path="docs/studio/network-canvas.md",
        )
    )
    registry.register(
        CapabilityDescriptor(
            capability_id="studio.simulation_workbench",
            title="Simulation Workbench",
            summary="Model and ODE simulation workflow for traces and spike statistics.",
            status=CapabilityStatus.STABLE,
            requirements=(
                CapabilityRequirement("sc_neurocore.studio.simulation", True, "importable"),
                CapabilityRequirement("sc_neurocore.studio.models", True, "model browser active"),
            ),
            evidence=(EvidenceClass.CONTRACT_TEST,),
            ui_placement="Run",
            docs_path="docs/studio/index.md",
        )
    )
    registry.register(
        CapabilityDescriptor(
            capability_id="studio.analysis_suite",
            title="Analysis Suite",
            summary="f-I, bifurcation, sensitivity, heatmap, nullcline, frequency, and compare views.",
            status=CapabilityStatus.STABLE,
            requirements=(
                CapabilityRequirement(
                    "sc_neurocore.studio.analysis", True, "analysis endpoints active"
                ),
            ),
            evidence=(EvidenceClass.CONTRACT_TEST,),
            ui_placement="Analyze",
            docs_path="docs/studio/index.md",
        )
    )
    registry.register(
        CapabilityDescriptor(
            capability_id="studio.compiler_inspector",
            title="Compiler Inspector",
            summary="IR build, verification, SystemVerilog emission, and co-simulation surfaces.",
            status=CapabilityStatus.EXPERIMENTAL,
            requirements=(
                CapabilityRequirement(
                    "sc_neurocore.studio.compiler", True, "compiler endpoints active"
                ),
            ),
            evidence=(EvidenceClass.CONTRACT_TEST,),
            ui_placement="Compile",
            docs_path="docs/studio/compiler-inspector.md",
        )
    )
    registry.register(
        CapabilityDescriptor(
            capability_id="studio.synthesis_dashboard",
            title="Synthesis Dashboard",
            summary="SystemVerilog synthesis and target resource reporting.",
            status=CapabilityStatus.EXPERIMENTAL,
            requirements=(_executable_requirement("yosys", resolver=executable_resolver),),
            evidence=(EvidenceClass.STATIC_INVENTORY,),
            ui_placement="Deploy",
            docs_path="docs/studio/synthesis-dashboard.md",
        )
    )
    registry.register(
        CapabilityDescriptor(
            capability_id="studio.training_monitor",
            title="Training Monitor",
            summary="Surrogate-gradient training job controls and live metric streaming.",
            status=CapabilityStatus.EXPERIMENTAL,
            requirements=(
                CapabilityRequirement(
                    "sc_neurocore.studio.training", True, "training controller active"
                ),
            ),
            evidence=(EvidenceClass.CONTRACT_TEST,),
            ui_placement="Train",
            docs_path="docs/studio/training-monitor.md",
        )
    )
    registry.register(
        CapabilityDescriptor(
            capability_id="studio.project_workspace",
            title="Project Workspace",
            summary="Server-side project save/load/delete and graph-to-synthesis pipeline.",
            status=CapabilityStatus.EXPERIMENTAL,
            requirements=(
                CapabilityRequirement("sc_neurocore.studio.project", True, "project API active"),
            ),
            evidence=(EvidenceClass.CONTRACT_TEST,),
            ui_placement="Workspace",
            docs_path="docs/studio/integration.md",
        )
    )
    registry.register(
        CapabilityDescriptor(
            capability_id="studio.export_tools",
            title="Export Tools",
            summary=(
                "JSON, CSV, SVG, generated script, URL-sharing, and evidence-bundle "
                "export surfaces."
            ),
            status=CapabilityStatus.STABLE,
            requirements=(
                CapabilityRequirement("browser.download", True, "frontend download APIs available"),
                CapabilityRequirement(
                    "sc_neurocore.studio.svg_export", True, "server SVG export active"
                ),
                CapabilityRequirement(
                    "sc_neurocore.studio.platform.evidence_bundle",
                    True,
                    "evidence bundle export active",
                ),
            ),
            evidence=(EvidenceClass.CONTRACT_TEST,),
            ui_placement="Export",
            docs_path="docs/studio/svg-export.md",
        )
    )
    return registry
