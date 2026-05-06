# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — NIR neuromorphic target adapter packages

"""SDK-free adapter packages for Loihi 2 and SpiNNaker2 planning.

The functions in this module deliberately do not invoke Lava, SpiNNTools, or
physical hardware. They create deterministic handoff artefacts from a NIR graph
and the existing silicon-mapping report so downstream vendor-specific runs have
an explicit manifest, fallback list, and hardware-noise contract.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sc_neurocore.nir_bridge.silicon_mapping import (
    SiliconMappingConfig,
    build_silicon_mapping_report,
)

ADAPTER_SCHEMA_VERSION = "sc-neurocore.nir-neuromorphic-adapter.v1"
SUPPORTED_ADAPTER_TARGETS = ("loihi2", "spinnaker2")

_TARGET_HANDOFFS: dict[str, dict[str, str]] = {
    "loihi2": {
        "adapter_name": "Loihi 2 Lava handoff",
        "vendor_stack": "Intel Lava / Loihi 2",
        "sdk_dependency": "lava-nc",
        "handoff_entrypoint": "Lava Process graph construction from mapped NIR nodes",
        "hardware_status": "requires Lava installation and Loihi 2 access for execution",
    },
    "spinnaker2": {
        "adapter_name": "SpiNNaker2 SpiNNTools handoff",
        "vendor_stack": "SpiNNaker2 / SpiNNTools",
        "sdk_dependency": "SpiNNTools for SpiNNaker2",
        "handoff_entrypoint": "SpiNNTools population/projection construction from mapped NIR nodes",
        "hardware_status": "requires SpiNNaker2 SDK and board access for execution",
    },
}


@dataclass(frozen=True)
class NeuromorphicAdapterPackage:
    """Deterministic handoff package for one neuromorphic hardware target."""

    target_id: str
    adapter_name: str
    vendor_stack: str
    sdk_dependency: str
    handoff_entrypoint: str
    hardware_status: str
    mapping_report: dict[str, Any]
    target_report: dict[str, Any]

    def manifest(self) -> dict[str, Any]:
        """Return a JSON-serialisable adapter manifest."""

        return {
            "adapter_name": self.adapter_name,
            "fallback_requirements": list(self.target_report["fallback_requirements"]),
            "handoff_entrypoint": self.handoff_entrypoint,
            "hardware_status": self.hardware_status,
            "lowering_status": self.target_report["lowering_status"],
            "noise_back_annotation_hooks": list(self.target_report["noise_back_annotation_hooks"]),
            "schema_version": ADAPTER_SCHEMA_VERSION,
            "sdk_dependency": self.sdk_dependency,
            "summary": dict(self.target_report["summary"]),
            "target_id": self.target_id,
            "vendor_stack": self.vendor_stack,
        }

    def files(self) -> dict[str, str]:
        """Return deterministic package files keyed by relative path."""

        manifest = self.manifest()
        limitations = "\n".join(f"- {item}" for item in self.target_report["limitations"])
        fallback = "\n".join(
            f"- {item['node']} ({item['node_type']}): {item['requirement']}"
            for item in self.target_report["fallback_requirements"]
        )
        if not fallback:
            fallback = "- none"
        readme = (
            f"# {self.adapter_name}\n\n"
            f"Target: `{self.target_id}`\n\n"
            f"Vendor stack: {self.vendor_stack}\n\n"
            f"SDK dependency: `{self.sdk_dependency}`\n\n"
            f"Lowering status: `{self.target_report['lowering_status']}`\n\n"
            "## Handoff Boundary\n\n"
            f"{self.handoff_entrypoint}. {self.hardware_status}.\n\n"
            "This package is a deterministic SC-NeuroCore planning artefact. "
            "It does not claim execution on vendor hardware until the vendor SDK "
            "run and board logs are attached.\n\n"
            "## Fallback Requirements\n\n"
            f"{fallback}\n\n"
            "## Limitations\n\n"
            f"{limitations}\n"
        )
        return {
            f"{self.target_id}/adapter_manifest.json": json.dumps(
                manifest, indent=2, sort_keys=True
            )
            + "\n",
            f"{self.target_id}/nir_silicon_mapping_report.json": json.dumps(
                self.mapping_report, indent=2, sort_keys=True
            )
            + "\n",
            f"{self.target_id}/README.md": readme,
        }


def build_neuromorphic_adapter_package(
    source: Any,
    target_id: str,
    config: SiliconMappingConfig | None = None,
) -> NeuromorphicAdapterPackage:
    """Build one Loihi 2 or SpiNNaker2 adapter handoff package."""

    target = _normalise_adapter_target(target_id)
    cfg = _target_config(target, config)
    report = build_silicon_mapping_report(source, cfg)
    target_report = report["targets"][0]
    handoff = _TARGET_HANDOFFS[target]
    return NeuromorphicAdapterPackage(
        target_id=target,
        adapter_name=handoff["adapter_name"],
        vendor_stack=handoff["vendor_stack"],
        sdk_dependency=handoff["sdk_dependency"],
        handoff_entrypoint=handoff["handoff_entrypoint"],
        hardware_status=handoff["hardware_status"],
        mapping_report=report,
        target_report=target_report,
    )


def build_neuromorphic_adapter_bundle(
    source: Any,
    targets: tuple[str, ...] = SUPPORTED_ADAPTER_TARGETS,
    config: SiliconMappingConfig | None = None,
) -> dict[str, NeuromorphicAdapterPackage]:
    """Build deterministic adapter packages for multiple targets."""

    return {
        _normalise_adapter_target(target): build_neuromorphic_adapter_package(
            source, target, config
        )
        for target in targets
    }


def write_neuromorphic_adapter_bundle(
    output_dir: str | Path,
    source: Any,
    targets: tuple[str, ...] = SUPPORTED_ADAPTER_TARGETS,
    config: SiliconMappingConfig | None = None,
) -> dict[str, Path]:
    """Write Loihi 2/SpiNNaker2 adapter manifests and reports to disk."""

    output = Path(output_dir)
    packages = build_neuromorphic_adapter_bundle(source, targets, config)
    written: dict[str, Path] = {}
    for target, package in packages.items():
        for rel_path, content in package.files().items():
            path = output / rel_path
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
            written[f"{target}:{rel_path}"] = path
    return written


def _normalise_adapter_target(target_id: str) -> str:
    target = target_id.lower().replace("-", "_")
    if target not in SUPPORTED_ADAPTER_TARGETS:
        known = ", ".join(SUPPORTED_ADAPTER_TARGETS)
        raise ValueError(f"unsupported adapter target '{target_id}'. Known targets: {known}")
    return target


def _target_config(
    target: str,
    config: SiliconMappingConfig | None,
) -> SiliconMappingConfig:
    if config is None:
        return SiliconMappingConfig(targets=(target,))
    return SiliconMappingConfig(
        targets=(target,),
        bitstream_length=config.bitstream_length,
        event_rate_hz=config.event_rate_hz,
        noise_observations=config.noise_observations,
        artefact_name=config.artefact_name,
    )
