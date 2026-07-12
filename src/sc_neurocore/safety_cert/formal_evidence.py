# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Automated Safety & Regulatory Certification Generator

"""Formal-property evidence, proof coverage, and gap detection."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List

from sc_neurocore.safety_cert.standards import SILLevel


@dataclass
class FormalProperty:
    """One formally verified property."""

    prop_id: str
    module: str
    description: str
    property_type: str  # "assert" | "cover" | "assume"
    status: str = "unknown"  # "proven" | "failed" | "unknown"
    engine: str = "SymbiYosys"
    depth: int = 20
    sby_file: str = ""

    def __post_init__(self) -> None:
        """Validate a formal-property evidence record."""
        if not isinstance(self.prop_id, str) or not self.prop_id.strip():
            raise ValueError("prop_id must be a non-empty string")
        if not isinstance(self.module, str) or not self.module.strip():
            raise ValueError("module must be a non-empty string")
        if not isinstance(self.description, str) or not self.description.strip():
            raise ValueError("description must be a non-empty string")
        if not isinstance(self.property_type, str) or self.property_type not in {
            "assert",
            "cover",
            "assume",
        }:
            raise ValueError("property_type must be one of: assert, cover, assume")
        if not isinstance(self.status, str) or self.status not in {"proven", "failed", "unknown"}:
            raise ValueError("status must be one of: proven, failed, unknown")
        if not isinstance(self.engine, str) or not self.engine.strip():
            raise ValueError("engine must be a non-empty string")
        if isinstance(self.depth, bool) or not isinstance(self.depth, int) or self.depth < 0:
            raise ValueError("depth must be a non-negative integer")
        if not isinstance(self.sby_file, str):
            raise ValueError("sby_file must be a string")


@dataclass
class FormalProofCertificate:
    """Immutable-content identifier and report for formal-property evidence."""

    properties: List[FormalProperty] = field(default_factory=list)
    generation_timestamp: str = ""
    tool_version: str = "SymbiYosys"
    certificate_hash: str = ""

    def __post_init__(self) -> None:
        """Validate the certificate container and its property records."""
        if not isinstance(self.generation_timestamp, str):
            raise ValueError("generation_timestamp must be a string")
        if not isinstance(self.tool_version, str) or not self.tool_version.strip():
            raise ValueError("tool_version must be a non-empty string")
        if not isinstance(self.certificate_hash, str):
            raise ValueError("certificate_hash must be a string")
        for prop in self.properties:
            if not isinstance(prop, FormalProperty):
                raise ValueError("properties must contain FormalProperty entries")

    def add_property(self, prop: FormalProperty) -> None:
        """Append one formal-property evidence record."""
        if not isinstance(prop, FormalProperty):
            raise ValueError("prop must be a FormalProperty")
        self.properties.append(prop)

    @property
    def proven_count(self) -> int:
        """Return the number of properties explicitly marked proven."""
        for prop in self.properties:
            if not isinstance(prop, FormalProperty):
                raise ValueError("properties must contain FormalProperty entries")
            if not isinstance(prop.status, str) or prop.status not in {
                "proven",
                "failed",
                "unknown",
            }:
                raise ValueError("properties statuses must be one of: proven, failed, unknown")
        return sum(1 for p in self.properties if p.status == "proven")

    @property
    def total_count(self) -> int:
        """Return the number of property records."""
        for prop in self.properties:
            if not isinstance(prop, FormalProperty):
                raise ValueError("properties must contain FormalProperty entries")
            if not isinstance(prop.prop_id, str) or not prop.prop_id.strip():
                raise ValueError("properties prop_id values must be non-empty strings")
        return len(self.properties)

    @property
    def pass_rate(self) -> float:
        """Return the proven fraction, or zero for an empty certificate."""
        return self.proven_count / self.total_count if self.total_count > 0 else 0.0

    def _canonical_bytes(self) -> bytes:
        for prop in self.properties:
            if not isinstance(prop, FormalProperty):
                raise ValueError("properties must contain FormalProperty entries")
            if not isinstance(prop.prop_id, str) or not prop.prop_id.strip():
                raise ValueError("properties prop_id values must be non-empty strings")
            if not isinstance(prop.module, str) or not prop.module.strip():
                raise ValueError("properties modules must be non-empty strings")
            if not isinstance(prop.description, str) or not prop.description.strip():
                raise ValueError("properties descriptions must be non-empty strings")
            if not isinstance(prop.property_type, str) or prop.property_type not in {
                "assert",
                "cover",
                "assume",
            }:
                raise ValueError(
                    "properties property_type values must be one of: assert, cover, assume"
                )
            if not isinstance(prop.status, str) or prop.status not in {
                "proven",
                "failed",
                "unknown",
            }:
                raise ValueError("properties statuses must be one of: proven, failed, unknown")
            if not isinstance(prop.engine, str) or not prop.engine.strip():
                raise ValueError("properties engines must be non-empty strings")
            if isinstance(prop.depth, bool) or not isinstance(prop.depth, int) or prop.depth < 0:
                raise ValueError("properties depths must be non-negative integers")
            if not isinstance(prop.sby_file, str):
                raise ValueError("properties sby_file values must be strings")
        prop_ids = [p.prop_id for p in self.properties]
        if len(prop_ids) != len(set(prop_ids)):
            raise ValueError("properties must not contain duplicate prop_id values")
        payload = {
            "properties": [
                {
                    "depth": prop.depth,
                    "description": prop.description,
                    "engine": prop.engine,
                    "module": prop.module,
                    "prop_id": prop.prop_id,
                    "property_type": prop.property_type,
                    "sby_file": prop.sby_file,
                    "status": prop.status,
                }
                for prop in sorted(self.properties, key=lambda item: item.prop_id)
            ],
            "tool_version": self.tool_version,
        }
        return json.dumps(
            payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True
        ).encode("utf-8")

    def content_sha256(self) -> str:
        """Return the full SHA-256 digest of every material property field."""
        return hashlib.sha256(self._canonical_bytes()).hexdigest()

    def compute_hash(self, *, generated_at: str | None = None) -> str:
        """Set and return the historical 32-character certificate identifier.

        The identifier remains a prefix for compatibility. Integrity-sensitive
        consumers should use :meth:`content_sha256`, which returns all 64
        hexadecimal characters.
        """
        self.certificate_hash = self.content_sha256()[:32]
        timestamp = datetime.now(timezone.utc).isoformat() if generated_at is None else generated_at
        if not isinstance(timestamp, str) or not timestamp.strip():
            raise ValueError("generated_at must be a non-empty string when provided")
        self.generation_timestamp = timestamp
        return self.certificate_hash

    def generate_report(self, *, generated_at: str | None = None) -> str:
        """Render the formal-property evidence as Markdown."""
        for prop in self.properties:
            if not isinstance(prop, FormalProperty):
                raise ValueError("properties must contain FormalProperty entries")
            if not isinstance(prop.prop_id, str) or not prop.prop_id.strip():
                raise ValueError("properties prop_id values must be non-empty strings")
            if not isinstance(prop.property_type, str) or prop.property_type not in {
                "assert",
                "cover",
                "assume",
            }:
                raise ValueError(
                    "properties property_type values must be one of: assert, cover, assume"
                )
        if not self.certificate_hash:
            self.compute_hash(generated_at=generated_at)
        elif generated_at is not None:
            if not isinstance(generated_at, str) or not generated_at.strip():
                raise ValueError("generated_at must be a non-empty string when provided")
            self.generation_timestamp = generated_at
        timestamp = self.generation_timestamp or datetime.now(timezone.utc).isoformat()
        lines = [
            "# Formal Proof Certificate",
            f"Generated: {timestamp}",
            f"Evidence SHA-256: {self.content_sha256()}",
            f"Certificate ID: {self.certificate_hash}",
            f"Tool: {self.tool_version}",
            f"Properties: {self.proven_count}/{self.total_count} proven ({self.pass_rate:.0%})",
            "",
            "| Property | Module | Type | Status | Depth |",
            "|----------|--------|------|--------|-------|",
        ]
        for p in self.properties:
            lines.append(
                f"| {p.prop_id} | {p.module} | {p.property_type} | {p.status} | {p.depth} |"
            )
        return "\n".join(lines)


class ProofTestCoverage:
    """Screen formal-proof completeness without making a compliance claim."""

    @staticmethod
    def coverage_from_proofs(properties: List[FormalProperty]) -> float:
        """Formal proof coverage = proven / total asserts."""
        if not isinstance(properties, list):
            raise ValueError("properties must be a list")
        for prop in properties:
            if not isinstance(prop, FormalProperty):
                raise ValueError("properties must contain FormalProperty entries")
            if not isinstance(prop.prop_id, str) or not prop.prop_id.strip():
                raise ValueError("properties prop_id values must be non-empty strings")
            if not isinstance(prop.status, str) or prop.status not in {
                "proven",
                "failed",
                "unknown",
            }:
                raise ValueError("properties statuses must be one of: proven, failed, unknown")
            if not isinstance(prop.property_type, str) or prop.property_type not in {
                "assert",
                "cover",
                "assume",
            }:
                raise ValueError(
                    "properties property_type values must be one of: assert, cover, assume"
                )
        asserts = [p for p in properties if p.property_type == "assert"]
        if not asserts:
            return 0.0
        proven = sum(1 for p in asserts if p.status == "proven")
        return proven / len(asserts)

    @staticmethod
    def dc_to_sil(dc: float) -> SILLevel:
        """Return the legacy DC-only screening label.

        Diagnostic coverage alone is insufficient to establish a SIL. This
        compatibility method is a triage heuristic, not a certification result.
        """
        if isinstance(dc, bool) or not isinstance(dc, int | float):
            raise ValueError("dc must be a finite value in [0, 1]")
        value = float(dc)
        if not math.isfinite(value) or value < 0.0 or value > 1.0:
            raise ValueError("dc must be a finite value in [0, 1]")
        if value >= 0.99:
            return SILLevel.SIL_4
        if value >= 0.97:
            return SILLevel.SIL_3
        if value >= 0.90:
            return SILLevel.SIL_2
        if value >= 0.60:
            return SILLevel.SIL_1
        return SILLevel.SIL_1

    @staticmethod
    def uncovered_modules(properties: List[FormalProperty], all_modules: List[str]) -> List[str]:
        """Modules with no formal proofs."""
        if not isinstance(properties, list):
            raise ValueError("properties must be a list")
        for prop in properties:
            if not isinstance(prop, FormalProperty):
                raise ValueError("properties must contain FormalProperty entries")
            if not isinstance(prop.prop_id, str) or not prop.prop_id.strip():
                raise ValueError("properties prop_id values must be non-empty strings")
            if not isinstance(prop.module, str) or not prop.module.strip():
                raise ValueError("properties modules must be non-empty strings")
        if not isinstance(all_modules, list):
            raise ValueError("all_modules must be a list")
        for module in all_modules:
            if not isinstance(module, str) or not module.strip():
                raise ValueError("all_modules must contain non-empty strings")
            if module != module.strip():
                raise ValueError(
                    "all_modules entries must not contain leading or trailing whitespace"
                )
        covered = {p.module for p in properties}
        uncovered: list[str] = []
        seen: set[str] = set()
        for module in all_modules:
            if module not in covered and module not in seen:
                uncovered.append(module)
                seen.add(module)
        return uncovered


@dataclass
class PropertyGap:
    """One module with insufficient formal coverage."""

    module: str
    total_properties: int
    proven_properties: int
    missing_types: List[str]  # property types not covered

    def __post_init__(self) -> None:
        """Validate the formal-property gap counts."""
        if not isinstance(self.module, str) or not self.module.strip():
            raise ValueError("module must be a non-empty string")
        if isinstance(self.total_properties, bool) or not isinstance(self.total_properties, int):
            raise ValueError("total_properties must be a non-negative integer")
        if self.total_properties < 0:
            raise ValueError("total_properties must be a non-negative integer")
        if isinstance(self.proven_properties, bool) or not isinstance(self.proven_properties, int):
            raise ValueError("proven_properties must be a non-negative integer")
        if self.proven_properties < 0:
            raise ValueError("proven_properties must be a non-negative integer")
        if self.proven_properties > self.total_properties:
            raise ValueError("proven_properties cannot exceed total_properties")
        for item in self.missing_types:
            if not isinstance(item, str) or not item.strip():
                raise ValueError("missing_types must contain non-empty strings")

    @property
    def coverage(self) -> float:
        """Return the proven fraction, or zero when no properties exist."""
        return self.proven_properties / self.total_properties if self.total_properties > 0 else 0.0


class FormalPropertyGapDetector:
    """Detects modules with insufficient formal verification coverage."""

    REQUIRED_TYPES = ["assert", "cover"]

    @classmethod
    def detect(
        cls, properties: List[FormalProperty], required_modules: List[str]
    ) -> List[PropertyGap]:
        """Return missing or incomplete formal-property coverage by module."""
        if not isinstance(properties, list):
            raise ValueError("properties must be a list")
        for prop in properties:
            if not isinstance(prop, FormalProperty):
                raise ValueError("properties must contain FormalProperty entries")
            if not isinstance(prop.prop_id, str) or not prop.prop_id.strip():
                raise ValueError("properties prop_id values must be non-empty strings")
            if not isinstance(prop.property_type, str) or prop.property_type not in {
                "assert",
                "cover",
                "assume",
            }:
                raise ValueError(
                    "properties property_type values must be one of: assert, cover, assume"
                )
            if not isinstance(prop.status, str) or prop.status not in {
                "proven",
                "failed",
                "unknown",
            }:
                raise ValueError("properties statuses must be one of: proven, failed, unknown")
            if not isinstance(prop.module, str) or not prop.module.strip():
                raise ValueError("properties modules must be non-empty strings")
        if not isinstance(required_modules, list):
            raise ValueError("required_modules must be a list")
        for module in required_modules:
            if not isinstance(module, str) or not module.strip():
                raise ValueError("required_modules must contain non-empty strings")
            if module != module.strip():
                raise ValueError(
                    "required_modules entries must not contain leading or trailing whitespace"
                )
        by_module: Dict[str, List[FormalProperty]] = {}
        for p in properties:
            by_module.setdefault(p.module, []).append(p)

        gaps = []
        seen_required: set[str] = set()
        for mod in required_modules:
            if mod in seen_required:
                continue
            seen_required.add(mod)
            props = by_module.get(mod, [])
            proven = [p for p in props if p.status == "proven"]
            types_present = {p.property_type for p in props}
            missing = [t for t in cls.REQUIRED_TYPES if t not in types_present]

            if not props or len(proven) < len(props) or missing:
                gaps.append(
                    PropertyGap(
                        module=mod,
                        total_properties=len(props),
                        proven_properties=len(proven),
                        missing_types=missing,
                    )
                )
        return gaps

    @classmethod
    def is_fully_covered(
        cls, properties: List[FormalProperty], required_modules: List[str]
    ) -> bool:
        """Return whether every required module passes the configured screen."""
        return len(cls.detect(properties, required_modules)) == 0


__all__ = [
    "FormalProperty",
    "FormalProofCertificate",
    "ProofTestCoverage",
    "PropertyGap",
    "FormalPropertyGapDetector",
]
