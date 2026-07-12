# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — safety-evidence package assembly

"""Assemble explicit safety evidence into reproducible local packages.

This module organises caller-supplied evidence. It does not certify a product,
replace licensed standards, or make a conformity-assessment decision.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import tempfile
import textwrap
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path

from sc_neurocore.safety_cert.compliance import ChecklistItem, ComplianceChecklist
from sc_neurocore.safety_cert.failure_analysis import FMEDA, FailureMode
from sc_neurocore.safety_cert.formal_evidence import FormalProofCertificate, FormalProperty
from sc_neurocore.safety_cert.standards import (
    ASILLevel,
    SIL_TO_ASIL,
    SILLevel,
    SafetyStandard,
)
from sc_neurocore.safety_cert.timing_analysis import WCETAnalyzer
from sc_neurocore.safety_cert.traceability import Requirement, TraceabilityMatrix

_PACKAGE_KIND = "sc-neurocore.safety-evidence-package"
_PACKAGE_SCHEMA_VERSION = 1


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalise_generated_at(value: str | None) -> str:
    timestamp = _utc_now_iso() if value is None else value
    if not isinstance(timestamp, str) or not timestamp.strip():
        raise ValueError("generated_at must be a non-empty ISO-8601 string when provided")
    try:
        parsed = datetime.fromisoformat(timestamp)
    except ValueError as exc:
        raise ValueError("generated_at must be a valid ISO-8601 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("generated_at must include a UTC offset")
    return timestamp


def _markdown_cell(value: str) -> str:
    return value.replace("\n", "<br>").replace("|", "\\|")


def _write_bytes(path: Path, payload: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    path.chmod(0o600)


@dataclass
class CertificationPackage:
    """In-memory safety-evidence reports ready for deterministic materialisation."""

    standard: SafetyStandard
    sil_level: SILLevel
    traceability_report: str
    fmeda_report: str
    formal_cert_report: str
    wcet_report: str
    checklist: list[ChecklistItem]
    package_hash: str = ""
    generated: str = ""

    def __post_init__(self) -> None:
        """Validate package metadata and every checklist row."""
        if not isinstance(self.standard, SafetyStandard):
            raise ValueError("standard must be a SafetyStandard")
        if not isinstance(self.sil_level, SILLevel):
            raise ValueError("sil_level must be a SILLevel")
        for field_name in (
            "traceability_report",
            "fmeda_report",
            "formal_cert_report",
            "wcet_report",
            "package_hash",
            "generated",
        ):
            if not isinstance(getattr(self, field_name), str):
                raise ValueError(f"{field_name} must be a string")
        if self.package_hash and (
            len(self.package_hash) != 32
            or any(character not in "0123456789abcdef" for character in self.package_hash)
        ):
            raise ValueError("package_hash must be a 32-character lowercase hexadecimal ID")
        self._validate_checklist()

    def _validate_checklist(self) -> None:
        if not isinstance(self.checklist, list):
            raise ValueError("checklist must be a list")
        for item in self.checklist:
            if not isinstance(item, ChecklistItem):
                raise ValueError("checklist must contain ChecklistItem entries")
            if not isinstance(item.item_id, str) or not item.item_id.strip():
                raise ValueError("checklist item_id values must be non-empty strings")
            if not isinstance(item.clause, str) or not item.clause.strip():
                raise ValueError("checklist clauses must be non-empty strings")
            if not isinstance(item.status, str) or item.status not in {
                "compliant",
                "partial",
                "not_addressed",
            }:
                raise ValueError(
                    "checklist statuses must be one of: compliant, partial, not_addressed"
                )
            if item.status != "not_addressed" and not item.evidence.strip():
                raise ValueError("addressed checklist items require evidence")

    @property
    def checklist_coverage(self) -> float:
        """Return the fraction with explicit evidence, including partial rows."""
        self._validate_checklist()
        if not self.checklist:
            return 0.0
        addressed = sum(item.status != "not_addressed" for item in self.checklist)
        return addressed / len(self.checklist)

    def checklist_report(self) -> str:
        """Render checklist state without upgrading any evidence status."""
        self._validate_checklist()
        lines = [
            "# Compliance Checklist",
            "",
            "Status is evidence bookkeeping only; it is not a conformity decision.",
            "",
            "| Item | Clause | Description | Evidence | Status |",
            "|------|--------|-------------|----------|--------|",
        ]
        for item in self.checklist:
            lines.append(
                "| "
                + " | ".join(
                    _markdown_cell(value)
                    for value in (
                        item.item_id,
                        item.clause,
                        item.description,
                        item.evidence,
                        item.status,
                    )
                )
                + " |"
            )
        return "\n".join(lines)

    def artifacts(self) -> dict[str, str]:
        """Return the five deterministic report filenames and their contents."""
        return {
            "traceability_matrix.md": self.traceability_report,
            "fmeda_report.md": self.fmeda_report,
            "formal_proof_cert.md": self.formal_cert_report,
            "wcet_analysis.md": self.wcet_report,
            "compliance_checklist.md": self.checklist_report(),
        }

    def content_sha256(self) -> str:
        """Return a full digest over metadata and all report contents."""
        payload = {
            "artifacts": self.artifacts(),
            "generated": self.generated,
            "kind": _PACKAGE_KIND,
            "schema_version": _PACKAGE_SCHEMA_VERSION,
            "standard": self.standard.value,
            "target_sil": self.sil_level.value,
        }
        encoded = json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def write(self, directory: str | Path) -> Path:
        """Atomically materialise reports and a hash manifest in a new directory.

        Existing destinations are never overwritten. Files are written in a
        private sibling directory, flushed, and renamed only after every digest
        has been computed.
        """
        if not isinstance(directory, str | Path):
            raise ValueError("directory must be a string or Path")
        if isinstance(directory, str) and (not directory.strip() or "\x00" in directory):
            raise ValueError("directory must be a non-empty path without NUL bytes")
        destination = Path(directory)
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(f"destination already exists: {destination}")
        if not self.generated or not self.package_hash:
            raise ValueError("generated and package_hash must be populated before writing")
        content_digest = self.content_sha256()
        if self.package_hash != content_digest[:32]:
            raise ValueError("package_hash does not match the current package contents")
        artifacts = self.artifacts()
        for filename, content in artifacts.items():
            if not content:
                raise ValueError(f"artifact {filename} must not be empty")

        parent = destination.parent
        parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=parent))
        temporary.chmod(0o700)
        try:
            manifest_artifacts: list[dict[str, str | int]] = []
            for filename, content in artifacts.items():
                encoded = content.encode("utf-8")
                _write_bytes(temporary / filename, encoded)
                manifest_artifacts.append(
                    {
                        "bytes": len(encoded),
                        "filename": filename,
                        "sha256": hashlib.sha256(encoded).hexdigest(),
                    }
                )
            manifest = {
                "artifacts": manifest_artifacts,
                "content_sha256": content_digest,
                "generated": self.generated,
                "kind": _PACKAGE_KIND,
                "package_id": self.package_hash,
                "schema_version": _PACKAGE_SCHEMA_VERSION,
                "standard": self.standard.value,
                "target_sil": self.sil_level.value,
            }
            manifest_bytes = (
                json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
            ).encode("utf-8")
            _write_bytes(temporary / "manifest.json", manifest_bytes)
            if destination.exists() or destination.is_symlink():
                raise FileExistsError(f"destination already exists: {destination}")
            os.replace(temporary, destination)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
        return destination


class CertificationGenerator:
    """Assemble caller-supplied records without fabricating assurance evidence."""

    def generate(
        self,
        standard: SafetyStandard,
        target_sil: SILLevel,
        modules: list[str],
        formal_properties: list[FormalProperty],
        network_config: Mapping[str, int | float] | None = None,
        *,
        implementation_evidence: Mapping[str, Sequence[str]] | None = None,
        failure_modes: Sequence[FailureMode] | None = None,
        checklist_evidence: Mapping[str, str] | None = None,
        generated_at: str | None = None,
    ) -> CertificationPackage:
        """Build one evidence package from explicit records and assumptions."""
        if not isinstance(standard, SafetyStandard):
            raise ValueError("standard must be a SafetyStandard")
        if not isinstance(target_sil, SILLevel):
            raise ValueError("target_sil must be a SILLevel")
        if not isinstance(modules, list) or not modules:
            raise ValueError("modules must be a non-empty list")
        for module in modules:
            if not isinstance(module, str) or not module.strip():
                raise ValueError("modules must contain non-empty strings")
            if module != module.strip():
                raise ValueError("modules must not contain leading or trailing whitespace")
        if len(modules) != len(set(modules)):
            raise ValueError("modules must not contain duplicates")
        self._validate_formal_properties(formal_properties)
        config = self._validate_network_config(network_config)
        implementations = self._validate_implementation_evidence(
            modules,
            implementation_evidence,
        )
        timestamp = _normalise_generated_at(generated_at)

        traceability = TraceabilityMatrix()
        for index, module in enumerate(modules, start=1):
            requirement = Requirement(
                req_id=f"REQ_{index:03d}",
                description=f"Safety function for {module}",
                standard=standard,
                sil_level=target_sil,
            )
            traceability.add_requirement(requirement)
            for reference in implementations.get(module, ()):
                traceability.link_implementation(requirement.req_id, reference)
            for prop in formal_properties:
                if prop.module == module and prop.status == "proven":
                    traceability.link_verification(
                        requirement.req_id,
                        prop.sby_file or prop.prop_id,
                    )

        fmeda = FMEDA()
        if failure_modes is not None:
            if isinstance(failure_modes, str) or not isinstance(failure_modes, Sequence):
                raise ValueError("failure_modes must be a sequence when provided")
            for failure_mode in failure_modes:
                if not isinstance(failure_mode, FailureMode):
                    raise ValueError("failure_modes must contain FailureMode entries")
                fmeda.add_failure_mode(failure_mode)
        if fmeda.failure_modes:
            fmeda_report = fmeda.generate_report()
        else:
            fmeda_report = (
                "# FMEDA Report\n\n"
                "Status: not assessed. No caller-supplied failure modes or FIT data were provided."
            )

        formal_certificate = FormalProofCertificate(properties=list(formal_properties))
        formal_certificate.compute_hash(generated_at=timestamp)

        if config is None:
            wcet_report = (
                "# WCET Analysis\n\n"
                "Status: not assessed. Supply network_config with measured or justified inputs."
            )
        else:
            wcet = WCETAnalyzer.analyze(
                int(config["bitstream_length"]),
                int(config["num_inputs"]),
                int(config["num_neurons"]),
            )
            clock_mhz = float(config["clock_mhz"])
            wcet_report = (
                "# WCET Analysis\n\n"
                f"Input-derived bound: {wcet.total_cycles} cycles = "
                f"{wcet.wcet_ns(clock_mhz):.1f} ns @ {clock_mhz:g} MHz\n\n"
                f"Stages: {' → '.join(wcet.stages)}"
            )

        checklist = ComplianceChecklist.generate(standard, evidence=checklist_evidence)
        package = CertificationPackage(
            standard=standard,
            sil_level=target_sil,
            traceability_report=traceability.generate_report(generated_at=timestamp),
            fmeda_report=fmeda_report,
            formal_cert_report=formal_certificate.generate_report(generated_at=timestamp),
            wcet_report=wcet_report,
            checklist=checklist,
            generated=timestamp,
        )
        package.package_hash = package.content_sha256()[:32]
        return package

    @staticmethod
    def _validate_formal_properties(properties: list[FormalProperty]) -> None:
        if not isinstance(properties, list):
            raise ValueError("formal_properties must be a list")
        for prop in properties:
            if not isinstance(prop, FormalProperty):
                raise ValueError("formal_properties must contain FormalProperty entries")
            if not isinstance(prop.prop_id, str) or not prop.prop_id.strip():
                raise ValueError("formal_properties prop_id values must be non-empty strings")
            if prop.prop_id != prop.prop_id.strip():
                raise ValueError(
                    "formal_properties prop_id values must not contain leading or trailing whitespace"
                )
            if not isinstance(prop.module, str) or not prop.module.strip():
                raise ValueError("formal_properties modules must be non-empty strings")
            if prop.module != prop.module.strip():
                raise ValueError(
                    "formal_properties modules must not contain leading or trailing whitespace"
                )
            if not isinstance(prop.status, str) or prop.status not in {
                "proven",
                "failed",
                "unknown",
            }:
                raise ValueError(
                    "formal_properties statuses must be one of: proven, failed, unknown"
                )
            if not isinstance(prop.property_type, str) or prop.property_type not in {
                "assert",
                "cover",
                "assume",
            }:
                raise ValueError(
                    "formal_properties property_type values must be one of: assert, cover, assume"
                )
        property_ids = [prop.prop_id for prop in properties]
        if len(property_ids) != len(set(property_ids)):
            raise ValueError("formal_properties must not contain duplicate prop_id values")

    @staticmethod
    def _validate_network_config(
        config: Mapping[str, int | float] | None,
    ) -> dict[str, int | float] | None:
        if config is None:
            return None
        if not isinstance(config, Mapping):
            raise ValueError("network_config must be a mapping when provided")
        allowed_keys = {"bitstream_length", "num_inputs", "num_neurons", "clock_mhz"}
        if set(config) != allowed_keys:
            raise ValueError(
                "network_config must contain exactly bitstream_length, num_inputs, "
                "num_neurons, and clock_mhz"
            )
        result = dict(config)
        for key in ("bitstream_length", "num_inputs", "num_neurons"):
            value = result[key]
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"network_config[{key}] must be an integer >= 1")
        clock = result["clock_mhz"]
        if isinstance(clock, bool) or not isinstance(clock, int | float):
            raise ValueError("network_config[clock_mhz] must be a finite positive number")
        if not math.isfinite(float(clock)) or float(clock) <= 0.0:
            raise ValueError("network_config[clock_mhz] must be a finite positive number")
        return result

    @staticmethod
    def _validate_implementation_evidence(
        modules: list[str],
        evidence: Mapping[str, Sequence[str]] | None,
    ) -> dict[str, tuple[str, ...]]:
        if evidence is None:
            return {}
        if not isinstance(evidence, Mapping):
            raise ValueError("implementation_evidence must be a mapping when provided")
        for module in evidence:
            if not isinstance(module, str) or not module.strip():
                raise ValueError("implementation_evidence keys must be non-empty strings")
        unknown_modules = sorted(set(evidence) - set(modules))
        if unknown_modules:
            raise ValueError("implementation_evidence contains unknown modules")
        result: dict[str, tuple[str, ...]] = {}
        for module, references in evidence.items():
            if isinstance(references, str) or not isinstance(references, Sequence):
                raise ValueError("implementation_evidence values must be sequences")
            normalised: list[str] = []
            for reference in references:
                if not isinstance(reference, str) or not reference.strip():
                    raise ValueError(
                        "implementation_evidence values must contain non-empty strings"
                    )
                normalised.append(reference.strip())
            if len(normalised) != len(set(normalised)):
                raise ValueError("implementation_evidence references must be unique")
            result[module] = tuple(normalised)
        return result


class SafetyManualGenerator:
    """Render a clearly labelled safety-manual template from explicit inputs."""

    @staticmethod
    def generate(
        product_name: str,
        sil_level: SILLevel,
        modules: list[str],
        wcet_ns: float,
        *,
        generated_on: str | None = None,
    ) -> str:
        """Generate a non-certifying manual template for later expert review."""
        if not isinstance(product_name, str) or not product_name.strip():
            raise ValueError("product_name must be a non-empty string")
        if not isinstance(sil_level, SILLevel):
            raise ValueError("sil_level must be a SILLevel")
        if not isinstance(modules, list) or not modules:
            raise ValueError("modules must be a non-empty list")
        for module in modules:
            if not isinstance(module, str) or not module.strip():
                raise ValueError("modules must contain non-empty strings")
            if module != module.strip():
                raise ValueError("modules must not contain leading or trailing whitespace")
        if len(modules) != len(set(modules)):
            raise ValueError("modules must not contain duplicates")
        if isinstance(wcet_ns, bool) or not isinstance(wcet_ns, int | float):
            raise ValueError("wcet_ns must be a finite non-negative value")
        if not math.isfinite(float(wcet_ns)) or float(wcet_ns) < 0.0:
            raise ValueError("wcet_ns must be a finite non-negative value")
        report_date = (
            datetime.now(timezone.utc).date().isoformat() if generated_on is None else generated_on
        )
        if not isinstance(report_date, str):
            raise ValueError("generated_on must be an ISO date string when provided")
        try:
            date.fromisoformat(report_date)
        except ValueError as exc:
            raise ValueError("generated_on must be a valid ISO date") from exc
        asil_label = SIL_TO_ASIL.get(sil_level, ASILLevel.QM).value
        module_lines = "\n            ".join(f"- {module}" for module in modules)
        return textwrap.dedent(
            f"""\
            # Safety Manual Template — {product_name}
            ## Target label: SIL {sil_level.value} — {report_date}

            > Draft evidence template only. It is not a certificate, approval,
            > regulatory submission, or substitute for a qualified assessor.

            ### 1. Declared Scope
            The caller supplied {len(modules)} safety-related module names:
            {module_lines}

            ### 2. Timing Input
            - Caller-supplied WCET value: {float(wcet_ns):.1f} ns
            - Measurement method, clock source, operating corners, and margin: TODO

            ### 3. Integrity Labels
            - Target label supplied by caller: SIL {sil_level.value}
            - Legacy crosswalk label: ASIL {asil_label}
            - The crosswalk is non-normative and does not establish equivalence.

            ### 4. Required Evidence
            - Safety requirements and implementation traceability: TODO
            - FMEDA source data and provenance: TODO
            - Verification reports and tool versions: TODO
            - Operating conditions and limitations: TODO
            - Change-control and re-verification process: TODO

            ### 5. Review
            Applicability, completeness, independence, and conformity decisions
            require review against licensed standards by qualified personnel.
            """
        )


__all__ = [
    "CertificationPackage",
    "CertificationGenerator",
    "SafetyManualGenerator",
]
