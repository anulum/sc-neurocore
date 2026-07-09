# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — License matrix validator

"""Validate legal/IP model-data license matrices.

The validator is fail-closed: missing required legal fields, missing
provenance, and contradictory commercial-use statements raise a
validation error with a complete issue list.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, cast

from collections import Counter

LICENSE_MATRIX_SCHEMA_VERSION = "sc-neurocore.license-matrix.v0.1"
PROVENANCE_HASH_TYPE_ARTIFACT = "artifact-sha256"
PROVENANCE_HASH_TYPE_POLICY = "policy-category-digest"
KNOWN_PROVENANCE_HASH_TYPES = frozenset(
    {PROVENANCE_HASH_TYPE_ARTIFACT, PROVENANCE_HASH_TYPE_POLICY}
)

CommercialUse = Literal["allowed", "requires_license", "prohibited"]
RedistributionStatus = Literal["allowed", "restricted", "prohibited", "not_specified"]
ArtifactType = Literal["pretrained_model", "model_weights", "dataset"]


class LicenseMatrixValidationError(ValueError):
    """Validation failure with a stable, deterministic issue list."""

    def __init__(self, issues: list[str], *, path: Path | None = None):
        super().__init__(", ".join(issues))
        self.issues = issues
        self.path = path


@dataclass(frozen=True, slots=True)
class LicenseMatrixProject:
    """Project level legal metadata."""

    license_identifier: str
    commercial_license_available: bool
    ownership_notice: str
    all_rights_reserved: bool
    commercial_use: CommercialUse

    def as_dict(self) -> dict[str, object]:
        """Return JSON-ready deterministic payload."""

        return {
            "license_identifier": self.license_identifier,
            "commercial_license_available": self.commercial_license_available,
            "ownership_notice": self.ownership_notice,
            "all_rights_reserved": self.all_rights_reserved,
            "commercial_use": self.commercial_use,
        }


@dataclass(frozen=True, slots=True)
class LicenseMatrixArtifact:
    """One legal row for a model, pretrained artefact, or dataset."""

    entry_id: str
    entry_type: ArtifactType
    license_identifier: str
    source_uri: str
    provenance: dict[str, object]
    redistribution_status: RedistributionStatus | None
    attribution_requirements: tuple[str, ...]
    commercial_use: CommercialUse
    commercial_license_required: bool

    def as_dict(self) -> dict[str, object]:
        """Return JSON-ready deterministic payload."""

        payload = asdict(self)
        payload["attribution_requirements"] = list(self.attribution_requirements)
        return payload


@dataclass(frozen=True, slots=True)
class LicenseMatrix:
    """Validated legal matrix."""

    project: LicenseMatrixProject
    entries: tuple[LicenseMatrixArtifact, ...]

    def as_report(self) -> dict[str, object]:
        """Build a deterministic summary and integrity report."""

        entries = [
            entry.as_dict() for entry in sorted(self.entries, key=lambda entry: entry.entry_id)
        ]
        by_type = Counter(entry["entry_type"] for entry in entries)
        commercial_require = sum(
            1
            for entry in entries
            if entry["commercial_license_required"] and entry["commercial_use"] != "allowed"
        )
        redistribution_status_count = Counter(entry["redistribution_status"] for entry in entries)
        payload = {
            "schema_version": LICENSE_MATRIX_SCHEMA_VERSION,
            "status": "valid",
            "project": self.project.as_dict(),
            "entry_count": len(entries),
            "artifact_types": dict(sorted((key, by_type[key]) for key in by_type)),
            "commercial_license_required_count": commercial_require,
            "redistribution_status": {
                "allowed": redistribution_status_count.get("allowed", 0),
                "restricted": redistribution_status_count.get("restricted", 0),
                "prohibited": redistribution_status_count.get("prohibited", 0),
                "not_specified": redistribution_status_count.get("not_specified", 0),
            },
            "entries": entries,
        }
        matrix_bytes = (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode(
            "utf-8"
        )
        payload["matrix_sha256"] = hashlib.sha256(matrix_bytes).hexdigest()
        return payload

    def as_dict(self) -> dict[str, object]:
        """Backward-compatible full deterministic payload."""

        return {
            "schema_version": LICENSE_MATRIX_SCHEMA_VERSION,
            "project": self.project.as_dict(),
            "entries": [entry.as_dict() for entry in self.entries],
        }


def _as_str(value: Any, *, field: str, issues: list[str]) -> str:
    if not isinstance(value, str) or not value.strip():
        issues.append(f"{field} must be a non-empty string")
        return ""
    return str(value)


def _as_bool(value: Any, *, field: str, issues: list[str]) -> bool:
    if not isinstance(value, bool):
        issues.append(f"{field} must be a boolean")
        return False
    return bool(value)


def _as_list_of_str(value: Any, *, field: str, issues: list[str]) -> tuple[str, ...]:
    if not isinstance(value, list):
        issues.append(f"{field} must be a list")
        return ()
    output: list[str] = []
    for idx, item in enumerate(value):
        if not isinstance(item, str) or not item.strip():
            issues.append(f"{field}[{idx}] must be a non-empty string")
        else:
            output.append(item)
    return tuple(output)


def _is_sha256_digest(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(char in "0123456789abcdefABCDEF" for char in value)
    )


def _normalise_provenance_hash_type(
    provenance: dict[str, object],
    entry_id: str,
    issues: list[str],
) -> str:
    raw_hash_type = provenance.get("hash_type")
    if raw_hash_type is None:
        return PROVENANCE_HASH_TYPE_ARTIFACT
    if not isinstance(raw_hash_type, str) or not raw_hash_type.strip():
        issues.append(f"entry[{entry_id}].provenance.hash_type must be a non-empty string")
        return PROVENANCE_HASH_TYPE_ARTIFACT
    if raw_hash_type == "sha256" or raw_hash_type == "hash":
        return PROVENANCE_HASH_TYPE_ARTIFACT
    if raw_hash_type not in KNOWN_PROVENANCE_HASH_TYPES:
        issues.append(
            f"entry[{entry_id}].provenance.hash_type '{raw_hash_type}' must be one of "
            "artifact-sha256, policy-category-digest"
        )
        return raw_hash_type
    return raw_hash_type


def _validate_project(payload: dict[str, Any], issues: list[str]) -> LicenseMatrixProject:
    license_identifier = _as_str(
        payload.get("license_identifier"), field="project.license_identifier", issues=issues
    )
    commercial_license_available = _as_bool(
        payload.get("commercial_license_available"),
        field="project.commercial_license_available",
        issues=issues,
    )
    ownership_notice = _as_str(
        payload.get("ownership_notice"), field="project.ownership_notice", issues=issues
    )
    all_rights_reserved = _as_bool(
        payload.get("all_rights_reserved"),
        field="project.all_rights_reserved",
        issues=issues,
    )
    commercial_use = payload.get("commercial_use")
    if commercial_use not in ("allowed", "requires_license", "prohibited"):
        issues.append("project.commercial_use must be one of allowed, requires_license, prohibited")
        commercial_use = "prohibited"

    if all_rights_reserved and "all rights reserved" not in ownership_notice.lower():
        issues.append(
            "project all_rights_reserved requires all-rights-reserved wording in ownership_notice"
        )
    if commercial_use == "prohibited" and commercial_license_available:
        issues.append(
            "project commercial_use='prohibited' conflicts with commercial_license_available=True"
        )
    if commercial_use == "requires_license" and not commercial_license_available:
        issues.append(
            "project commercial_use='requires_license' requires commercial_license_available=True"
        )

    return LicenseMatrixProject(
        license_identifier=license_identifier,
        commercial_license_available=commercial_license_available,
        ownership_notice=ownership_notice,
        all_rights_reserved=all_rights_reserved,
        commercial_use=cast(CommercialUse, commercial_use),
    )


def _validate_artifact(
    raw: dict[str, Any],
    project: LicenseMatrixProject,
    issues: list[str],
) -> LicenseMatrixArtifact | None:
    entry_id = _as_str(raw.get("entry_id"), field="entry.entry_id", issues=issues)
    entry_type = raw.get("entry_type")
    if entry_type not in ("pretrained_model", "model_weights", "dataset"):
        issues.append("entry_type must be pretrained_model, model_weights, or dataset")
        entry_type = "dataset"

    license_identifier = _as_str(
        raw.get("license_identifier"), field=f"entry[{entry_id}].license_identifier", issues=issues
    )
    source_uri = _as_str(
        raw.get("source_uri"), field=f"entry[{entry_id}].source_uri", issues=issues
    )
    provenance = raw.get("provenance")
    if not isinstance(provenance, dict) or not provenance:
        issues.append(f"entry[{entry_id}].provenance must be a non-empty mapping")
        provenance = {}
    else:
        provenance_source = provenance.get("source")
        if not isinstance(provenance_source, str) or not provenance_source.strip():
            issues.append(f"entry[{entry_id}].provenance.source must be a non-empty string")

        hash_type = _normalise_provenance_hash_type(provenance, entry_id, issues)
        digest = provenance.get("sha256", provenance.get("hash"))
        if not _is_sha256_digest(digest):
            issues.append(f"entry[{entry_id}].provenance must include valid sha256 or hash")
        elif hash_type == PROVENANCE_HASH_TYPE_POLICY:
            policy_category = provenance.get("category")
            if not isinstance(policy_category, str) or not policy_category.strip():
                issues.append(
                    f"entry[{entry_id}].provenance.category must be a non-empty string for "
                    "policy-category-digest provenance"
                )
    attribution_requirements = _as_list_of_str(
        raw.get("attribution_requirements"),
        field=f"entry[{entry_id}].attribution_requirements",
        issues=issues,
    )
    if not attribution_requirements:
        issues.append(f"entry[{entry_id}].attribution_requirements must contain at least one item")

    commercial_use = raw.get("commercial_use")
    if commercial_use not in ("allowed", "requires_license", "prohibited"):
        issues.append(
            f"entry[{entry_id}].commercial_use must be one of allowed, requires_license, prohibited"
        )
        commercial_use = "prohibited"
    commercial_license_required = _as_bool(
        raw.get("commercial_license_required"),
        field=f"entry[{entry_id}].commercial_license_required",
        issues=issues,
    )

    redistribution_status = raw.get("redistribution_status")
    if entry_type in ("pretrained_model", "model_weights"):
        if redistribution_status not in ("allowed", "restricted", "prohibited", "not_specified"):
            issues.append(f"entry[{entry_id}] {entry_type} requires redistribution_status")
            redistribution_status = "not_specified"
    elif redistribution_status is not None and redistribution_status not in (
        "allowed",
        "restricted",
        "prohibited",
        "not_specified",
    ):
        issues.append(
            f"entry[{entry_id}].redistribution_status must be allowed, restricted, prohibited, or not_specified"
        )

    if commercial_use == "requires_license" and not commercial_license_required:
        issues.append(
            f"entry[{entry_id}].commercial_use='requires_license' requires commercial_license_required=True"
        )
    if commercial_use != "requires_license" and commercial_license_required:
        issues.append(
            f"entry[{entry_id}] has commercial_license_required=True but commercial_use != 'requires_license'"
        )
    if commercial_use == "requires_license" and not project.commercial_license_available:
        issues.append(
            f"entry[{entry_id}].commercial_use='requires_license' but project disallows commercial licensing"
        )
    if project.commercial_use == "prohibited" and commercial_use != "prohibited":
        issues.append(
            f"entry[{entry_id}].commercial_use conflicts with project.commercial_use='prohibited'"
        )

    return LicenseMatrixArtifact(
        entry_id=entry_id,
        entry_type=cast(ArtifactType, entry_type),
        license_identifier=license_identifier,
        source_uri=source_uri,
        provenance=cast(dict[str, object], provenance),
        redistribution_status=cast(RedistributionStatus | None, redistribution_status),
        attribution_requirements=attribution_requirements,
        commercial_use=cast(CommercialUse, commercial_use),
        commercial_license_required=commercial_license_required,
    )


def parse_license_matrix(payload: dict[str, Any], *, path: Path | None = None) -> LicenseMatrix:
    """Parse, validate, and return a deterministic license matrix."""

    issues: list[str] = []

    project_raw = payload.get("project")
    if not isinstance(project_raw, dict):
        issues.append("project must be a mapping")
        raise LicenseMatrixValidationError(issues, path=path)

    project = _validate_project(project_raw, issues)
    entries_raw = payload.get("entries")
    if not isinstance(entries_raw, list) or not entries_raw:
        issues.append("entries must be a non-empty list")
        raise LicenseMatrixValidationError(issues, path=path)

    entries: list[LicenseMatrixArtifact] = []
    seen_ids: set[str] = set()
    for raw_entry in entries_raw:
        if not isinstance(raw_entry, dict):
            issues.append("each entry must be a mapping")
            continue
        entry = _validate_artifact(raw_entry, project, issues)
        if entry is None:
            continue
        if entry.entry_id in seen_ids:
            issues.append(f"duplicate entry_id: {entry.entry_id}")
            continue
        seen_ids.add(entry.entry_id)
        if (
            not entry.license_identifier
            or not entry.source_uri
            or not entry.attribution_requirements
        ):
            continue
        entries.append(entry)

    if not entries:
        issues.append("at least one valid artifact entry is required")

    if issues:
        raise LicenseMatrixValidationError(issues, path=path)

    return LicenseMatrix(project=project, entries=tuple(entries))


def validate_license_matrix_file(path: str | Path) -> LicenseMatrix:
    """Load and validate a matrix JSON file."""

    matrix_path = Path(path)
    try:
        payload = json.loads(matrix_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise LicenseMatrixValidationError([f"cannot read file: {exc}"], path=matrix_path) from exc
    except json.JSONDecodeError as exc:
        raise LicenseMatrixValidationError([f"invalid JSON: {exc}"], path=matrix_path) from exc

    if not isinstance(payload, dict):
        raise LicenseMatrixValidationError(["matrix file must be a JSON object"], path=matrix_path)
    return parse_license_matrix(payload, path=matrix_path)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("matrix_path", type=Path, help="Path to the matrix JSON file")
    parser.add_argument(
        "--output-json",
        type=Path,
        help="Optional path to write the deterministic validation report",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""

    args = _build_parser().parse_args(argv)
    try:
        matrix = validate_license_matrix_file(args.matrix_path)
    except LicenseMatrixValidationError as exc:
        print(f"License matrix invalid: {exc}")
        if exc.path is not None:
            print(f"  path: {exc.path}")
            print(f"  issues: {', '.join(exc.issues)}")
        return 1

    report = matrix.as_report()
    if args.output_json is not None:
        args.output_json.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(f"report written: {args.output_json}")

    print("License matrix valid")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
