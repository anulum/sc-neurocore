#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li

"""Build and validate a deterministic offline release security artifact index."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

RELEASE_ARTIFACT_INDEX_SCHEMA_VERSION = "sc-neurocore.release-security-artifact-index.v1"


def build_parser() -> argparse.ArgumentParser:
    return argparse.ArgumentParser(
        description="Build a deterministic release security artifact index."
    )


def _resolve_artifact_root_path(root: Path, artifact_path: str) -> Path:
    candidate = Path(artifact_path)
    if candidate.is_absolute():
        return candidate
    return root / candidate


def validate_artifact_manifest(payload: dict[str, Any]) -> dict[str, Any]:
    findings: list[dict[str, str]] = []

    if payload.get("schema_version") != RELEASE_ARTIFACT_INDEX_SCHEMA_VERSION:
        findings.append(
            {
                "level": "error",
                "message": "schema_version is missing or not equal to expected version",
            }
        )

    artifacts_raw = payload.get("artifacts")
    if not isinstance(artifacts_raw, list):
        findings.append({"level": "error", "message": "artifacts must be a list"})
        return _final_report(findings)

    required_keys = {"id", "path", "required"}
    artifact_ids: set[str] = set()

    for artifact in artifacts_raw:
        if not isinstance(artifact, dict):
            findings.append({"level": "error", "message": "artifact entry must be an object"})
            continue

        artifact_id = artifact.get("id")
        if not isinstance(artifact_id, str) or not artifact_id:
            findings.append({"level": "error", "message": "artifact.id must be a non-empty string"})
            continue

        missing = sorted(required_keys - set(artifact))
        if missing:
            findings.append(
                {
                    "level": "error",
                    "message": f"artifact {artifact_id} missing fields: {', '.join(missing)}",
                }
            )

        artifact_path = artifact.get("path")
        if not isinstance(artifact_path, str) or not artifact_path:
            findings.append(
                {
                    "level": "error",
                    "message": f"artifact {artifact_id} path must be a non-empty string",
                }
            )

        if not isinstance(artifact.get("required"), bool):
            findings.append(
                {
                    "level": "error",
                    "message": f"artifact {artifact_id} required must be true/false",
                }
            )

        if artifact_id in artifact_ids:
            findings.append(
                {
                    "level": "error",
                    "message": f"duplicate artifact id {artifact_id}",
                }
            )
        artifact_ids.add(artifact_id)

    return _final_report(findings)


def _final_report(findings: list[dict[str, str]]) -> dict[str, Any]:
    errors = [finding for finding in findings if finding["level"] == "error"]
    return {
        "schema_version": RELEASE_ARTIFACT_INDEX_SCHEMA_VERSION,
        "passed": len(errors) == 0,
        "findings": findings,
        "errors": len(errors),
        "warnings": len([finding for finding in findings if finding["level"] == "warning"]),
    }


def build_artifact_index(
    manifest_payload: dict[str, Any],
    *,
    root: Path,
) -> dict[str, Any]:
    if not isinstance(root, Path):
        root = Path(root)

    report = validate_artifact_manifest(manifest_payload)
    if not report["passed"]:
        raise ValueError("invalid release artifact manifest")

    artifacts = manifest_payload["artifacts"]
    if not isinstance(artifacts, list):
        raise ValueError("invalid release artifact manifest")

    def _artifact_entry(raw_artifact: dict[str, Any]) -> dict[str, Any]:
        artifact_id = str(raw_artifact["id"])
        artifact_path = str(raw_artifact["path"])
        required = bool(raw_artifact["required"])
        normalized_path = Path(artifact_path)
        present = _resolve_artifact_root_path(root, artifact_path).exists()

        return {
            "id": artifact_id,
            "path": str(normalized_path),
            "required": required,
            "present": present,
        }

    built_artifacts = [
        _artifact_entry(artifact) for artifact in sorted(artifacts, key=lambda item: item["id"])
    ]
    missing_required = [
        artifact["id"]
        for artifact in built_artifacts
        if artifact["required"] and not artifact["present"]
    ]
    missing_optional = [
        artifact["id"]
        for artifact in built_artifacts
        if not artifact["required"] and not artifact["present"]
    ]

    return {
        "schema_version": RELEASE_ARTIFACT_INDEX_SCHEMA_VERSION,
        "required_count": sum(int(item["required"]) for item in built_artifacts),
        "optional_count": sum(int(not item["required"]) for item in built_artifacts),
        "missing_required": missing_required,
        "missing_optional": missing_optional,
        "artifacts": built_artifacts,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="Path to release artifact manifest JSON.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Root directory used when resolving artifact paths.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output path for the generated artifact index JSON.",
    )
    parser.add_argument(
        "--fail-on-missing-required",
        action="store_true",
        help=("Return non-zero if any required artifact is missing from the given root."),
    )
    args = parser.parse_args(argv)

    try:
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"failed to load manifest {args.manifest}: {exc}", file=sys.stderr)
        return 1

    manifest_report = validate_artifact_manifest(manifest)
    if not manifest_report["passed"]:
        print(json.dumps(manifest_report, indent=2, sort_keys=True))
        return 1

    artifact_index = build_artifact_index(manifest, root=args.root)
    output_payload = json.dumps(artifact_index, indent=2, sort_keys=True)
    if args.output is None:
        print(output_payload)
    else:
        try:
            args.output.write_text(output_payload + "\n", encoding="utf-8")
        except OSError as exc:
            print(f"failed to write artifact index to {args.output}: {exc}", file=sys.stderr)
            return 1

    if args.fail_on_missing_required and artifact_index["missing_required"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
