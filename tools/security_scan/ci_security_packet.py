#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — CI security packet builder

"""Build an offline CI security packet without executing scanner binaries."""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
from pathlib import Path
from typing import Any, cast

CI_SECURITY_PACKET_SCHEMA_VERSION = "sc-neurocore.ci-security-packet.v1"
RELEASE_SECURITY_ARTIFACT_INDEX_SCHEMA_VERSION = "sc-neurocore.release-security-artifact-index.v1"


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _script_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_module(module_name: str, module_path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load module {module_name} from {module_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build an offline security packet with deterministic planner outputs."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to write packet artifacts into.",
    )
    parser.add_argument(
        "--include-heavy",
        action="store_true",
        help="Include heavy scanners in Rust/supply-chain plan checks.",
    )
    parser.add_argument(
        "--fail-on-missing-required",
        action="store_true",
        help=("Return non-zero when required release artifacts are missing from packet root."),
    )
    return parser


def build_scanner_manifest() -> dict[str, Any]:
    module = _load_module(
        "security_scanner_manifest_for_packet",
        _script_root() / "tools" / "security_scanner_manifest.py",
    )
    return cast(dict[str, Any], module.build_scanner_manifest())


def build_scanner_plan(
    *, repo_root: Path, manifest_payload: dict[str, Any] | None = None
) -> dict[str, Any]:
    module = _load_module(
        "python_code_scanner_plan_for_packet",
        _script_root() / "tools" / "security_scan" / "python_code_scanner_plan.py",
    )
    if manifest_payload is None:
        return cast(dict[str, Any], module.build_scanner_plan(repo_root=repo_root))
    return cast(
        dict[str, Any],
        module.build_scanner_plan(repo_root=repo_root, manifest_payload=manifest_payload),
    )


def build_rust_supply_chain_plan(
    manifest_payload: dict[str, Any], *, repo_root: Path, include_heavy: bool
) -> dict[str, Any]:
    module = _load_module(
        "rust_supply_chain_scanner_plan_for_packet",
        _script_root() / "tools" / "security_scan" / "rust_supply_chain_scanner_plan.py",
    )
    return cast(
        dict[str, Any],
        module.build_rust_supply_chain_plan(
            manifest_payload,
            repo_root=repo_root,
            include_heavy=include_heavy,
        ),
    )


def build_artifact_index(manifest_payload: dict[str, Any], *, root: Path) -> dict[str, Any]:
    module = _load_module(
        "release_security_artifact_index_for_packet",
        _script_root() / "tools" / "security_scan" / "release_security_artifact_index.py",
    )
    return cast(dict[str, Any], module.build_artifact_index(manifest_payload, root=root))


def _load_release_manifest(root: Path) -> dict[str, Any]:
    manifest_path = root / "security" / "release_artifacts_manifest.json"
    return cast(dict[str, Any], json.loads(manifest_path.read_text(encoding="utf-8")))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _copy_matrix_to_packet(root: Path, source_root: Path) -> None:
    source = source_root / "security" / "model_data_license_matrix.json"
    if not source.exists():
        raise FileNotFoundError(f"missing model data license matrix source: {source}")

    target = root / "model_data_license_matrix.json"
    security_target = root / "security" / "model_data_license_matrix.json"
    security_target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    shutil.copy2(source, security_target)


def _build_artifact_paths(
    artifact_index_payload: dict[str, Any], *, output_dir: Path
) -> list[dict[str, str]]:
    artifacts = artifact_index_payload.get("artifacts", [])
    if not isinstance(artifacts, list):
        return []

    paths: list[dict[str, str]] = []
    for artifact in artifacts:
        if not isinstance(artifact, dict):
            continue
        artifact_id = artifact.get("id")
        artifact_path = artifact.get("path")
        if not isinstance(artifact_id, str) or not isinstance(artifact_path, str):
            continue
        artifact_path_obj = Path(artifact_path)
        if not artifact_path_obj.is_absolute():
            artifact_path_obj = output_dir / artifact_path_obj
        paths.append(
            {
                "id": artifact_id,
                "path": str(artifact_path_obj),
            }
        )

    return sorted(paths, key=lambda entry: (entry["id"], entry["path"]))


def _build_summary(output_dir: Path, artifact_index_payload: dict[str, Any]) -> dict[str, Any]:
    python_plan = _load_packet_plan(output_dir / "python_code_scanner_plan.json")
    rust_plan = _load_packet_plan(output_dir / "rust_supply_chain_scanner_plan.json")
    return {
        "schema_version": CI_SECURITY_PACKET_SCHEMA_VERSION,
        "output_dir": str(output_dir.resolve()),
        "artifact_paths": _build_artifact_paths(artifact_index_payload, output_dir=output_dir),
        "missing_required": artifact_index_payload.get("missing_required", []),
        "missing_optional": artifact_index_payload.get("missing_optional", []),
        "missing_required_vulnerability_status": artifact_index_payload.get(
            "missing_required_vulnerability_status", []
        ),
        "missing_optional_vulnerability_status": artifact_index_payload.get(
            "missing_optional_vulnerability_status", []
        ),
        "missing_required_scanner_inputs": _missing_required_scanner_inputs(
            {
                "python_code_scanner_plan": python_plan,
                "rust_supply_chain_scanner_plan": rust_plan,
            }
        ),
        "vulnerability_summary": artifact_index_payload.get("vulnerability_summary", {}),
    }


def _load_packet_plan(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return {}
    if isinstance(payload, dict):
        return payload
    return {}


def _missing_required_scanner_inputs(plans: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    missing: list[dict[str, Any]] = []
    for plan_name, plan in plans.items():
        scanners = plan.get("scanners", [])
        if not isinstance(scanners, list):
            continue
        for scanner in scanners:
            if not isinstance(scanner, dict):
                continue
            if scanner.get("run_class") != "missing_required_input":
                continue
            scanner_name = scanner.get("name")
            missing_inputs = scanner.get("missing_required_inputs")
            if not isinstance(scanner_name, str) or not isinstance(missing_inputs, list):
                continue
            inputs = sorted(item for item in missing_inputs if isinstance(item, str))
            if not inputs:
                continue
            missing.append(
                {
                    "inputs": inputs,
                    "plan": plan_name,
                    "scanner": scanner_name,
                }
            )
    return sorted(missing, key=lambda item: (item["plan"], item["scanner"], item["inputs"]))


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    security_output_dir = output_dir / "security"
    security_output_dir.mkdir(parents=True, exist_ok=True)

    try:
        scanner_manifest = build_scanner_manifest()
        _write_json(output_dir / "security_scanner_manifest.json", scanner_manifest)
        _write_json(security_output_dir / "security_scanner_manifest.json", scanner_manifest)

        python_plan = build_scanner_plan(
            repo_root=_project_root(),
            manifest_payload=scanner_manifest,
        )
        _write_json(output_dir / "python_code_scanner_plan.json", python_plan)

        rust_plan = build_rust_supply_chain_plan(
            scanner_manifest,
            repo_root=_project_root(),
            include_heavy=args.include_heavy,
        )
        _write_json(output_dir / "rust_supply_chain_scanner_plan.json", rust_plan)

        release_manifest = _load_release_manifest(_project_root())
        _copy_matrix_to_packet(output_dir, _project_root())
        artifact_index = build_artifact_index(release_manifest, root=output_dir)
        _write_json(output_dir / "release_security_artifact_index.json", artifact_index)

    except (RuntimeError, FileNotFoundError, OSError, json.JSONDecodeError, KeyError) as exc:
        print(f"failed to build CI security packet: {exc}", file=sys.stderr)
        return 1

    summary = _build_summary(output_dir=output_dir, artifact_index_payload=artifact_index)
    print(json.dumps(summary, indent=2, sort_keys=True))

    if args.fail_on_missing_required and (
        summary["missing_required"]
        or summary["missing_required_vulnerability_status"]
        or summary["missing_required_scanner_inputs"]
    ):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
