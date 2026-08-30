#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — benchmark evidence gate

"""Fail-closed benchmark evidence gate for committed JSON artefacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

BENCHMARK_EVIDENCE_GATE_SCHEMA_VERSION = "sc-neurocore.benchmark-evidence-gate.v1"


@dataclass(frozen=True)
class GateFailure:
    """One benchmark evidence gate failure."""

    gate_id: str
    reason: str
    path: str

    def to_json(self) -> dict[str, str]:
        return {"gate_id": self.gate_id, "reason": self.reason, "path": self.path}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("benchmarks/benchmark_regression_gates.json"),
        help="JSON manifest describing required benchmark artefact gates.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/results/benchmark_evidence_gate_report.json"),
        help="Path for the machine-readable gate report.",
    )
    return parser


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _path_value(payload: Any, dotted_path: str) -> Any:
    current = payload
    remaining = dotted_path
    while isinstance(current, dict):
        if remaining in current:
            return current[remaining]
        part, separator, remaining = remaining.partition(".")
        if not separator or part not in current:
            break
        current = current[part]
    raise KeyError(dotted_path)


def _is_finite_number(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fail(failures: list[GateFailure], gate_id: str, reason: str, path: str) -> None:
    failures.append(GateFailure(gate_id=gate_id, reason=reason, path=path))


def _check_required_numbers(
    *,
    gate_id: str,
    payload: Any,
    required_numbers: list[str],
    failures: list[GateFailure],
) -> None:
    for metric_path in required_numbers:
        try:
            value = _path_value(payload, metric_path)
        except KeyError:
            _fail(failures, gate_id, "missing_required_numeric_metric", metric_path)
            continue
        if not _is_finite_number(value):
            _fail(failures, gate_id, "required_metric_is_not_finite_number", metric_path)


def _check_expected_values(
    *,
    gate_id: str,
    payload: Any,
    expected_values: dict[str, Any],
    failures: list[GateFailure],
) -> None:
    for metric_path, expected in sorted(expected_values.items()):
        try:
            value = _path_value(payload, metric_path)
        except KeyError:
            _fail(failures, gate_id, "missing_expected_value", metric_path)
            continue
        if value != expected:
            _fail(failures, gate_id, f"expected_{expected!r}_found_{value!r}", metric_path)


def _check_source_hashes(
    *,
    repo_root: Path,
    gate_id: str,
    payload: Any,
    source_hashes: dict[str, str],
    failures: list[GateFailure],
) -> None:
    for source_path, json_path in sorted(source_hashes.items()):
        source = repo_root / source_path
        if not source.is_file():
            _fail(failures, gate_id, "missing_source_for_hash_check", source_path)
            continue
        try:
            recorded = _path_value(payload, json_path)
        except KeyError:
            _fail(failures, gate_id, "missing_recorded_source_hash", json_path)
            continue
        actual = _sha256(source)
        if recorded != actual:
            _fail(failures, gate_id, "source_hash_mismatch", f"{source_path}->{json_path}")


def _check_regression_limits(
    *,
    gate_id: str,
    payload: Any,
    regression_limits: dict[str, dict[str, float]],
    failures: list[GateFailure],
) -> None:
    for metric_path, limits in sorted(regression_limits.items()):
        try:
            value = _path_value(payload, metric_path)
        except KeyError:
            _fail(failures, gate_id, "missing_regression_metric", metric_path)
            continue
        if not _is_finite_number(value):
            _fail(failures, gate_id, "regression_metric_is_not_finite_number", metric_path)
            continue
        maximum = limits.get("max")
        minimum = limits.get("min")
        if maximum is not None and value > maximum:
            _fail(failures, gate_id, f"metric_above_max_{maximum}", metric_path)
        if minimum is not None and value < minimum:
            _fail(failures, gate_id, f"metric_below_min_{minimum}", metric_path)


def _check_parity_groups(
    *,
    gate_id: str,
    payload: Any,
    parity_groups: list[dict[str, Any]],
    failures: list[GateFailure],
) -> None:
    for index, group in enumerate(parity_groups):
        paths = group.get("paths")
        if not isinstance(paths, list) or len(paths) < 2:
            _fail(
                failures,
                gate_id,
                "parity_group_requires_at_least_two_paths",
                f"parity_groups.{index}.paths",
            )
            continue
        tolerance = group.get("tolerance", 0.0)
        if not _is_finite_number(tolerance) or tolerance < 0.0:
            _fail(
                failures,
                gate_id,
                "parity_group_tolerance_is_not_finite_non_negative",
                f"parity_groups.{index}.tolerance",
            )
            continue

        values: list[float] = []
        for metric_path in paths:
            if not isinstance(metric_path, str):
                _fail(
                    failures,
                    gate_id,
                    "parity_metric_path_is_not_string",
                    f"parity_groups.{index}.paths",
                )
                continue
            try:
                value = _path_value(payload, metric_path)
            except KeyError:
                _fail(failures, gate_id, "missing_parity_metric", metric_path)
                continue
            if not _is_finite_number(value):
                _fail(failures, gate_id, "parity_metric_is_not_finite_number", metric_path)
                continue
            values.append(float(value))
        if len(values) != len(paths):
            continue
        if max(values) - min(values) > float(tolerance):
            _fail(failures, gate_id, "parity_group_mismatch", f"parity_groups.{index}")


def _check_manifest_contract(manifest: Any, failures: list[GateFailure]) -> list[Any]:
    if not isinstance(manifest, dict):
        _fail(failures, "manifest", "manifest_root_is_not_object", "manifest")
        return []
    if manifest.get("SPDX-License-Identifier") != "AGPL-3.0-or-later":
        _fail(failures, "manifest", "manifest_missing_spdx_marker", "SPDX-License-Identifier")
    if manifest.get("schema_version") != "sc-neurocore.benchmark-regression-gates.v1":
        _fail(failures, "manifest", "manifest_schema_version_mismatch", "schema_version")
    gates = manifest.get("gates", [])
    if not isinstance(gates, list) or not gates:
        _fail(failures, "manifest", "manifest_has_no_gates", "gates")
        return []
    return gates


def _check_gate_manifest_entry(
    *,
    gate: dict[str, Any],
    gate_id: str,
    artefact_path: str,
    seen_ids: set[str],
    seen_artefacts: set[str],
    failures: list[GateFailure],
) -> None:
    if gate_id in seen_ids:
        _fail(failures, gate_id, "duplicate_gate_id", "id")
    seen_ids.add(gate_id)
    if artefact_path in seen_artefacts:
        _fail(failures, gate_id, "duplicate_gate_artefact", artefact_path)
    seen_artefacts.add(artefact_path)
    if not gate.get("required_numbers") and not gate.get("expected_values"):
        _fail(failures, gate_id, "gate_has_no_required_metrics_or_contracts", artefact_path)
    for field in (
        "required_numbers",
        "expected_values",
        "source_hashes",
        "regression_limits",
        "parity_groups",
    ):
        if field not in gate:
            continue
        value = gate[field]
        expected_type = list if field in {"required_numbers", "parity_groups"} else dict
        if not isinstance(value, expected_type):
            _fail(failures, gate_id, f"{field}_has_wrong_type", field)


def evaluate_benchmark_evidence_gate(
    *,
    manifest_path: Path,
    output_path: Path,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    root = repo_root or Path.cwd()
    failures: list[GateFailure] = []
    try:
        manifest = _load_json(manifest_path)
    except (json.JSONDecodeError, OSError):
        _fail(failures, "manifest", "manifest_is_missing_or_invalid_json", str(manifest_path))
        report = {
            "SPDX-License-Identifier": "AGPL-3.0-or-later",
            "schema_version": BENCHMARK_EVIDENCE_GATE_SCHEMA_VERSION,
            "manifest": str(manifest_path),
            "gate_count": 0,
            "evaluated_gates": [],
            "failure_count": len(failures),
            "failures": [failure.to_json() for failure in failures],
            "passed": False,
        }
        _write_json(output_path, report)
        return report

    gates = _check_manifest_contract(manifest, failures)

    evaluated: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    seen_artefacts: set[str] = set()
    for gate in gates:
        if not isinstance(gate, dict):
            _fail(failures, "manifest", "gate_entry_is_not_object", str(manifest_path))
            continue
        gate_id = str(gate.get("id", "unnamed"))
        artefact_path = str(gate.get("artefact", ""))
        if not artefact_path:
            _fail(failures, gate_id, "gate_missing_artefact_path", str(manifest_path))
            continue
        _check_gate_manifest_entry(
            gate=gate,
            gate_id=gate_id,
            artefact_path=artefact_path,
            seen_ids=seen_ids,
            seen_artefacts=seen_artefacts,
            failures=failures,
        )

        artefact = root / artefact_path
        evaluated.append({"id": gate_id, "artefact": artefact_path})
        if not artefact.is_file():
            _fail(failures, gate_id, "missing_benchmark_artefact", artefact_path)
            continue

        try:
            payload = _load_json(artefact)
        except (json.JSONDecodeError, OSError):
            _fail(failures, gate_id, "artefact_is_not_valid_json", artefact_path)
            continue
        if not isinstance(payload, dict):
            _fail(failures, gate_id, "artefact_root_is_not_object", artefact_path)
            continue

        _check_required_numbers(
            gate_id=gate_id,
            payload=payload,
            required_numbers=list(gate.get("required_numbers", [])),
            failures=failures,
        )
        _check_expected_values(
            gate_id=gate_id,
            payload=payload,
            expected_values=dict(gate.get("expected_values", {})),
            failures=failures,
        )
        _check_source_hashes(
            repo_root=root,
            gate_id=gate_id,
            payload=payload,
            source_hashes=dict(gate.get("source_hashes", {})),
            failures=failures,
        )
        _check_regression_limits(
            gate_id=gate_id,
            payload=payload,
            regression_limits=dict(gate.get("regression_limits", {})),
            failures=failures,
        )
        _check_parity_groups(
            gate_id=gate_id,
            payload=payload,
            parity_groups=list(gate.get("parity_groups", [])),
            failures=failures,
        )

    report = {
        "SPDX-License-Identifier": "AGPL-3.0-or-later",
        "schema_version": BENCHMARK_EVIDENCE_GATE_SCHEMA_VERSION,
        "manifest": str(manifest_path),
        "gate_count": len(gates),
        "evaluated_gates": evaluated,
        "failure_count": len(failures),
        "failures": [failure.to_json() for failure in failures],
        "passed": not failures,
    }
    _write_json(output_path, report)
    return report


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = evaluate_benchmark_evidence_gate(
        manifest_path=args.manifest,
        output_path=args.output,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
