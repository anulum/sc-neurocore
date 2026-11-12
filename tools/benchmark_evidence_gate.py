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
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

BENCHMARK_EVIDENCE_GATE_SCHEMA_VERSION = "sc-neurocore.benchmark-evidence-gate.v1"


@dataclass(frozen=True)
class GateFailure:
    """One benchmark evidence gate failure."""

    gate_id: str
    reason: str
    path: str

    def to_json(self) -> dict[str, str]:
        """Return the failure as the three string fields the report stores."""
        return {"gate_id": self.gate_id, "reason": self.reason, "path": self.path}


def build_parser() -> argparse.ArgumentParser:
    """Return the command-line parser with the manifest and report paths."""
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


def _git_bytes(repo_root: Path, *args: str) -> bytes | None:
    """Read a Git object or revision without changing repository state."""
    result = subprocess.run(
        ["git", "-C", str(repo_root), *args],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.stdout if result.returncode == 0 else None


def tracked_files(repo_root: Path, directory: str) -> tuple[str, ...] | None:
    """List the files Git tracks below a directory of a work tree.

    Parameters
    ----------
    repo_root : Path
        Directory that is expected to be the top of a Git work tree.
    directory : str
        Repository-relative directory whose tracked files are wanted.

    Returns
    -------
    tuple of str or None
        Repository-relative paths in Git's order, including files staged and not
        yet committed. ``None`` when ``repo_root`` is not the top of a work tree
        (a packaged or temporary tree, or a subdirectory of a repository): Git
        cannot say there which files are evidence, and the caller walks the
        directory instead. ``None`` also when no ``git`` program can be started.
        A name that is not UTF-8 is given as the file system decoding gives it,
        which is the form a directory walk yields for the same file.
    """
    try:
        top = _git_bytes(repo_root, "rev-parse", "--show-toplevel")
        if top is None:
            return None
        if Path(os.fsdecode(top.removesuffix(b"\n"))).resolve() != repo_root.resolve():
            return None
        listed = _git_bytes(repo_root, "ls-files", "-z", "--", directory)
    except OSError:
        return None
    if listed is None:
        return None
    return tuple(name for name in os.fsdecode(listed).split("\0") if name)


def _safe_repo_path(path: str) -> bool:
    """Restrict provenance lookups to ordinary repository-relative paths."""
    return (
        bool(path)
        and not PurePosixPath(path).is_absolute()
        and not any(part in {"", ".", ".."} for part in path.split("/"))
        and not any(char in path for char in ("\\", ":", "\x00"))
    )


def _artefact_commit(
    repo_root: Path,
    artefact_path: str,
    gate_id: str,
    failures: list[GateFailure],
) -> str | None:
    """Find the committed source snapshot and reject modified result bytes."""
    if not _safe_repo_path(artefact_path):
        _fail(failures, gate_id, "invalid_artefact_path", artefact_path)
        return None
    if _git_bytes(repo_root, "rev-parse", "--is-shallow-repository") == b"true\n":
        _fail(failures, gate_id, "history_is_shallow", artefact_path)
        return None
    raw_revision = _git_bytes(repo_root, "log", "-1", "--format=%H", "--", artefact_path)
    revision = raw_revision.decode("ascii", errors="ignore").strip() if raw_revision else ""
    if len(revision) not in {40, 64} or not all(char in "0123456789abcdef" for char in revision):
        _fail(failures, gate_id, "artefact_commit_unavailable", artefact_path)
        return None
    committed = _git_bytes(repo_root, "show", f"{revision}:{artefact_path}")
    if committed is None:
        _fail(failures, gate_id, "committed_artefact_unavailable", artefact_path)
        return None
    try:
        current = (repo_root / artefact_path).read_bytes()
    except OSError:
        _fail(failures, gate_id, "missing_benchmark_artefact", artefact_path)
        return None
    if committed != current:
        _fail(failures, gate_id, "artefact_differs_from_commit", artefact_path)
        return None
    return revision


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
    artefact_path: str,
    source_hash_policy: str,
    failures: list[GateFailure],
) -> None:
    revision = None
    if source_hashes and source_hash_policy == "artefact_commit":
        revision = _artefact_commit(repo_root, artefact_path, gate_id, failures)
        if revision is None:
            return
    for source_path, json_path in sorted(source_hashes.items()):
        if not _safe_repo_path(source_path):
            _fail(failures, gate_id, "invalid_source_path", source_path)
            continue
        try:
            recorded = _path_value(payload, json_path)
        except KeyError:
            _fail(failures, gate_id, "missing_recorded_source_hash", json_path)
            continue
        if source_hash_policy == "artefact_commit":
            source_bytes = _git_bytes(repo_root, "show", f"{revision}:{source_path}")
            if source_bytes is None:
                _fail(failures, gate_id, "missing_historical_source", source_path)
                continue
            actual = hashlib.sha256(source_bytes).hexdigest()
        else:
            source = repo_root / source_path
            if not source.is_file():
                _fail(failures, gate_id, "missing_source_for_hash_check", source_path)
                continue
            actual = _sha256(source)
        if recorded != actual:
            _fail(failures, gate_id, "source_hash_mismatch", f"{source_path}->{json_path}")


#: File suffixes that mark a ``source_hashes`` key as a repository path.
_SOURCE_PATH_SUFFIXES = (
    ".py",
    ".rs",
    ".go",
    ".jl",
    ".mojo",
    ".toml",
    ".json",
    ".yml",
    ".yaml",
    ".lock",
    ".sby",
    ".v",
    ".h",
    ".md",
    ".txt",
    ".sh",
    ".mod",
    ".sum",
)


def _record_source_hash_failures(
    artefact: Path,
    *,
    repo_root: Path,
    source_hash_paths: Mapping[str, str] | None,
    source_hash_policy: str,
) -> list[GateFailure]:
    """Compare a record's source digests under one policy and return the failures."""
    failures: list[GateFailure] = []
    try:
        relative = artefact.resolve().relative_to(repo_root.resolve()).as_posix()
        payload = _load_json(artefact)
    except (ValueError, OSError, json.JSONDecodeError):
        _fail(failures, "record", "artefact_is_missing_or_invalid", str(artefact))
        return failures
    if source_hash_paths is None:
        hashes = payload.get("source_hashes") if isinstance(payload, dict) else None
        if not isinstance(hashes, dict):
            _fail(failures, "record", "missing_source_hashes", relative)
            return failures
        paths = {
            path: f"source_hashes.{path}"
            for path in hashes
            if isinstance(path, str) and path.endswith(_SOURCE_PATH_SUFFIXES)
        }
    else:
        paths = dict(source_hash_paths)
    if not paths:
        _fail(failures, "record", "missing_path_keyed_source_hashes", relative)
        return failures
    _check_source_hashes(
        repo_root=repo_root,
        gate_id="record",
        payload=payload,
        source_hashes=paths,
        artefact_path=relative,
        source_hash_policy=source_hash_policy,
        failures=failures,
    )
    return failures


def committed_source_hash_failures(
    artefact: Path,
    *,
    repo_root: Path,
    source_hash_paths: Mapping[str, str] | None = None,
) -> list[GateFailure]:
    """Validate a measured record against its committed source snapshot.

    ``source_hash_paths`` maps repository source paths to their JSON paths for
    records that key source digests by lane or use a different field name.
    """
    return _record_source_hash_failures(
        artefact,
        repo_root=repo_root,
        source_hash_paths=source_hash_paths,
        source_hash_policy="artefact_commit",
    )


def current_source_hash_failures(
    artefact: Path,
    *,
    repo_root: Path,
    source_hash_paths: Mapping[str, str] | None = None,
) -> list[GateFailure]:
    """Compare a measured record with the sources as they are in the tree now.

    :func:`committed_source_hash_failures` answers whether the record was
    measured on the sources it names, at the commit that recorded it. This
    function answers the other question: whether those sources are still the
    same files. A failure with the reason ``source_hash_mismatch`` or
    ``missing_source_for_hash_check`` names a source that changed or is gone;
    it says that the record describes earlier bytes, not that the measured
    value changed.

    Parameters
    ----------
    artefact : Path
        The benchmark record, inside ``repo_root``.
    repo_root : Path
        Repository root whose working tree is read; no Git history is used.
    source_hash_paths : mapping of str to str, optional
        Repository source paths mapped to their JSON paths, for records that
        key source digests by lane or use a different field name.

    Returns
    -------
    list of GateFailure
        Empty when every path-keyed digest equals the file in the tree. The
        reasons ``missing_source_hashes`` and ``missing_path_keyed_source_hashes``
        mean that the record binds no source by path.
    """
    return _record_source_hash_failures(
        artefact,
        repo_root=repo_root,
        source_hash_paths=source_hash_paths,
        source_hash_policy="working_tree",
    )


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
    if manifest.get("source_hash_policy", "worktree") not in {
        "worktree",
        "artefact_commit",
    }:
        _fail(failures, "manifest", "invalid_source_hash_policy", "source_hash_policy")
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
    """Evaluate every gate of the manifest, write the report and return it.

    Parameters
    ----------
    manifest_path : Path
        JSON manifest that lists the required benchmark artefact gates.
    output_path : Path
        Where the machine-readable report is written.
    repo_root : Path, optional
        Repository root; the current directory when omitted.

    Returns
    -------
    dict
        The report: whether every gate passed, the policy used for source
        digests, and one entry per failure.
    """
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
    source_hash_policy = manifest.get("source_hash_policy", "worktree")

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
            artefact_path=artefact_path,
            source_hash_policy=source_hash_policy,
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
        "source_hash_policy": source_hash_policy,
        "evaluated_gates": evaluated,
        "failure_count": len(failures),
        "failures": [failure.to_json() for failure in failures],
        "passed": not failures,
    }
    _write_json(output_path, report)
    return report


def main(argv: list[str] | None = None) -> int:
    """Run the gate from the command line and return 0 only when it passes."""
    args = build_parser().parse_args(argv)
    report = evaluate_benchmark_evidence_gate(
        manifest_path=args.manifest,
        output_path=args.output,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
