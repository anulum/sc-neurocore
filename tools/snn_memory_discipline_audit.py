# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — SNN memory discipline audit

from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence


SCHEMA_VERSION = 1
DEFAULT_PROJECT = "SC-NEUROCORE"
CANONICAL_KEYS = frozenset(
    {
        "content",
        "project",
        "actor",
        "timestamp",
        "entities",
        "kind",
        "source_ref",
    }
)
CONTINUITY_EXTENSION_KEYS = frozenset({"records", "seat", "source_identity"})
CANONICAL_CONTINUITY_KEYS = CANONICAL_KEYS | CONTINUITY_EXTENSION_KEYS
CONTROLLED_ACTORS = frozenset({"codex", "claude", "gemini", "grok", "system", "operator"})
CONTROLLED_KINDS = frozenset({"event", "finding", "decision", "state"})
CONTINUITY_KINDS = frozenset({"session_evidence", "task_completion"})
SUPERSEDES_RECORD = re.compile(r"\bSupersedes\s+([A-Za-z0-9][A-Za-z0-9_.-]*\.json)\b", re.I)


@dataclass(frozen=True)
class ProducerCandidate:
    """Tracked source surface that can write SNN memory records."""

    path: str
    function: str
    source_refs: tuple[str, ...]

    def to_json(self) -> dict[str, object]:
        """Return a stable JSON-compatible producer object."""

        return {
            "path": self.path,
            "function": self.function,
            "source_refs": list(self.source_refs),
        }


@dataclass(frozen=True)
class StimulusViolation:
    """Single memory-discipline violation for one stimulus file."""

    path: str
    code: str
    detail: str

    def to_json(self) -> dict[str, str]:
        """Return a stable JSON-compatible violation object."""

        return {"path": self.path, "code": self.code, "detail": self.detail}


@dataclass(frozen=True)
class MemoryDisciplineAudit:
    """Audit result for SC-NeuroCore SNN memory producers and records."""

    schema_version: int
    project: str
    producer_candidates: tuple[ProducerCandidate, ...]
    stimulus_dir: str
    checked_records: int
    violations: tuple[StimulusViolation, ...]

    @property
    def passed(self) -> bool:
        """Return whether producers exist and every checked record is canonical."""

        return bool(self.producer_candidates) and not self.violations

    def to_json(self) -> dict[str, object]:
        """Return a stable JSON-compatible audit payload."""

        return {
            "schema_version": self.schema_version,
            "passed": self.passed,
            "project": self.project,
            "producer_candidate_count": len(self.producer_candidates),
            "producer_candidates": [item.to_json() for item in self.producer_candidates],
            "stimulus_dir": self.stimulus_dir,
            "checked_records": self.checked_records,
            "violation_count": len(self.violations),
            "violations": [item.to_json() for item in self.violations],
        }


def discover_snn_producers(repo: Path) -> tuple[ProducerCandidate, ...]:
    """Return tracked Python functions that write canonical SNN stimuli.

    Parameters
    ----------
    repo
        Repository root used for `git ls-files` discovery.

    Returns
    -------
    tuple[ProducerCandidate, ...]
        Candidate writer functions with their declared `source_ref` values.
    """

    candidates: list[ProducerCandidate] = []
    for relative in _tracked_python_files(repo):
        path = repo / relative
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if not _looks_like_snn_writer(node):
                continue
            refs = tuple(sorted(_source_refs(node)))
            candidates.append(
                ProducerCandidate(
                    path=relative.as_posix(),
                    function=node.name,
                    source_refs=refs,
                )
            )
    return tuple(sorted(candidates, key=lambda item: (item.path, item.function)))


def audit_memory_discipline(repo: Path, stimulus_dir: Path, project: str) -> MemoryDisciplineAudit:
    """Build a producer and stimulus-file memory-discipline audit."""

    records = tuple(sorted(stimulus_dir.glob("*.json")))
    violations_by_record = {
        record: validate_stimulus_file(record, stimulus_dir, project) for record in records
    }
    superseded_records = _superseded_records(
        records,
        violations_by_record=violations_by_record,
        stimulus_dir=stimulus_dir,
    )
    violations: list[StimulusViolation] = []
    for record in records:
        if record not in superseded_records:
            violations.extend(violations_by_record[record])
    return MemoryDisciplineAudit(
        schema_version=SCHEMA_VERSION,
        project=project,
        producer_candidates=discover_snn_producers(repo),
        stimulus_dir=str(stimulus_dir),
        checked_records=len(records),
        violations=tuple(violations),
    )


def _superseded_records(
    records: tuple[Path, ...],
    *,
    violations_by_record: dict[Path, tuple[StimulusViolation, ...]],
    stimulus_dir: Path,
) -> set[Path]:
    """Return invalid predecessors named by later canonical successor records."""

    superseded: set[Path] = set()
    for successor in records:
        if violations_by_record[successor]:
            continue
        payload = json.loads(successor.read_text(encoding="utf-8"))
        content = payload.get("content")
        if not isinstance(content, str):
            continue
        for target_name in SUPERSEDES_RECORD.findall(content):
            target = stimulus_dir / target_name
            if (
                target != successor
                and target in violations_by_record
                and violations_by_record[target]
                and successor.stat().st_mtime > target.stat().st_mtime
            ):
                superseded.add(target)
    return superseded


def validate_stimulus_file(path: Path, root: Path, project: str) -> tuple[StimulusViolation, ...]:
    """Validate one SNN stimulus JSON file against the fleet schema."""

    relative = _display_path(path, root)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return (
            StimulusViolation(
                path=relative,
                code="invalid_json",
                detail=f"{exc.msg} at line {exc.lineno}, column {exc.colno}",
            ),
        )
    if not isinstance(payload, dict):
        return (
            StimulusViolation(
                path=relative,
                code="invalid_payload",
                detail="top-level JSON value must be an object",
            ),
        )
    return tuple(_validate_payload(relative, payload, project))


def repair_stimulus_file(path: Path, project: str) -> dict[str, object]:
    """Rewrite one legacy SNN stimulus file into canonical schema form."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} top-level JSON value must be an object")
    repaired = _repair_payload(payload, project, path.name)
    path.write_text(json.dumps(repaired, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return repaired


def repair_stimulus_dir(stimulus_dir: Path, project: str) -> int:
    """Repair every JSON stimulus file in a directory and return the count."""

    repaired = 0
    for path in sorted(stimulus_dir.glob("*.json")):
        before = path.read_text(encoding="utf-8")
        repair_stimulus_file(path, project)
        after = path.read_text(encoding="utf-8")
        if before != after:
            repaired += 1
    return repaired


def _tracked_python_files(repo: Path) -> tuple[Path, ...]:
    result = subprocess.run(
        ["git", "ls-files", "*.py"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    files = []
    for line in result.stdout.splitlines():
        relative = Path(line)
        if relative.parts and relative.parts[0] in {"tests", "docs"}:
            continue
        files.append(relative)
    return tuple(sorted(files))


def _looks_like_snn_writer(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    names = {subnode.id for subnode in ast.walk(node) if isinstance(subnode, ast.Name)}
    string_values = {
        subnode.value
        for subnode in ast.walk(node)
        if isinstance(subnode, ast.Constant) and isinstance(subnode.value, str)
    }
    has_schema_payload = CANONICAL_KEYS.issubset(string_values)
    has_file_write = "json" in names and "open" in names
    return has_schema_payload and has_file_write and "stimulus" in node.name.lower()


def _source_refs(node: ast.FunctionDef | ast.AsyncFunctionDef) -> Iterable[str]:
    for subnode in ast.walk(node):
        if not isinstance(subnode, ast.Dict):
            continue
        for key, value in zip(subnode.keys, subnode.values, strict=False):
            if not _constant_str_equals(key, "source_ref"):
                continue
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                yield value.value


def _constant_str_equals(node: ast.expr | None, expected: str) -> bool:
    return isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value == expected


def _validate_payload(
    path: str, payload: dict[str, Any], project: str
) -> Iterable[StimulusViolation]:
    keys = set(payload)
    if keys not in {CANONICAL_KEYS, CANONICAL_CONTINUITY_KEYS}:
        missing = sorted(CANONICAL_KEYS - keys)
        extra = sorted(keys - CANONICAL_CONTINUITY_KEYS)
        detail = f"missing={missing}; extra={extra}"
        yield StimulusViolation(path=path, code="noncanonical_keys", detail=detail)

    content = payload.get("content")
    if not isinstance(content, str) or len(content.strip()) < 15:
        yield StimulusViolation(
            path=path, code="invalid_content", detail="content must be a string with >=15 chars"
        )

    if payload.get("project") != project:
        yield StimulusViolation(
            path=path, code="invalid_project", detail=f"project must be {project}"
        )

    actor = payload.get("actor")
    if actor not in CONTROLLED_ACTORS:
        yield StimulusViolation(
            path=path,
            code="invalid_actor",
            detail=f"actor must be one of {sorted(CONTROLLED_ACTORS)}",
        )

    if not _valid_timestamp(payload.get("timestamp")):
        yield StimulusViolation(
            path=path,
            code="invalid_timestamp",
            detail="timestamp must be positive Unix seconds or ISO-8601 text",
        )

    entities = payload.get("entities")
    if (
        not isinstance(entities, list)
        or not entities
        or not all(isinstance(item, str) and item for item in entities)
    ):
        yield StimulusViolation(
            path=path, code="invalid_entities", detail="entities must be a non-empty string list"
        )

    kind = payload.get("kind")
    allowed_kinds = CONTINUITY_KINDS if keys == CANONICAL_CONTINUITY_KEYS else CONTROLLED_KINDS
    if kind not in allowed_kinds:
        yield StimulusViolation(
            path=path, code="invalid_kind", detail=f"kind must be one of {sorted(allowed_kinds)}"
        )

    source_ref = payload.get("source_ref")
    if not isinstance(source_ref, str) or not source_ref.strip():
        yield StimulusViolation(
            path=path, code="invalid_source_ref", detail="source_ref must be a non-empty string"
        )

    if keys == CANONICAL_CONTINUITY_KEYS:
        yield from _validate_continuity_extension(path, payload, project)


def _validate_continuity_extension(
    path: str, payload: dict[str, Any], project: str
) -> Iterable[StimulusViolation]:
    """Validate recovery-grade fields on the exact Tier-0 extension schema."""

    records = payload.get("records")
    expected_prefixes = {
        "session": ".coordination/sessions/",
        "handover": ".coordination/handovers/",
    }
    records_valid = isinstance(records, dict) and set(records) == set(expected_prefixes)
    if records_valid:
        records_valid = all(
            isinstance(records[name], str)
            and records[name].startswith(prefix)
            and records[name].endswith(".md")
            for name, prefix in expected_prefixes.items()
        )
    if not records_valid:
        yield StimulusViolation(
            path=path,
            code="invalid_records",
            detail="records must contain canonical session and handover Markdown paths",
        )

    for field in ("seat", "source_identity"):
        value = payload.get(field)
        if not isinstance(value, str) or not value.startswith(f"{project}/"):
            yield StimulusViolation(
                path=path,
                code=f"invalid_{field}",
                detail=f"{field} must name a {project}/ identity",
            )


def _valid_timestamp(value: object) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        return value > 0
    if not isinstance(value, str) or not value.strip():
        return False
    candidate = value.strip()
    if candidate.endswith("Z"):
        candidate = candidate[:-1] + "+00:00"
    try:
        datetime.fromisoformat(candidate)
    except ValueError:
        return False
    return True


def _repair_payload(
    payload: dict[str, Any], project: str, fallback_source: str
) -> dict[str, object]:
    content = payload.get("content")
    if not isinstance(content, str) or len(content.strip()) < 15:
        content = _content_from_legacy_payload(payload)

    return {
        "actor": _normalise_actor(payload.get("actor")),
        "content": content,
        "entities": _normalise_entities(payload.get("entities"), project),
        "kind": _normalise_kind(payload.get("kind")),
        "project": project,
        "source_ref": _normalise_source_ref(
            payload.get("source_ref"), payload.get("commit"), fallback_source
        ),
        "timestamp": _normalise_timestamp(payload.get("timestamp"), payload.get("unix_epoch")),
    }


def _content_from_legacy_payload(payload: dict[str, Any]) -> str:
    parts: list[str] = []
    summary = payload.get("summary")
    if isinstance(summary, str) and summary.strip():
        parts.append(summary.strip())
    commit = payload.get("commit")
    if isinstance(commit, str) and commit.strip():
        parts.append(f"Commit: {commit.strip()}.")
    todo_rows = payload.get("todo_rows_closed")
    if isinstance(todo_rows, list) and todo_rows:
        rows = ", ".join(str(item) for item in todo_rows)
        parts.append(f"TODO rows closed: {rows}.")
    evidence = payload.get("evidence")
    if isinstance(evidence, list) and evidence:
        items = "; ".join(str(item) for item in evidence)
        parts.append(f"Evidence: {items}.")
    if parts:
        return " ".join(parts)
    return "Legacy SC-NEUROCORE SNN memory record normalised to the canonical write schema."


def _normalise_actor(value: object) -> str:
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered.startswith("codex"):
            return "codex"
        if lowered in CONTROLLED_ACTORS:
            return lowered
    return "system"


def _normalise_entities(value: object, project: str) -> list[str]:
    if isinstance(value, list):
        entities = [item for item in value if isinstance(item, str) and item]
        if entities:
            return sorted(set(entities))
    return [project]


def _normalise_kind(value: object) -> str:
    if isinstance(value, str) and value in CONTROLLED_KINDS:
        return value
    return "event"


def _normalise_source_ref(source_ref: object, commit: object, fallback_source: str) -> str:
    if isinstance(source_ref, str) and source_ref.strip():
        return source_ref.strip()
    if isinstance(commit, str) and commit.strip():
        return commit.strip()
    return fallback_source


def _normalise_timestamp(timestamp: object, unix_epoch: object) -> int | float | str:
    if _valid_timestamp(timestamp):
        return timestamp if isinstance(timestamp, (int, float)) else str(timestamp)
    if _valid_timestamp(unix_epoch):
        return unix_epoch if isinstance(unix_epoch, (int, float)) else str(unix_epoch)
    raise ValueError("legacy payload does not contain a valid timestamp")


def _display_path(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd(), help="Repository root.")
    parser.add_argument(
        "--stimulus-dir",
        type=Path,
        required=True,
        help="Directory containing SC-NeuroCore SNN stimulus JSON files.",
    )
    parser.add_argument(
        "--project", default=DEFAULT_PROJECT, help="Expected uppercase project slug."
    )
    parser.add_argument("--output", type=Path, help="Optional JSON report path.")
    parser.add_argument(
        "--repair", action="store_true", help="Normalise legacy records before auditing."
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the SNN memory-discipline audit CLI."""

    args = _parse_args(sys.argv[1:] if argv is None else argv)
    repo = args.repo.resolve()
    stimulus_dir = args.stimulus_dir.resolve()
    if args.repair:
        repair_stimulus_dir(stimulus_dir, args.project)
    audit = audit_memory_discipline(repo, stimulus_dir, args.project)
    rendered = json.dumps(audit.to_json(), indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.write_text(rendered, encoding="utf-8")
    else:
        sys.stdout.write(rendered)
    return 0 if audit.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
