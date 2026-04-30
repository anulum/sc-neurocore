#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li

"""Audit model documentation evidence without rewriting model pages.

The audit intentionally separates machine-checkable evidence from scientific
review. It can prove that a page links source, tests, benchmark artifacts, and
required sections; it cannot prove that the biological interpretation is
correct. Use the generated manifest to prioritize human review batches.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "sc-neurocore.model-doc-audit.v1"
DEFAULT_TIMESTAMP = "2026-04-30T000000Z"
DOC_REQUIRED_SECTIONS: tuple[str, ...] = (
    "equation",
    "parameter",
    "default",
    "benchmark",
    "verification",
)
RUBRIC_KEYS: tuple[str, ...] = (
    "has_doc_page",
    "has_equation_section",
    "has_parameter_section",
    "has_default_section",
    "has_benchmark_section",
    "has_verification_section",
    "has_source_link",
    "has_tests",
    "has_benchmark_artifact",
)
STATUS_ORDER: tuple[str, ...] = (
    "NEEDS_DOC_PAGE",
    "NEEDS_TEST",
    "NEEDS_BENCHMARK",
    "NEEDS_DOC_EVIDENCE",
    "PASS",
)


@dataclass(frozen=True)
class ModelSource:
    """Source evidence extracted from one model module."""

    stem: str
    path: str
    classes: list[str]
    init_parameters: dict[str, list[str]]
    has_module_docstring: bool


@dataclass(frozen=True)
class DocEvidence:
    """Documentation evidence for one model source."""

    path: str | None
    has_equation_section: bool
    has_parameter_section: bool
    has_default_section: bool
    has_benchmark_section: bool
    has_verification_section: bool
    has_source_link: bool


@dataclass(frozen=True)
class ModelAuditEntry:
    """One row in the model documentation audit manifest."""

    model: str
    source: ModelSource
    doc: DocEvidence
    tests: list[str]
    benchmark_artifacts: list[str]
    missing: list[str]
    status: str


def discover_sources(source_dir: Path) -> list[ModelSource]:
    """Return auditable neuron model source modules in stable order."""
    sources: list[ModelSource] = []
    for path in sorted(source_dir.glob("*.py")):
        if path.name == "__init__.py":
            continue
        sources.append(_inspect_source(path))
    return sources


def build_manifest(repo: Path, *, timestamp: str = DEFAULT_TIMESTAMP) -> dict[str, Any]:
    """Build a JSON-serialisable model documentation audit manifest."""
    source_dir = repo / "src/sc_neurocore/neurons/models"
    docs_dir = repo / "docs/api/models"
    tests_dir = repo / "tests"
    benchmarks_dir = repo / "benchmarks/results"

    sources = discover_sources(source_dir)
    doc_paths = sorted(docs_dir.glob("*.md"))
    test_paths = sorted(tests_dir.glob("test_*.py"))
    benchmark_paths = sorted(
        path
        for path in benchmarks_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in {".json", ".md", ".log", ".txt"}
    )

    entries = [
        audit_model(source, doc_paths, test_paths, benchmark_paths, repo=repo) for source in sources
    ]
    orphan_doc_pages = _orphan_doc_pages(entries, doc_paths, repo)

    status_counts: dict[str, int] = {}
    for entry in entries:
        status_counts[entry.status] = status_counts.get(entry.status, 0) + 1

    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": timestamp,
        "repo": str(repo),
        "counts": {
            "source_models": len(sources),
            "doc_pages": len(doc_paths),
            "orphan_doc_pages": len(orphan_doc_pages),
            "status": dict(sorted(status_counts.items())),
        },
        "rubric_keys": list(RUBRIC_KEYS),
        "entries": [asdict(entry) for entry in entries],
        "orphan_doc_pages": orphan_doc_pages,
        "notes": [
            "Machine evidence only; biological fidelity and equation correctness remain human review gates.",
            "Older docs-debt counts are stale once source/docs inventory changes; use this manifest as the current baseline.",
        ],
    }


def audit_model(
    source: ModelSource,
    doc_paths: Sequence[Path],
    test_paths: Sequence[Path],
    benchmark_paths: Sequence[Path],
    *,
    repo: Path,
) -> ModelAuditEntry:
    """Audit one source model against docs, tests, and benchmark artifacts."""
    aliases = _aliases_for_source(source)
    doc_path = _match_first(doc_paths, aliases)
    tests = _matching_paths(test_paths, aliases, repo)
    benchmark_artifacts = _matching_paths(benchmark_paths, aliases, repo)
    doc = _inspect_doc(doc_path, source, repo=repo) if doc_path else _missing_doc()

    missing = _missing_rubric_items(
        doc=doc,
        has_doc_page=doc_path is not None,
        has_tests=bool(tests),
        has_benchmark_artifact=bool(benchmark_artifacts),
    )
    status = _status_for_missing(missing)

    return ModelAuditEntry(
        model=source.stem,
        source=source,
        doc=doc,
        tests=tests,
        benchmark_artifacts=benchmark_artifacts,
        missing=missing,
        status=status,
    )


def render_markdown(manifest: dict[str, Any]) -> str:
    """Render an append-only human-readable audit report."""
    counts = manifest["counts"]
    status_counts = counts["status"]
    lines = [
        f"# Model Documentation Audit - {manifest['generated_at']}",
        "",
        "This report is append-only audit evidence. It does not rewrite public model pages.",
        "",
        "## Scope",
        "",
        f"- Source model modules: {counts['source_models']}",
        f"- Model documentation pages: {counts['doc_pages']}",
        f"- Documentation pages without a direct source-module match: {counts['orphan_doc_pages']}",
        "",
        "## Status Counts",
        "",
        "| Status | Count |",
        "|---|---:|",
    ]
    for status, count in sorted(status_counts.items()):
        lines.append(f"| {status} | {count} |")

    entries = manifest["entries"]
    needs_review = [entry for entry in entries if entry["status"] != "PASS"]
    lines.extend(
        [
            "",
            "## Rubric",
            "",
            "| Key | Meaning |",
            "|---|---|",
            "| `has_doc_page` | A `docs/api/models/*.md` page maps to the source module. |",
            "| `has_equation_section` | The page contains equation/formulation language. |",
            "| `has_parameter_section` | The page contains parameter language. |",
            "| `has_default_section` | The page documents defaults or initial values. |",
            "| `has_benchmark_section` | The page contains benchmark/performance language. |",
            "| `has_verification_section` | The page contains verification/parity/test language. |",
            "| `has_source_link` | The page names the source path or module stem. |",
            "| `has_tests` | At least one test file maps by source/module/class alias. |",
            "| `has_benchmark_artifact` | At least one benchmark result artifact maps by source/module/class alias. |",
            "",
            "## Priority Review Queue",
            "",
            "| Model | Status | Missing Evidence | Doc | Tests | Benchmarks |",
            "|---|---|---|---|---:|---:|",
        ]
    )
    for entry in needs_review[:75]:
        doc_path = entry["doc"]["path"] or "-"
        missing = ", ".join(entry["missing"]) or "-"
        lines.append(
            f"| `{entry['model']}` | {entry['status']} | {missing} | "
            f"`{doc_path}` | {len(entry['tests'])} | {len(entry['benchmark_artifacts'])} |"
        )

    if len(needs_review) > 75:
        lines.append(
            f"| ... | ... | {len(needs_review) - 75} more entries in JSON manifest | ... | ... | ... |"
        )

    lines.extend(
        [
            "",
            "## Orphan Documentation Pages",
            "",
            "These pages need manual classification before they are counted as model-doc coverage.",
            "",
        ]
    )
    orphan_doc_pages = manifest["orphan_doc_pages"]
    if orphan_doc_pages:
        lines.extend(f"- `{path}`" for path in orphan_doc_pages[:100])
        if len(orphan_doc_pages) > 100:
            lines.append(f"- ... {len(orphan_doc_pages) - 100} more in JSON manifest")
    else:
        lines.append("- None detected.")

    lines.extend(
        [
            "",
            "## Acceleration Plan",
            "",
            "1. Treat `PASS` entries as candidates for short human spot-check, not automatic scientific approval.",
            "2. Batch `NEEDS_TEST` and `NEEDS_BENCHMARK` through code/test work before prose upgrades.",
            "3. Batch `NEEDS_DOC_EVIDENCE` by model family and append dated evidence sections to affected pages.",
            "4. Require human review for equations, references, and biological interpretation before status promotion.",
            "",
        ]
    )
    return "\n".join(lines)


def render_review_batch(
    manifest: dict[str, Any],
    *,
    statuses: Sequence[str] = (),
    missing: Sequence[str] = (),
    limit: int = 25,
) -> str:
    """Render a focused append-only review queue from a full manifest."""
    selected_statuses = _normalise_statuses(statuses)
    selected_missing = _normalise_missing_keys(missing)
    entries = _review_batch_entries(manifest, selected_statuses, selected_missing, limit=limit)
    status_label = ", ".join(selected_statuses) if selected_statuses else "non-PASS"
    missing_label = ", ".join(selected_missing) if selected_missing else "any"

    lines = [
        f"# Model Documentation Review Batch - {manifest['generated_at']}",
        "",
        f"Filter: `{status_label}`",
        f"Missing evidence: `{missing_label}`",
        f"Limit: `{limit}`",
        "",
        "This is a focused work queue derived from the full audit manifest. It does not rewrite model pages.",
        "",
        "| Model | Status | Missing Evidence | Doc | Suggested Next Action |",
        "|---|---|---|---|---|",
    ]
    for entry in entries:
        missing = ", ".join(entry["missing"]) or "-"
        doc_path = entry["doc"]["path"] or "-"
        lines.append(
            f"| `{entry['model']}` | {entry['status']} | {missing} | "
            f"`{doc_path}` | {_suggest_next_action(entry)} |"
        )

    if not entries:
        lines.append("| - | - | - | - | No matching entries. |")

    lines.extend(
        [
            "",
            "## Review Rules",
            "",
            "1. Add tests or benchmark artifacts before upgrading prose-only pages.",
            "2. Append dated evidence sections to existing model pages; do not replace current content.",
            "3. Keep biological interpretation and equation correctness as human-review gates.",
            "",
        ]
    )
    return "\n".join(lines)


def write_manifest_outputs(
    manifest: dict[str, Any], out_dir: Path, timestamp: str
) -> tuple[Path, Path]:
    """Write timestamped JSON and Markdown audit files."""
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_timestamp = re.sub(r"[^0-9A-Za-z]+", "_", timestamp).strip("_")
    json_path = out_dir / f"model_doc_audit_{safe_timestamp}.json"
    md_path = out_dir / f"model_doc_audit_{safe_timestamp}.md"
    json_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    md_path.write_text(render_markdown(manifest))
    return json_path, md_path


def write_review_batch_output(
    manifest: dict[str, Any],
    out_dir: Path,
    timestamp: str,
    *,
    statuses: Sequence[str] = (),
    missing: Sequence[str] = (),
    limit: int = 25,
) -> Path:
    """Write a timestamped focused review batch Markdown file."""
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_timestamp = _safe_filename_part(timestamp)
    status_slug = "_".join(_normalise_statuses(statuses)).lower() if statuses else "non_pass"
    missing_slug = "_".join(_normalise_missing_keys(missing)).lower() if missing else "any_missing"
    path = out_dir / f"model_doc_batch_{safe_timestamp}_{status_slug}_{missing_slug}.md"
    path.write_text(
        render_review_batch(
            manifest,
            statuses=statuses,
            missing=missing,
            limit=limit,
        )
    )
    return path


def _safe_filename_part(value: str) -> str:
    return re.sub(r"[^0-9A-Za-z]+", "_", value).strip("_")


def _inspect_source(path: Path) -> ModelSource:
    tree = ast.parse(path.read_text(), filename=str(path))
    classes: list[str] = []
    init_parameters: dict[str, list[str]] = {}
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            classes.append(node.name)
            init_parameters[node.name] = _init_parameters(node)

    return ModelSource(
        stem=path.stem,
        path=str(path),
        classes=classes,
        init_parameters=init_parameters,
        has_module_docstring=ast.get_docstring(tree) is not None,
    )


def _init_parameters(node: ast.ClassDef) -> list[str]:
    for body_node in node.body:
        if isinstance(body_node, ast.FunctionDef) and body_node.name == "__init__":
            return [arg.arg for arg in body_node.args.args if arg.arg != "self"]
    return []


def _aliases_for_source(source: ModelSource) -> set[str]:
    aliases = {source.stem, source.stem.replace("_neuron", ""), source.stem.replace("_cell", "")}
    for class_name in source.classes:
        snake = _to_snake_case(class_name)
        aliases.add(snake)
        aliases.add(snake.replace("_neuron", ""))
        aliases.add(snake.replace("_cell", ""))
    return {alias for alias in aliases if alias}


def _to_snake_case(value: str) -> str:
    value = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", value)
    value = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", value)
    return value.lower()


def _match_first(paths: Sequence[Path], aliases: set[str]) -> Path | None:
    for path in paths:
        if path.stem in aliases:
            return path
    return None


def _matching_paths(paths: Sequence[Path], aliases: set[str], repo: Path) -> list[str]:
    matches: list[str] = []
    for path in paths:
        haystack = f"{path.stem} {path.as_posix()}".lower()
        if any(_alias_hits_text(alias, haystack) for alias in aliases):
            matches.append(str(path.relative_to(repo)))
    return matches


def _alias_hits_text(alias: str, text: str) -> bool:
    normalized = alias.lower()
    if not normalized:
        return False
    if normalized in text:
        return True
    return normalized.replace("_", "-") in text


def _inspect_doc(path: Path, source: ModelSource, *, repo: Path) -> DocEvidence:
    text = path.read_text().lower()
    rel_path = str(path.relative_to(repo))
    source_rel = (
        str(Path(source.path).relative_to(repo)) if Path(source.path).is_absolute() else source.path
    )
    return DocEvidence(
        path=rel_path,
        has_equation_section=_contains_any(
            text, ("equation", "formulation", "dynamics", "update rule")
        ),
        has_parameter_section=_contains_any(text, ("parameter", "parameters")),
        has_default_section=_contains_any(text, ("default", "initial")),
        has_benchmark_section=_contains_any(
            text, ("benchmark", "performance", "throughput", "latency")
        ),
        has_verification_section=_contains_any(
            text, ("verification", "parity", "test", "validated")
        ),
        has_source_link=source_rel.lower() in text or source.stem.lower() in text,
    )


def _contains_any(text: str, needles: Iterable[str]) -> bool:
    return any(needle in text for needle in needles)


def _missing_doc() -> DocEvidence:
    return DocEvidence(
        path=None,
        has_equation_section=False,
        has_parameter_section=False,
        has_default_section=False,
        has_benchmark_section=False,
        has_verification_section=False,
        has_source_link=False,
    )


def _missing_rubric_items(
    *,
    doc: DocEvidence,
    has_doc_page: bool,
    has_tests: bool,
    has_benchmark_artifact: bool,
) -> list[str]:
    checks = {
        "has_doc_page": has_doc_page,
        "has_equation_section": doc.has_equation_section,
        "has_parameter_section": doc.has_parameter_section,
        "has_default_section": doc.has_default_section,
        "has_benchmark_section": doc.has_benchmark_section,
        "has_verification_section": doc.has_verification_section,
        "has_source_link": doc.has_source_link,
        "has_tests": has_tests,
        "has_benchmark_artifact": has_benchmark_artifact,
    }
    return [key for key in RUBRIC_KEYS if not checks[key]]


def _status_for_missing(missing: Sequence[str]) -> str:
    if not missing:
        return "PASS"
    if "has_doc_page" in missing:
        return "NEEDS_DOC_PAGE"
    if "has_tests" in missing:
        return "NEEDS_TEST"
    if "has_benchmark_artifact" in missing:
        return "NEEDS_BENCHMARK"
    return "NEEDS_DOC_EVIDENCE"


def _normalise_statuses(statuses: Sequence[str]) -> tuple[str, ...]:
    normalised = tuple(dict.fromkeys(status.upper() for status in statuses))
    unknown = [status for status in normalised if status not in STATUS_ORDER]
    if unknown:
        expected = ", ".join(STATUS_ORDER)
        raise argparse.ArgumentTypeError(
            f"unknown audit status {unknown[0]!r}; expected one of: {expected}"
        )
    return normalised


def _normalise_missing_keys(missing: Sequence[str]) -> tuple[str, ...]:
    normalised = tuple(dict.fromkeys(key.lower() for key in missing))
    unknown = [key for key in normalised if key not in RUBRIC_KEYS]
    if unknown:
        expected = ", ".join(RUBRIC_KEYS)
        raise argparse.ArgumentTypeError(
            f"unknown missing-evidence key {unknown[0]!r}; expected one of: {expected}"
        )
    return normalised


def _review_batch_entries(
    manifest: dict[str, Any],
    statuses: Sequence[str],
    missing: Sequence[str],
    *,
    limit: int,
) -> list[dict[str, Any]]:
    if limit < 1:
        raise argparse.ArgumentTypeError("--batch-limit must be at least 1")
    selected = set(statuses)
    selected_missing = set(missing)
    entries = [
        entry
        for entry in manifest["entries"]
        if (entry["status"] in selected if selected else entry["status"] != "PASS")
        and (selected_missing.issubset(set(entry["missing"])) if selected_missing else True)
    ]
    status_rank = {status: index for index, status in enumerate(STATUS_ORDER)}
    entries.sort(key=lambda entry: (status_rank.get(entry["status"], 999), entry["model"]))
    return entries[:limit]


def _suggest_next_action(entry: dict[str, Any]) -> str:
    missing = set(entry["missing"])
    if "has_doc_page" in missing:
        return "Create a model page from source, then add evidence sections."
    if "has_tests" in missing:
        return "Add or map behavioural/parity tests before prose promotion."
    if "has_benchmark_artifact" in missing:
        return "Add benchmark evidence or attach existing artifact mapping."
    return "Append dated equation, parameter, benchmark, or verification evidence."


def _orphan_doc_pages(
    entries: Sequence[ModelAuditEntry], doc_paths: Sequence[Path], repo: Path
) -> list[str]:
    matched = {entry.doc.path for entry in entries if entry.doc.path is not None}
    return [
        str(path.relative_to(repo))
        for path in doc_paths
        if str(path.relative_to(repo)) not in matched
    ]


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd(), help="Repository root")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("docs/internal"),
        help="Directory for timestamped JSON and Markdown outputs",
    )
    parser.add_argument(
        "--timestamp",
        default=DEFAULT_TIMESTAMP,
        help="Audit timestamp embedded in output filenames and manifests",
    )
    parser.add_argument("--json", action="store_true", help="Print manifest JSON to stdout")
    parser.add_argument(
        "--batch-status",
        action="append",
        default=[],
        metavar="STATUS",
        help="Also write a focused review batch for STATUS. Repeat for multiple statuses.",
    )
    parser.add_argument(
        "--batch-missing",
        action="append",
        default=[],
        metavar="RUBRIC_KEY",
        help="Restrict the review batch to entries missing RUBRIC_KEY. Repeat to require multiple keys.",
    )
    parser.add_argument(
        "--batch-limit",
        type=int,
        default=25,
        help="Maximum number of entries in the focused review batch",
    )
    parser.add_argument(
        "--check", action="store_true", help="Return non-zero if any entry is not PASS"
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the model documentation audit CLI."""
    args = _parse_args(argv)
    repo = args.repo.resolve()
    out_dir = args.out_dir if args.out_dir.is_absolute() else repo / args.out_dir
    manifest = build_manifest(repo, timestamp=args.timestamp)
    json_path, md_path = write_manifest_outputs(manifest, out_dir, args.timestamp)
    batch_path = (
        write_review_batch_output(
            manifest,
            out_dir,
            args.timestamp,
            statuses=args.batch_status,
            missing=args.batch_missing,
            limit=args.batch_limit,
        )
        if args.batch_status or args.batch_missing
        else None
    )

    if args.json:
        print(json.dumps(manifest, indent=2, sort_keys=True))
    else:
        print(f"Wrote {json_path}")
        print(f"Wrote {md_path}")
        if batch_path is not None:
            print(f"Wrote {batch_path}")
        print(json.dumps(manifest["counts"], indent=2, sort_keys=True))

    if args.check and manifest["counts"]["status"] != {"PASS": manifest["counts"]["source_models"]}:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
