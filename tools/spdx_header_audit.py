# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — SPDX header audit tool

"""Audit and repair direct SPDX headers for tracked source/config files."""

from __future__ import annotations

import argparse
import fnmatch
import subprocess
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Iterable, Sequence


SPDX_MARKER = "SPDX-License-Identifier: AGPL-3.0-or-later"
ORCID_MARKER = "ORCID: 0009-0009-3560-0851"
HEADER_BODY = (
    SPDX_MARKER,
    "Commercial license available",
    "© Concepts 1996–2026 Miroslav Šotek. All rights reserved.",
    "© Code 2020–2026 Miroslav Šotek. All rights reserved.",
    ORCID_MARKER,
    "Contact: www.anulum.li | protoscience@anulum.li",
    "SC-NeuroCore — Source/config provenance header",
)
HEADER_WINDOW_LINES = 20
LEGACY_METADATA_PREFIXES = (
    "SPDX-License-Identifier:",
    "Commercial license available",
    "© Concepts ",
    "© Code ",
    "ORCID:",
    "Contact:",
    "SC-NeuroCore — Source/config provenance header",
)

DIRECT_HEADER_PREFIX_BY_EXTENSION = {
    ".go": "//",
    ".jl": "#",
    ".js": "//",
    ".mojo": "#",
    ".py": "#",
    ".rs": "//",
    ".sh": "#",
    ".sv": "//",
    ".tcl": "#",
    ".toml": "#",
    ".ts": "//",
    ".tsx": "//",
    ".v": "//",
    ".vh": "//",
    ".yaml": "#",
    ".yml": "#",
}
"""Comment-safe source/config extensions that require direct headers."""

REUSE_METADATA_EXTENSIONS = {
    ".ipynb",
    ".json",
    ".lock",
    ".md",
    ".png",
    ".rst",
    ".svg",
    ".txt",
    ".xml",
}
"""Extensions covered by repository-level REUSE annotations or binary policy."""

GENERATED_OR_DATA_GLOBS = (
    ".tmp_audit_compile/**",
    "benchmarks/baselines/**",
    "benchmarks/results/**",
    "data/**",
    "docs/_generated/**",
    "examples/output/**",
    "hdl/reports/**",
    "results/**",
    "sc_shd_pynq/**",
    "site/**",
    "studio/frontend/package-lock.json",
    "weights/**",
)
"""Tracked generated/data surfaces intentionally not rewritten in-place."""

SKIP_PARTS = {
    ".git",
    ".mypy_cache",
    ".pixi",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "node_modules",
    "target",
    "venv",
}


class CoverageKind(str, Enum):
    """SPDX coverage category for a tracked repository file."""

    DIRECT_HEADER = "direct_header"
    GENERATED_OR_DATA = "generated_or_data"
    REUSE_METADATA = "reuse_metadata"
    UNCHECKED = "unchecked"


@dataclass(frozen=True, slots=True)
class SpdxRecord:
    """SPDX coverage classification for one tracked file."""

    path: str
    kind: CoverageKind
    missing_direct_header: bool
    reason: str


def tracked_files(root: Path) -> list[str]:
    """Return tracked files from ``git ls-files`` under ``root``."""

    completed = subprocess.run(
        ["git", "ls-files"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return [line for line in completed.stdout.splitlines() if line]


def _is_generated_or_data(path: str) -> bool:
    """Return true when ``path`` is generated evidence or data."""

    return any(fnmatch.fnmatch(path, pattern) for pattern in GENERATED_OR_DATA_GLOBS)


def _has_skipped_part(path: str) -> bool:
    """Return true when ``path`` lives below a local build/cache tree."""

    return any(part in SKIP_PARTS for part in Path(path).parts)


def _direct_header_prefix(path: str) -> str | None:
    """Return the comment prefix used for direct headers, if any."""

    return DIRECT_HEADER_PREFIX_BY_EXTENSION.get(Path(path).suffix.lower())


def direct_header_text(path: str) -> str:
    """Return the seven-line direct header for ``path``."""

    prefix = _direct_header_prefix(path)
    if prefix is None:
        raise ValueError(f"no direct-header style for {path}")
    return "\n".join(f"{prefix} {line}" for line in HEADER_BODY)


def has_direct_header(root: Path, path: str) -> bool:
    """Return true when ``path`` carries the direct SPDX header near the top."""

    try:
        leading = (root / path).read_text(encoding="utf-8").splitlines()[:HEADER_WINDOW_LINES]
    except OSError:
        return False
    text = "\n".join(leading)
    return SPDX_MARKER in text and ORCID_MARKER in text


def classify_path(root: Path, path: str) -> SpdxRecord:
    """Classify SPDX coverage for one tracked path."""

    if _has_skipped_part(path):
        return SpdxRecord(path, CoverageKind.UNCHECKED, False, "local cache/build path")
    if _is_generated_or_data(path):
        return SpdxRecord(path, CoverageKind.GENERATED_OR_DATA, False, "generated/data artifact")
    if _direct_header_prefix(path) is not None:
        missing = not has_direct_header(root, path)
        return SpdxRecord(path, CoverageKind.DIRECT_HEADER, missing, "comment-safe source/config")

    extension = Path(path).suffix.lower()
    if extension in REUSE_METADATA_EXTENSIONS or path == "REUSE.toml":
        return SpdxRecord(path, CoverageKind.REUSE_METADATA, False, "REUSE metadata annotation")
    return SpdxRecord(path, CoverageKind.UNCHECKED, False, "not a source/config policy target")


def collect_records(root: Path, paths: Iterable[str] | None = None) -> list[SpdxRecord]:
    """Collect SPDX coverage records for tracked or supplied files."""

    selected = tracked_files(root) if paths is None else list(paths)
    return [classify_path(root, path) for path in selected]


def missing_direct_header_paths(root: Path, paths: Iterable[str] | None = None) -> list[str]:
    """Return direct-header policy targets missing the seven-line header."""

    return [
        record.path
        for record in collect_records(root, paths)
        if record.kind is CoverageKind.DIRECT_HEADER and record.missing_direct_header
    ]


def _insertion_index(lines: list[str]) -> int:
    """Return the line index where a direct header can be inserted."""

    index = 0
    if lines and lines[0].startswith("#!"):
        index = 1
    while index < len(lines) and lines[index].startswith("/// <reference "):
        index += 1
    return index


def _strip_comment_prefix(line: str, prefix: str) -> str | None:
    """Return comment body for ``line`` when it uses ``prefix``."""

    stripped = line.lstrip()
    if not stripped.startswith(prefix):
        return None
    return stripped[len(prefix) :].strip()


def _drop_legacy_metadata(lines: list[str], index: int, prefix: str) -> list[str]:
    """Remove old partial SPDX metadata at the insertion point."""

    rewritten = [*lines[:index]]
    cursor = index
    while cursor < len(lines):
        body = _strip_comment_prefix(lines[cursor], prefix)
        if body is None or not body.startswith(LEGACY_METADATA_PREFIXES):
            break
        cursor += 1
    if cursor > index and cursor < len(lines) and lines[cursor] == "":
        cursor += 1
    return [*rewritten, *lines[cursor:]]


def apply_direct_header(root: Path, path: str) -> bool:
    """Insert the direct header into ``path`` if it is missing."""

    if has_direct_header(root, path):
        return False

    prefix = _direct_header_prefix(path)
    if prefix is None:
        raise ValueError(f"no direct-header style for {path}")

    target = root / path
    original = target.read_text(encoding="utf-8")
    newline = "\n" if original.endswith("\n") else ""
    lines = original.splitlines()
    index = _insertion_index(lines)
    lines = _drop_legacy_metadata(lines, index, prefix)
    header = direct_header_text(path).splitlines()
    rewritten = "\n".join([*lines[:index], *header, "", *lines[index:]]) + newline
    target.write_text(rewritten, encoding="utf-8")
    return True


def fix_missing_headers(root: Path, paths: Iterable[str] | None = None) -> list[str]:
    """Apply direct headers to all missing direct-header targets."""

    repaired: list[str] = []
    for path in missing_direct_header_paths(root, paths):
        if apply_direct_header(root, path):
            repaired.append(path)
    return repaired


def _print_summary(records: Sequence[SpdxRecord]) -> None:
    """Print a stable SPDX coverage summary."""

    for kind in CoverageKind:
        count = sum(1 for record in records if record.kind is kind)
        print(f"{kind.value}: {count}")
    missing = sum(1 for record in records if record.missing_direct_header)
    print(f"missing_direct_headers: {missing}")


def _parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="repository root")
    parser.add_argument("--fix", action="store_true", help="insert missing direct headers")
    parser.add_argument("--list-missing", action="store_true", help="print missing paths")
    parser.add_argument("--check", action="store_true", help="fail if direct headers are missing")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the SPDX header audit CLI."""

    args = _parser().parse_args(argv)
    root = args.root.resolve()

    repaired: list[str] = []
    if args.fix:
        repaired = fix_missing_headers(root)
        for path in repaired:
            print(f"repaired: {path}")

    records = collect_records(root)
    missing = [record.path for record in records if record.missing_direct_header]
    _print_summary(records)

    if args.list_missing or missing:
        for path in missing:
            print(path)

    should_check = args.check or not args.fix
    if should_check and missing:
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point.
    sys.exit(main())
