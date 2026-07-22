#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — engine crate-root (lib.rs) benchmark-evidence provenance gate

"""Fail-closed freshness gate for engine crate-root provenance in committed evidence.

The manifest-driven :mod:`tools.benchmark_evidence_gate` only inspects the
hand-declared artefacts listed in ``benchmarks/benchmark_regression_gates.json``,
and per-benchmark drift is otherwise caught only by hand-written committed-evidence
tests. A benchmark JSON that records an ``engine/src/lib.rs`` source hash outside
that hand list — or without a bespoke per-bench test — can drift undetected: a
stale crate-root hash slips past both the manifest gate and the pytest suite. This
gate closes that class. It discovers the file set instead of declaring it: every
committed benchmark JSON is enumerated, its top-level canonical provenance map is
read, and any recorded ``engine/src/lib.rs`` hash that disagrees with the live
crate root fails the gate. No hand list of files is maintained here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeGuard

ENGINE_CRATE_ROOT = "engine/src/lib.rs"
# Top-level keys under which committed evidence can record source provenance.
# ``source_sha256`` is also used as a scalar aggregate digest; only its mapping
# form can carry a direct crate-root binding.
PROVENANCE_MAP_KEYS = ("source_hashes", "source_sha256")
_SHA256_HEX_LENGTH = 64
DEFAULT_BENCHMARKS_ROOT = Path("benchmarks")


@dataclass(frozen=True)
class ProvenanceFailure:
    """One stale (or unresolvable) engine crate-root provenance binding."""

    artefact: str
    recorded: str
    expected: str
    reason: str

    def to_json(self) -> dict[str, str]:
        """Return the JSON-serialisable form of this failure."""
        return {
            "artefact": self.artefact,
            "recorded": self.recorded,
            "expected": self.expected,
            "reason": self.reason,
        }


def _looks_like_sha256(value: Any) -> TypeGuard[str]:
    """Return whether ``value`` is a lowercase 64-character hex sha256 string."""
    return (
        isinstance(value, str)
        and len(value) == _SHA256_HEX_LENGTH
        and all(character in "0123456789abcdef" for character in value)
    )


def _sha256(path: Path) -> str:
    """Return the sha256 hex digest of the file at ``path``."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artefact_name(artefact: Path, repo_root: Path) -> str:
    """Return a stable repository-relative name when possible."""
    try:
        return artefact.relative_to(repo_root).as_posix()
    except ValueError:
        return artefact.as_posix()


def iter_crate_root_bindings(benchmarks_root: Path) -> Iterator[tuple[Path, str]]:
    """Yield ``(artefact, recorded_hash)`` for every benchmark JSON binding the crate root.

    Only the top-level canonical provenance map is read; nested ``variants[*]``
    blocks are historical per-configuration snapshots and are deliberately skipped.
    """
    for artefact in sorted(benchmarks_root.rglob("*.json")):
        try:
            payload = json.loads(artefact.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict):
            continue
        for key in PROVENANCE_MAP_KEYS:
            mapping = payload.get(key)
            if not isinstance(mapping, dict):
                continue
            recorded = mapping.get(ENGINE_CRATE_ROOT)
            if _looks_like_sha256(recorded):
                yield artefact, recorded


def evaluate(repo_root: Path, benchmarks_root: Path | None = None) -> list[ProvenanceFailure]:
    """Return every stale engine crate-root provenance binding under the benchmarks tree."""
    root = benchmarks_root if benchmarks_root is not None else repo_root / DEFAULT_BENCHMARKS_ROOT
    crate_root = repo_root / ENGINE_CRATE_ROOT
    if not crate_root.is_file():
        return [
            ProvenanceFailure(
                artefact=ENGINE_CRATE_ROOT,
                recorded="",
                expected="",
                reason="engine_crate_root_missing",
            )
        ]
    expected = _sha256(crate_root)
    failures: list[ProvenanceFailure] = []
    for artefact in sorted(root.rglob("*.json")):
        try:
            payload = json.loads(artefact.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            failures.append(
                ProvenanceFailure(
                    artefact=_artefact_name(artefact, repo_root),
                    recorded="",
                    expected=expected,
                    reason="invalid_benchmark_json",
                )
            )
            continue
        if not isinstance(payload, dict):
            continue
        for key in PROVENANCE_MAP_KEYS:
            if key not in payload:
                continue
            mapping = payload[key]
            if key == "source_sha256" and _looks_like_sha256(mapping):
                continue
            if not isinstance(mapping, dict):
                failures.append(
                    ProvenanceFailure(
                        artefact=_artefact_name(artefact, repo_root),
                        recorded=repr(mapping),
                        expected=expected,
                        reason="invalid_provenance_map",
                    )
                )
                continue
            if ENGINE_CRATE_ROOT in mapping and not _looks_like_sha256(mapping[ENGINE_CRATE_ROOT]):
                failures.append(
                    ProvenanceFailure(
                        artefact=_artefact_name(artefact, repo_root),
                        recorded=repr(mapping[ENGINE_CRATE_ROOT]),
                        expected=expected,
                        reason="invalid_engine_crate_root_hash",
                    )
                )
    for artefact, recorded in iter_crate_root_bindings(root):
        if recorded != expected:
            failures.append(
                ProvenanceFailure(
                    artefact=_artefact_name(artefact, repo_root),
                    recorded=recorded,
                    expected=expected,
                    reason="stale_engine_crate_root_hash",
                )
            )
    return failures


def build_parser() -> argparse.ArgumentParser:
    """Return the command-line parser for the crate-root provenance gate."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path.cwd(),
        help="Repository root whose engine crate root and benchmarks are gated.",
    )
    parser.add_argument(
        "--benchmarks-root",
        type=Path,
        default=None,
        help="Benchmarks directory to sweep (default: <repo-root>/benchmarks).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the gate and return ``0`` when every crate-root binding is fresh, else ``1``."""
    args = build_parser().parse_args(argv)
    failures = evaluate(repo_root=args.repo_root, benchmarks_root=args.benchmarks_root)
    report = {
        "SPDX-License-Identifier": "AGPL-3.0-or-later",
        "engine_crate_root": ENGINE_CRATE_ROOT,
        "failure_count": len(failures),
        "failures": [failure.to_json() for failure in failures],
        "passed": not failures,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
