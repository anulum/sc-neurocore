# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — exact Go documentation measurement inputs

"""Bind native Go documentation figures to their submitted files and parser bytes."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path, PurePosixPath
import subprocess
import re
from typing import cast

from tools.docstring_policy_git import git_bytes
from tools.docstring_policy_scope import PolicyError

GO_COVERAGE_TOOL = "tools/godoc_coverage/main.go"
GO_COVERAGE_SCHEMA_VERSION = "sc-neurocore.go-doc-coverage.v3"
GO_MEASUREMENT_SCHEMA_VERSION = "sc-neurocore.go-doc-measurement.v2"
_COUNTS = (
    "undocumented",
    "files_scanned",
    "files_with_findings",
    "packages_total",
    "packages_undocumented",
)
_KINDS = {"package", "func", "method", "type", "const", "var"}
_CONFIGS = ("go.mod", "go.sum", "go.work", "go.work.sum")


class GoMeasurementError(ValueError):
    """Refuse an unavailable producer, ambiguous cohort or inconsistent summary."""


@dataclass(frozen=True)
class SourceSnapshot:
    """Record the ordered submitted cohort and its source and parser byte hashes."""

    paths: tuple[str, ...]
    pins: dict[str, str]
    source_sha256: str
    scope_sha256: str

    def verify(self, root: Path) -> None:
        """Require the same readable source and parser bytes in the repository."""
        if capture_source(root, self.paths) != self:
            raise GoMeasurementError("Go measurement inputs changed while the native parser ran.")


@dataclass(frozen=True)
class GoDocMeasurement:
    """Carry a validated native figure with unchanged observed measurement inputs."""

    summary: dict[str, object]
    version: str
    source: SourceSnapshot

    @property
    def undocumented(self) -> int:
        """Return the validated number of undocumented exported declarations."""
        return cast(int, self.summary["undocumented"])

    @property
    def files(self) -> int:
        """Return the validated number of submitted files carrying findings."""
        return cast(int, self.summary["files_with_findings"])

    def verify(self, root: Path, *, tracked: bool = False) -> None:
        """Recheck source bytes, native Go version and optional Git-discovered scope."""
        self.source.verify(root)
        if _run(root, ["go", "version"]).strip() != self.version:
            raise GoMeasurementError("The Go toolchain version changed during native measurement.")
        if tracked:
            if tuple(go_scope(root)) != self.source.paths:
                raise GoMeasurementError(
                    "The Git-discovered Go cohort changed during native measurement."
                )

    def to_public_dict(self) -> dict[str, object]:
        """Serialize the exact local cohort, source hashes, native argv and result.

        External compiler and standard-library bytes are not captured. This
        record does not establish protected historical debt or ignored-source
        ownership beyond the submitted cohort.
        """
        return {
            "schema_version": GO_MEASUREMENT_SCHEMA_VERSION,
            "argv": ["go", "run", GO_COVERAGE_TOOL],
            "go": self.version,
            "paths": list(self.source.paths),
            "source_pins": dict(self.source.pins),
            "source_sha256": self.source.source_sha256,
            "scope_sha256": self.source.scope_sha256,
            "summary": dict(self.summary),
            "qualification_scope": "submitted sources, parser and present root Go module/workspace files",
        }


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _run(root: Path, argv: list[str], *, input_text: str | None = None) -> str:
    try:
        result = subprocess.run(
            argv,
            cwd=root,
            input=input_text,
            capture_output=True,
            text=True,
            check=False,
            timeout=120 if input_text is None else 1800,
        )
    except (OSError, subprocess.SubprocessError, UnicodeError) as exc:
        raise GoMeasurementError("The native Go measurement command could not complete.") from exc
    if result.returncode != 0:
        raise GoMeasurementError("The native Go measurement command failed.")
    return result.stdout


def go_scope(root: Path) -> list[str]:
    """Query native Git for tracked and nonignored untracked Go source files.

    Ignored, generated and external ownership still need explicit qualification.
    """
    try:
        output = git_bytes(
            root, "ls-files", "--cached", "--others", "--exclude-standard", "-z", "--", "*.go"
        ).decode("utf-8")
    except (PolicyError, UnicodeError) as exc:
        raise GoMeasurementError("The native Git Go source cohort could not be read.") from exc
    if output and not output.endswith("\0"):
        raise GoMeasurementError("Git did not return a complete NUL-delimited Go cohort.")
    return sorted(set(output[:-1].split("\0"))) if output else []


def capture_source(root: Path, paths: Sequence[str]) -> SourceSnapshot:
    """Hash a nonempty, unique, unambiguous cohort and the actual parser inputs.

    Parameters
    ----------
    root : Path
        Repository whose source files the native parser reads.
    paths : sequence of str
        Exact relative Go paths submitted through the line-delimited protocol.

    Returns
    -------
    SourceSnapshot
        Ordered cohort and deterministic SHA-256 hashes of observed input bytes.

    Raises
    ------
    GoMeasurementError
        If the scope is empty, ambiguous, unreadable or escapes the repository.
    """
    if not paths or len(set(paths)) != len(paths):
        raise GoMeasurementError("The Go cohort must contain unique nonempty paths.")
    for path in paths:
        if (
            not path
            or path.strip() != path
            or "\\" in path
            or any(ord(char) < 32 for char in path)
            or PurePosixPath(path).is_absolute()
            or ".." in PurePosixPath(path).parts
            or PurePosixPath(path).as_posix() != path
            or not path.endswith(".go")
        ):
            raise GoMeasurementError("Go paths must be unambiguous relative source paths.")
    physical_root = root.resolve()
    inputs = set(paths) | {GO_COVERAGE_TOOL}
    inputs.update(path for path in _CONFIGS if (physical_root / path).exists())
    pins: dict[str, str] = {}
    try:
        for path in sorted(inputs):
            source = physical_root / path
            if source.resolve() != source:
                raise GoMeasurementError("Go measurement inputs must not use symbolic links.")
            pins[path] = hashlib.sha256(source.read_bytes()).hexdigest()
    except OSError as exc:
        raise GoMeasurementError(
            "Every submitted Go source and parser input must be readable."
        ) from exc
    return SourceSnapshot(tuple(paths), pins, _digest(pins), _digest(list(paths)))


def validate_summary(raw: str, *, files: int) -> dict[str, object]:
    """Validate the native JSON protocol against the exact submitted file count.

    Raises
    ------
    GoMeasurementError
        If schema, count types, cohort size or finding totals are inconsistent.
    """
    if isinstance(files, bool) or files <= 0:
        raise GoMeasurementError("The submitted Go file count must be a positive integer.")
    try:
        value: object = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise GoMeasurementError("The Go parser must return valid summary JSON.") from exc
    if not isinstance(value, dict) or value.get("schema_version") != GO_COVERAGE_SCHEMA_VERSION:
        raise GoMeasurementError("The Go parser summary has an unsupported schema.")
    data = cast(dict[str, object], value)
    counts: dict[str, int] = {}
    for key in _COUNTS:
        count = data.get(key)
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise GoMeasurementError("Go parser counts must be nonnegative integers.")
        counts[key] = count
    kinds = data.get("by_kind")
    if not isinstance(kinds, dict):
        raise GoMeasurementError("The Go parser must classify its findings.")
    by_kind: dict[str, int] = {}
    for key, count in kinds.items():
        if key not in _KINDS or isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise GoMeasurementError("Go parser finding kinds and counts are invalid.")
        by_kind[key] = count
    hashes = data.get("source_sha256")
    if (
        not isinstance(hashes, dict)
        or len(hashes) != files
        or any(
            not isinstance(path, str)
            or not isinstance(digest, str)
            or re.fullmatch(r"[0-9a-f]{64}", digest) is None
            for path, digest in hashes.items()
        )
    ):
        raise GoMeasurementError("The native Go source hashes must cover every submitted file.")
    debt = counts["undocumented"]
    affected = counts["files_with_findings"]
    packages = counts["packages_total"]
    missing_packages = counts["packages_undocumented"]
    if (
        counts["files_scanned"] != files
        or not 1 <= packages <= files
        or affected > min(files, debt)
        or (affected == 0) != (debt == 0)
        or missing_packages > packages
        or by_kind.get("package", 0) != missing_packages
        or sum(by_kind.values()) != debt
    ):
        raise GoMeasurementError("The Go parser summary disagrees with its cohort or findings.")
    declarations = data.get("declarations")
    if not isinstance(declarations, list) or len(declarations) < debt:
        raise GoMeasurementError(
            "The native Go summary must retain all exported declaration identities."
        )
    return {
        "schema_version": GO_COVERAGE_SCHEMA_VERSION,
        "declarations": declarations,
        **counts,
        "by_kind": by_kind,
        "source_sha256": dict(hashes),
    }


def measure_source(root: Path, paths: Sequence[str], *, tracked: bool = False) -> GoDocMeasurement:
    """Measure real sources and reject observed input or tool-version drift."""
    before = capture_source(root, paths)
    version = _run(root, ["go", "version"]).strip()
    if not version.startswith("go version go"):
        raise GoMeasurementError("The Go toolchain returned no usable native version.")
    raw = _run(root, ["go", "run", GO_COVERAGE_TOOL], input_text="\n".join(paths))
    summary = validate_summary(raw, files=len(paths))
    if summary["source_sha256"] != {path: before.pins[path] for path in paths}:
        raise GoMeasurementError(
            "Native Go parsed bytes disagree with the observed source snapshot."
        )
    measurement = GoDocMeasurement(summary, version, before)
    measurement.verify(root, tracked=tracked)
    return measurement


def measure_go_coverage(root: Path, paths: Sequence[str]) -> dict[str, object]:
    """Return a validated native summary for the exact submitted Go source bytes."""
    return measure_source(root, paths).summary


def qualify_repository(root: Path) -> GoDocMeasurement:
    """Measure tracked and nonignored untracked Go with unchanged native enumeration."""
    paths = go_scope(root)
    return measure_source(root, paths, tracked=True)
