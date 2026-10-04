# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — native individual Go documentation findings

"""Validate individual native Go findings and bind them to actual parsed source bytes."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path, PurePosixPath
import subprocess
import tempfile
from typing import cast

from tools.go_doc_measurement import (
    GO_COVERAGE_TOOL,
    GoDocMeasurement,
    GoMeasurementError,
    capture_source,
    validate_summary,
)


@dataclass(frozen=True, order=True)
class DebtCase:
    """Identify one declaration independently of line numbers or package-doc placement."""

    source: str
    kind: str
    package: str
    receiver: str
    name: str

    def to_public_dict(self) -> dict[str, str]:
        """Serialize the declaration identity used by the protected debt allowance."""
        return {
            "source": self.source,
            "kind": self.kind,
            "package": self.package,
            "receiver": self.receiver,
            "name": self.name,
        }


@dataclass(frozen=True)
class FindingMeasurement:
    """Bind native individual cases to source bytes and the observed producer bytes."""

    measurement: GoDocMeasurement
    cases: frozenset[DebtCase]
    declarations: frozenset[DebtCase]
    producer_sha256: str
    argv: tuple[str, ...]

    def to_public_dict(self) -> dict[str, object]:
        """Record native source proof, exact compiler argv and individual case identities."""
        measured = self.measurement.to_public_dict()
        measured["argv"] = list(self.argv)
        return {
            "measurement": measured,
            "cases": [case.to_public_dict() for case in sorted(self.cases)],
            "declarations": [case.to_public_dict() for case in sorted(self.declarations)],
            "producer_sha256": self.producer_sha256,
            "argv": list(self.argv),
        }


def parse_findings(
    raw: str, paths: Sequence[str], summary: dict[str, object], *, check_totals: bool = True
) -> frozenset[DebtCase]:
    """Validate individual native findings against the measured cohort and totals.

    Parameters
    ----------
    raw : str
        Native JSON array of declaration records.
    paths : sequence of str
        Exact submitted source cohort.
    summary : dict of str to object
        Validated native debt counts associated with this source cohort.
    check_totals : bool, optional
        Compare debt totals; disable only for the full retained declaration array.

    Raises
    ------
    GoMeasurementError
        If fields, declaration identities, file coverage or finding totals disagree.
    """
    try:
        value: object = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise GoMeasurementError("The native Go findings must be valid JSON.") from exc
    if not isinstance(value, list):
        raise GoMeasurementError("The native Go findings must be a JSON array.")
    cases: set[DebtCase] = set()
    files: set[str] = set()
    kinds: Counter[str] = Counter()
    for item in value:
        if not isinstance(item, dict):
            raise GoMeasurementError("Each native Go finding must be an object.")
        entry = cast(dict[str, object], item)
        for key in ("file", "kind", "package", "name", "receiver"):
            if not isinstance(entry.get(key), str) or (key != "receiver" and not entry[key]):
                raise GoMeasurementError("The native Go finding has invalid text fields.")
        path, kind, package, name, receiver = (
            cast(str, entry[key]) for key in ("file", "kind", "package", "name", "receiver")
        )
        line = entry.get("line")
        if (
            isinstance(line, bool)
            or not isinstance(line, int)
            or line <= 0
            or path not in paths
            or kind not in {"func", "method", "type", "const", "var", "package"}
            or (kind == "method") != bool(receiver)
            or (kind == "package" and name != package)
        ):
            raise GoMeasurementError("The native Go finding disagrees with its declaration scope.")
        source = PurePosixPath(path).parent.as_posix() if kind == "package" else path
        case = DebtCase(source, kind, package, receiver, name)
        if case in cases:
            raise GoMeasurementError("Native Go declaration identities must be unique.")
        cases.add(case)
        files.add(path)
        kinds[kind] += 1
    if check_totals and (
        len(cases) != summary["undocumented"]
        or len(files) != summary["files_with_findings"]
        or dict(kinds) != summary["by_kind"]
    ):
        raise GoMeasurementError("Individual Go findings disagree with the native summary.")
    return frozenset(cases)


def measure_findings(
    root: Path, paths: Sequence[str], *, producer_root: Path | None = None
) -> FindingMeasurement:
    """Run the real Go parser with a separate immutable source image when requested.

    The producer remains explicitly observed and hashed. Native source hashes
    identify the exact bytes passed to the Go parser, including during input
    changes that before/after filesystem observations alone cannot exclude.
    """
    before = capture_source(root, paths)
    producer = ((producer_root or root) / GO_COVERAGE_TOOL).resolve()
    try:
        producer_sha = hashlib.sha256(producer.read_bytes()).hexdigest()
        version = subprocess.run(
            ["go", "version"], cwd=root, capture_output=True, text=True, check=False, timeout=120
        )
        if version.returncode != 0 or not version.stdout.strip().startswith("go version go"):
            raise GoMeasurementError("The native Go version could not be qualified.")
        with tempfile.TemporaryDirectory(prefix="go-doc-findings-") as folder:
            output = Path(folder) / "findings.json"
            argv = ("go", "run", str(producer), "-findings", str(output))
            result = subprocess.run(
                argv,
                cwd=root,
                input="\n".join(paths),
                capture_output=True,
                text=True,
                check=False,
                timeout=1800,
            )
            if result.returncode != 0:
                raise GoMeasurementError("The native Go individual findings command failed.")
            summary = validate_summary(result.stdout, files=len(paths))
            cases = parse_findings(output.read_text(encoding="utf-8"), paths, summary)
            declarations = parse_findings(
                json.dumps(summary["declarations"]), paths, summary, check_totals=False
            )
            if not cases <= declarations:
                raise GoMeasurementError(
                    "Native Go findings must belong to the retained declaration cohort."
                )
    except (OSError, subprocess.SubprocessError, UnicodeError) as exc:
        raise GoMeasurementError("Native Go individual findings could not be read.") from exc
    if summary["source_sha256"] != {path: before.pins[path] for path in paths}:
        raise GoMeasurementError("The native Go findings parsed different source bytes.")
    measurement = GoDocMeasurement(summary, version.stdout.strip(), before)
    measurement.verify(root)
    try:
        final_producer_sha = hashlib.sha256(producer.read_bytes()).hexdigest()
    except OSError as exc:
        raise GoMeasurementError("The Go findings producer is no longer readable.") from exc
    if final_producer_sha != producer_sha:
        raise GoMeasurementError("The Go findings producer changed during measurement.")
    return FindingMeasurement(measurement, cases, declarations, producer_sha, argv)
