# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — immutable Go documentation baseline and individual debt

"""Protect original Git source cohorts, scalar ceilings and individual Go debt cases."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import tarfile
import tempfile
from typing import cast

from tools.docstring_policy_git import baseline_commit, git_bytes
from tools.docstring_policy_scope import PolicyError
from tools.go_doc_findings import DebtCase, FindingMeasurement, measure_findings
from tools.go_doc_measurement import GoMeasurementError

LEGACY_CEILING_SCHEMA = "sc-neurocore.go-doc-ceiling.v1"
INDIVIDUAL_CEILING_SCHEMA = "sc-neurocore.go-doc-ceiling.v2"
_CONFIGS = {"go.mod", "go.sum", "go.work", "go.work.sum"}


@dataclass(frozen=True)
class ProtectedDebt:
    """Record original native measurements and their immutable source revision."""

    revision: str
    original: FindingMeasurement
    original_ceiling: int | None
    original_ceiling_sha256: str | None

    def to_public_dict(self) -> dict[str, object]:
        """Serialize reproducible original-source and scalar-ceiling evidence."""
        return {
            "git_revision": self.revision,
            "original": self.original.to_public_dict(),
            "original_ceiling": self.original_ceiling,
            "original_ceiling_sha256": self.original_ceiling_sha256,
        }


def _object(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise GoMeasurementError("The Go ceiling must be a JSON object.")
    return cast(dict[str, object], value)


def _ceiling_document(raw: bytes) -> dict[str, object]:
    try:
        data = _object(json.loads(raw))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise GoMeasurementError("The Go ceiling must be valid JSON.") from exc
    count = data.get("undocumented")
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise GoMeasurementError("The Go ceiling must contain a nonnegative integer.")
    if data.get("schema_version") not in (None, LEGACY_CEILING_SCHEMA, INDIVIDUAL_CEILING_SCHEMA):
        raise GoMeasurementError("The Go ceiling schema is unsupported.")
    return data


def _record_cases(document: dict[str, object]) -> frozenset[DebtCase] | None:
    if document.get("schema_version") != INDIVIDUAL_CEILING_SCHEMA:
        return None
    provenance = _object(document.get("provenance"))
    raw = provenance.get("debt_cases")
    if not isinstance(raw, list):
        raise GoMeasurementError("Individual Go ceilings require a case array.")
    cases: set[DebtCase] = set()
    for value in raw:
        entry = _object(value)
        if set(entry) != {"source", "kind", "package", "receiver", "name"}:
            raise GoMeasurementError("The Go ceiling case has an invalid identity shape.")
        if any(not isinstance(entry[k], str) for k in entry):
            raise GoMeasurementError("Go ceiling case fields must be strings.")
        case = DebtCase(**cast(dict[str, str], entry))
        if case in cases:
            raise GoMeasurementError("Go ceiling cases must be unique.")
        cases.add(case)
    if len(cases) != document["undocumented"]:
        raise GoMeasurementError("The Go ceiling case count disagrees with its scalar.")
    return frozenset(cases)


def _original_source(root: Path, revision: str, names: list[str], image: Path) -> None:
    archive = git_bytes(root, "archive", "--format=tar", revision, "--", *names)
    with tarfile.open(fileobj=io.BytesIO(archive)) as stream:
        for name in names:
            member = stream.getmember(name)
            if (
                not member.isfile()
                or PurePosixPath(name).is_absolute()
                or ".." in PurePosixPath(name).parts
            ):
                raise GoMeasurementError("Original Go sources must be regular repository files.")
            source = stream.extractfile(member)
            if source is None:
                raise GoMeasurementError("An original Go source could not be read.")
            target = image / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(source.read())


def protect_debt(
    root: Path, current: FindingMeasurement, ceiling: Path, *, allow_create: bool = False
) -> ProtectedDebt:
    """Reject original cohort shrinkage, ceiling inflation and new individual debt.

    Parameters
    ----------
    root : Path
        Exact repository whose original Git history establishes the baseline.
    current : FindingMeasurement
        Actual current native individual findings with parsed-source hashes.
    ceiling : Path
        Candidate ceiling; a custom path cannot create new declaration allowances.
    allow_create : bool
        Permit an explicit update to initialize a missing ceiling from measured debt.

    Returns
    -------
    ProtectedDebt
        Original immutable source revision and its independently measured debt.

    Raises
    ------
    GoMeasurementError
        If native Git provenance, original measurements or debt protection fail.
    """
    try:
        revision = baseline_commit(root, os.environ)
        raw_names = git_bytes(root, "ls-tree", "-r", "-z", "--name-only", revision)
        names = [name.decode("utf-8") for name in raw_names.split(b"\0") if name]
        paths = sorted(name for name in names if name.endswith(".go"))
        if not set(paths) <= set(current.measurement.source.paths):
            raise GoMeasurementError("The original Go source cohort must not shrink.")
        try:
            ceiling_name = ceiling.resolve().relative_to(root.resolve()).as_posix()
        except ValueError:
            ceiling_name = ""
        original_raw = (
            git_bytes(root, "show", revision + ":" + ceiling_name)
            if ceiling_name in names
            else None
        )
        original_document = _ceiling_document(original_raw) if original_raw is not None else None
        with tempfile.TemporaryDirectory(prefix="go-doc-original-") as folder:
            image = Path(folder)
            _original_source(root, revision, paths + sorted(_CONFIGS & set(names)), image)
            original = measure_findings(image, paths, producer_root=root)
    except (PolicyError, OSError, UnicodeError, tarfile.TarError, KeyError) as exc:
        raise GoMeasurementError("The original Go source revision could not be qualified.") from exc
    original_ceiling: int | None = None
    if original_document is not None:
        original_ceiling = cast(int, original_document["undocumented"])
        if original.measurement.undocumented != original_ceiling:
            raise GoMeasurementError(
                "The original Go native measurement does not reproduce its ceiling."
            )
        original_cases = _record_cases(original_document)
        if original_cases is not None and original_cases != original.cases:
            raise GoMeasurementError("The original Go native cases do not reproduce their ceiling.")
        provenance = original_document.get("provenance")
        if (
            isinstance(provenance, dict)
            and "go" in provenance
            and provenance["go"] != original.measurement.version
        ):
            raise GoMeasurementError(
                "The native Go version differs from the original ceiling producer."
            )
    if allow_create and not ceiling.exists():
        candidate: dict[str, object] = {"undocumented": original.measurement.undocumented}
    else:
        try:
            candidate = _ceiling_document(ceiling.read_bytes())
        except OSError as exc:
            raise GoMeasurementError(
                "no ceiling record is available for Go documentation."
            ) from exc
    if original_ceiling is not None and cast(int, candidate["undocumented"]) > original_ceiling:
        raise GoMeasurementError(
            "The candidate Go ceiling must not exceed its original Git ceiling."
        )
    candidate_cases = _record_cases(candidate)
    if (
        original_document is not None
        and _record_cases(original_document) is not None
        and candidate_cases is None
    ):
        raise GoMeasurementError("An individual Go ceiling cannot revert to an aggregate protocol.")
    if candidate_cases is not None and not candidate_cases <= original.cases:
        raise GoMeasurementError("The candidate Go ceiling adds original debt allowances.")
    allowed = original.cases if candidate_cases is None else candidate_cases
    new_cases = current.cases - allowed
    if new_cases:
        raise GoMeasurementError(
            "New undocumented Go declarations are not allowed: "
            + "; ".join(f"{case.source}:{case.receiver}.{case.name}" for case in sorted(new_cases))
        )
    if not original.cases <= current.declarations:
        raise GoMeasurementError(
            "Original undocumented Go declarations must remain in the source cohort."
        )
    return ProtectedDebt(
        revision,
        original,
        original_ceiling,
        hashlib.sha256(original_raw).hexdigest() if original_raw is not None else None,
    )
