# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Measured function resources and numerical error reports

"""Combine admitted mappings with separately measured numerical error and domains."""

from __future__ import annotations

import hashlib
import os
import stat
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from filelock import Timeout

from sc_neurocore.compiler import expr_lut_tables, lut_accuracy
from tools.function_synthesis_cache import (
    FunctionSynthesisAttemptError,
    FunctionSynthesisCache,
)
from tools.function_synthesis_records import SynthesisCase, synthesis_cases


def _digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _sources(cache: FunctionSynthesisCache) -> dict[str, str]:
    sources = cache.source_bindings
    root = Path(__file__).resolve().parents[1]
    for name in (
        "tools/function_synthesis_report.py",
        "src/sc_neurocore/neurons/equation_namespace.py",
    ):
        sources[name] = _digest(root / name)
    return sources


def _attempt_artifacts(directory: Path) -> dict[str, str]:
    """Hash regular failed-attempt artifacts while refusing linked or special evidence."""
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("A failed synthesis attempt requires its original directory")
    artifacts: dict[str, str] = {}
    for path in sorted(directory.rglob("*")):
        mode = path.lstat().st_mode
        if stat.S_ISDIR(mode):
            continue
        if not stat.S_ISREG(mode):
            raise ValueError("Failed synthesis artifacts must be regular unlinked files")
        with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW), "rb") as stream:
            artifacts[path.relative_to(directory).as_posix()] = hashlib.file_digest(
                stream, "sha256"
            ).hexdigest()
    return artifacts


def _row(
    case: SynthesisCase,
    cache: FunctionSynthesisCache,
    measure_missing: bool,
    synthesis_timeout: float,
    proof_timeout: float,
    lock_timeout: float,
) -> dict[str, object]:
    row: dict[str, object] = {
        "case": case.key,
        "function": case.function,
        "lowering": case.lowering,
        "data_width": case.data_width,
        "fraction": case.fraction,
        "hardware": None,
        "accuracy": None,
    }
    if not case.supported:
        row.update(status="unsupported", reason="No implementation in this lowering")
        return row
    try:
        result = (
            cache.measure(
                case,
                synthesis_timeout=synthesis_timeout,
                proof_timeout=proof_timeout,
                lock_timeout=lock_timeout,
            )
            if measure_missing
            else cache.get(case)
        )
        if result is None:
            row["status"] = "missing"
            return row
        shared_exact = (
            case.lowering == "lut-v1" and case.function in expr_lut_tables.EXACT_FUNCTIONS
        )
        assessment = lut_accuracy.measure(
            case.function,
            case.data_width,
            case.fraction,
            lowering="v2" if shared_exact else case.lowering,
        )
        row.update(
            status="measured",
            hardware=result.measurement.to_dict(),
            accuracy=assessment.to_public_dict(),
            assessment_basis="shared_exact_arithmetic" if shared_exact else "selected_lowering",
            cache_identity=result.identity,
            reused=result.reused,
            recorded_at=result.recorded_at.isoformat(),
        )
    except FunctionSynthesisAttemptError as error:
        try:
            artifacts = _attempt_artifacts(error.directory)
        except Exception:
            row["status"] = "evidence_rejected"
        else:
            row.update(
                status="timeout" if error.timed_out else "tool_failure",
                returncode=error.returncode,
                attempt_artifact_sha256=artifacts,
            )
    except Timeout:
        row["status"] = "writer_timeout"
    except Exception:
        row["status"] = "evidence_rejected"
    return row


def build_function_synthesis_report(
    cache: FunctionSynthesisCache,
    *,
    cases: Sequence[SynthesisCase] | None = None,
    measure_missing: bool = False,
    synthesis_timeout: float = 240.0,
    proof_timeout: float = 90.0,
    lock_timeout: float = 600.0,
) -> dict[str, object]:
    """Report real mapping and error evidence without silently dropping failed cases.

    Parameters
    ----------
    cache : FunctionSynthesisCache
        Source-bound store using the actual inspected mapping and proof tools.
    cases : sequence of SynthesisCase or None, optional
        None selects the full comparison matrix. An explicit selection is always
        a diagnostic; it cannot establish completion of the official matrix.
    measure_missing : bool, optional
        Execute missing mappings when True. The default only reads retained
        successes and creates no native attempt for a missing measurement.
    synthesis_timeout, proof_timeout, lock_timeout : float, optional
        Native command and same-key writer deadlines, passed to the cache.

    Returns
    -------
    dict of str to object
        All requested dispositions, actual resources and artifact hashes,
        historical measurement times and separately assessed numerical errors.
        Missing, rejected and timed-out measurements have no hardware values.
        Unsupported legacy cases remain explicit unavailable rows. Only the
        entire official matrix with every supported case measured is complete.
        Sampled numerical maxima are not declared full-domain error bounds.

    Raises
    ------
    ValueError
        For an empty or duplicate selection, or a changing source checkout.
    OSError
        If owning provenance cannot be read.

    Notes
    -----
    The four legacy exact functions emit the same function-unit arithmetic as
    v2. Their assessment retains its actual v2 lowering and explicitly identifies
    this shared basis; it is not presented as a legacy lookup-table measurement.
    Mapping area is for xc7 without I/O or clock buffers. Logical table bits are
    source declarations, not physical BRAM counts or measured timing.

    """
    selected = synthesis_cases() if cases is None else tuple(cases)
    if not selected or len({case.key for case in selected}) != len(selected):
        raise ValueError("A synthesis report needs a nonempty unique case selection")
    sources = _sources(cache)
    rows = [
        _row(case, cache, measure_missing, synthesis_timeout, proof_timeout, lock_timeout)
        for case in selected
    ]
    if _sources(cache) != sources:
        raise ValueError("Report sources changed during numerical assessment")
    cache.tools.verify()
    counts = {
        status: sum(row["status"] == status for row in rows)
        for status in (
            "measured",
            "unsupported",
            "missing",
            "timeout",
            "tool_failure",
            "writer_timeout",
            "evidence_rejected",
        )
    }
    admitted = counts["measured"] + counts["unsupported"] == len(selected)
    return {
        "schema_version": "sc-neurocore.function-synthesis-report.v1",
        "scope": "complete_matrix" if cases is None else "diagnostic",
        "status": "complete" if admitted else "incomplete",
        "matrix_complete": cases is None and admitted,
        "generated_at": datetime.now(UTC).isoformat(),
        "target": "xc7",
        "source_sha256": sources,
        "toolchain": cache.tools.manifest(),
        "counts": counts,
        "cases": rows,
    }
