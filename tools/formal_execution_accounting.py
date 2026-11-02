# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — External formal receipt consistency checks

"""Require retained native accounting or an exact reviewed historical receipt."""

from __future__ import annotations

import hashlib
from pathlib import Path
import re

from tools.formal_evidence_validation import (
    confined_file,
    digest,
    finite_number,
    object_fields,
    read_object,
    require,
)

EXCEPTION_REGISTRY = "benchmarks/results/formal/exceptions.json"


def validate_execution_accounting(
    repository: Path, receipt: dict[str, object], receipt_sha256: str
) -> tuple[float, str | None, str | None]:
    """Validate actual exit custody and a declared monotonic wall clock.

    Parameters
    ----------
    repository : Path
        Checkout with immutable technical evidence and any exception registry.
    receipt : dict of str to object
        Actual native receipt.
    receipt_sha256 : str
        Digest of the exact receipt bytes, never a job-wide exception key.

    Returns
    -------
    tuple of float, str or None, str or None
        Actual wall, accepted exception identifier and registry digest. The
        latter two are absent for retained native execution accounting.

    Raises
    ------
    ValueError
        If native accounting is incomplete or a historical recovery is not
        registered against the exact receipt, original custody and invocation.
    """
    measurement = object_fields(receipt.get("measurement"))
    native_code = measurement.get("native_exit_code")
    require(type(native_code) is int and native_code == 0, "Native exit code does not qualify.")
    exception_id: object = None
    registry_digest: str | None = None
    if measurement.get("unit_accounting") == "garbage-collected":
        registry_path = confined_file(repository, EXCEPTION_REGISTRY)
        registry = read_object(registry_path)
        require(
            type(registry.get("schema_version")) is int and registry["schema_version"] == 1,
            "Unknown execution exception registry.",
        )
        entries = object_fields(registry.get("receipts"))
        exception = object_fields(entries.get(receipt_sha256))
        exception_id = exception.get("id")
        require(
            isinstance(exception_id, str) and bool(exception_id.strip()),
            "Historical receipt exception identifier is missing.",
        )
        log = object_fields(receipt.get("log"))
        for field, actual in {
            "job": receipt.get("job"),
            "original_result_sha256": measurement.get("original_result_sha256"),
            "journal_sha256": measurement.get("journal_sha256"),
            "invocation_id": measurement.get("invocation_id"),
            "raw_log_sha256": log.get("raw_sha256"),
        }.items():
            require(
                exception.get(field) == actual and actual is not None,
                f"Historical exception {field} binding differs.",
            )
        require(
            measurement.get("unit_exit_code") is None
            and measurement.get("unit_result") is None
            and measurement.get("native_exit_source") == "sby-final-DONE",
            "Garbage-collected accounting cannot claim retained systemd exit facts.",
        )
        done = object_fields(measurement.get("native_done"))
        line, raw_line_digest = done.get("normalized_line"), done.get("raw_line_sha256")
        require(
            isinstance(line, str)
            and re.fullmatch(r"[^\r\n]*DONE \(PASS, rc=0\)", line) is not None,
            "Historical native DONE line is invalid.",
        )
        require(
            isinstance(raw_line_digest, str)
            and re.fullmatch(r"[0-9a-f]{64}", raw_line_digest) is not None,
            "Historical raw DONE line digest is invalid.",
        )
        public_log = confined_file(repository, log.get("path"))
        require(
            str(line) in public_log.read_text().splitlines(),
            "Historical native DONE line is absent.",
        )
        require(
            hashlib.sha256(str(line).encode()).hexdigest() == done.get("normalized_line_sha256"),
            "Historical normalized DONE line digest differs.",
        )
        status = object_fields(measurement.get("status_file"))
        status_path = confined_file(repository, status.get("path"))
        status_words = status_path.read_text().split()
        require(
            digest(status_path) == status.get("sha256")
            and bool(status_words)
            and status_words[0] == "PASS",
            "Historical native SBY status file does not qualify.",
        )
        require(
            "wall_seconds" in measurement
            and measurement["wall_seconds"] is None
            and measurement.get("wall_clock_source") == "journal-recovered",
            "Historical original wall must stay null and its clock journal-recovered.",
        )
        wall = finite_number(measurement.get("journal_recovered_wall_seconds"), "recovered wall")
        registry_digest = digest(registry_path)
    else:
        require(
            measurement.get("unit_accounting") == "retained", "Native unit accounting is missing."
        )
        unit_code = measurement.get("unit_exit_code")
        require(
            type(unit_code) is int
            and unit_code == 0
            and measurement.get("unit_result") == "success"
            and measurement.get("native_exit_source") == "systemd-exec",
            "Retained native systemd exit facts do not qualify.",
        )
        require(
            measurement.get("wall_clock_source") == "systemd-exec-monotonic",
            "Retained native monotonic clock is missing.",
        )
        wall = finite_number(measurement.get("wall_seconds"), "wall_seconds")
    start = measurement.get("start_monotonic_microseconds")
    stop = measurement.get("stop_monotonic_microseconds")
    require(
        type(start) is int and type(stop) is int and start > 0 and stop > start,
        "Actual monotonic clock stamps are invalid.",
    )
    require(
        wall == (int(str(stop)) - int(str(start))) / 1_000_000,
        "Native wall differs from its actual monotonic stamps.",
    )
    return wall, str(exception_id) if exception_id is not None else None, registry_digest
