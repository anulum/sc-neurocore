# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — External formal receipt consistency checks

"""Check the exact real historical recovery and refuse altered native facts."""

from __future__ import annotations

import json
from pathlib import Path
import shutil

import pytest

from tools.formal_evidence_validation import digest, object_fields, read_object
from tools.formal_execution_accounting import EXCEPTION_REGISTRY, validate_execution_accounting

ROOT = Path(__file__).resolve().parents[2]
RECEIPT = "benchmarks/results/formal/sc_amari_field/receipt.json"
ACTUAL_RECEIPT_SHA256 = "188b5cd2741c55b2bcc4b3331971c263a9c30a05b9b1948a0f48ab463471bd75"


@pytest.fixture
def historical(tmp_path: Path) -> tuple[Path, dict[str, object]]:
    """Copy only the real Amari receipt, normalized log, status and exact registry."""
    for name in (
        RECEIPT,
        EXCEPTION_REGISTRY,
        "benchmarks/results/formal/sc_amari_field/native.log",
        "benchmarks/results/formal/sc_amari_field/status.txt",
    ):
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
    assert digest(tmp_path / RECEIPT) == ACTUAL_RECEIPT_SHA256
    return tmp_path, read_object(tmp_path / RECEIPT)


def test_actual_amari_recovered_accounting(historical: tuple[Path, dict[str, object]]) -> None:
    """Admit only the actual receipt's recorded journal clock and explicit exception."""
    repository, receipt = historical
    wall, exception, registry_sha = validate_execution_accounting(
        repository, receipt, ACTUAL_RECEIPT_SHA256
    )
    assert wall == 3382.523562
    assert exception == "amari-gc-20261101"
    assert registry_sha == digest(repository / EXCEPTION_REGISTRY)
    assert object_fields(receipt["measurement"])["wall_seconds"] is None


@pytest.mark.parametrize(
    "field,value,reason",
    [
        ("original_result_sha256", "0" * 64, "original_result"),
        ("journal_sha256", "0" * 64, "journal_sha256"),
        ("invocation_id", "0" * 32, "invocation_id"),
        ("unit_exit_code", 0, "cannot claim retained"),
        ("unit_result", "success", "cannot claim retained"),
        ("native_exit_source", "systemd-exec", "cannot claim retained"),
        ("native_exit_code", False, "exit code"),
        ("wall_seconds", 3382.523562, "must stay null"),
        ("wall_clock_source", "systemd-exec-monotonic", "journal-recovered"),
        ("journal_recovered_wall_seconds", 3382, "monotonic stamps"),
        ("start_monotonic_microseconds", None, "clock stamps"),
        ("stop_monotonic_microseconds", 0, "clock stamps"),
        ("native_done", {"normalized_line": "DONE (FAIL, rc=2)"}, "DONE line is invalid"),
    ],
)
def test_changed_historical_measurement_refuses(
    historical: tuple[Path, dict[str, object]], field: str, value: object, reason: str
) -> None:
    """Refuse isolated changes while preserving the real outer registry identity."""
    repository, receipt = historical
    measurement = object_fields(receipt["measurement"])
    measurement[field] = value
    receipt["measurement"] = measurement
    with pytest.raises(ValueError, match=reason):
        validate_execution_accounting(repository, receipt, ACTUAL_RECEIPT_SHA256)


@pytest.mark.parametrize(
    "field,value",
    [
        ("raw_line_sha256", None),
        ("raw_line_sha256", "invalid"),
        ("normalized_line_sha256", "0" * 64),
    ],
)
def test_changed_done_custody_refuses(
    historical: tuple[Path, dict[str, object]], field: str, value: object
) -> None:
    """A PASS token cannot replace a matching actual line and both custody hashes."""
    repository, receipt = historical
    measurement = object_fields(receipt["measurement"])
    done = object_fields(measurement["native_done"])
    done[field] = value
    measurement["native_done"] = done
    receipt["measurement"] = measurement
    with pytest.raises(ValueError, match="DONE line digest"):
        validate_execution_accounting(repository, receipt, ACTUAL_RECEIPT_SHA256)


def test_absent_actual_done_refuses(historical: tuple[Path, dict[str, object]]) -> None:
    """Refuse an attested terminal line removed from the actual normalized log."""
    repository, receipt = historical
    log = repository / str(object_fields(receipt["log"])["path"])
    log.write_text(log.read_text().replace("DONE (PASS, rc=0)", "missing"))
    with pytest.raises(ValueError, match="DONE line is absent"):
        validate_execution_accounting(repository, receipt, ACTUAL_RECEIPT_SHA256)


@pytest.mark.parametrize("content", ["", "FAIL 2 1\n"])
def test_changed_actual_status_refuses(
    historical: tuple[Path, dict[str, object]], content: str
) -> None:
    """Reject an empty or failing native status even when its test digest is refreshed."""
    repository, receipt = historical
    measurement = object_fields(receipt["measurement"])
    status = object_fields(measurement["status_file"])
    path = repository / str(status["path"])
    path.write_text(content)
    status["sha256"] = digest(path)
    measurement["status_file"] = status
    receipt["measurement"] = measurement
    with pytest.raises(ValueError, match="SBY status file"):
        validate_execution_accounting(repository, receipt, ACTUAL_RECEIPT_SHA256)


@pytest.mark.parametrize("change", ["different_receipt", "schema", "identifier"])
def test_registry_cannot_extend_historical_acceptance(
    historical: tuple[Path, dict[str, object]], change: str
) -> None:
    """A second receipt or malformed registry has no inherited recovery permission."""
    repository, receipt = historical
    sha = ACTUAL_RECEIPT_SHA256
    if change == "different_receipt":
        sha = "0" * 64
    else:
        path = repository / EXCEPTION_REGISTRY
        registry = read_object(path)
        if change == "schema":
            registry["schema_version"] = 2
        else:
            entries = object_fields(registry["receipts"])
            entry = object_fields(entries[sha])
            entry["id"] = ""
            entries[sha] = entry
            registry["receipts"] = entries
        path.write_text(json.dumps(registry))
    with pytest.raises(ValueError):
        validate_execution_accounting(repository, receipt, sha)
