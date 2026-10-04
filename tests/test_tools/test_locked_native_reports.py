# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Native audit report validation contracts

"""Exercise public response parsers with untrusted npm and cargo-audit input data."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from tools.security_scan.locked_native_reports import (
    npm_lock_packages,
    npm_report_passed,
    rust_report_passed,
)

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def npm_report() -> dict[str, Any]:
    """Return the public npm report shape for direct untrusted-input contracts.

    Returns
    -------
    dict
        Version-2 response with complete counters for one installation identity.
        This input does not replace a scanner or assert registry audit success.
    """
    return {
        "auditReportVersion": 2,
        "vulnerabilities": {},
        "metadata": {
            "vulnerabilities": {
                "info": 0,
                "low": 0,
                "moderate": 0,
                "high": 0,
                "critical": 0,
                "total": 0,
            },
            "dependencies": {"total": 1},
        },
    }


@pytest.fixture
def rust_report() -> dict[str, Any]:
    """Return the cargo-audit response shape for direct untrusted-input contracts.

    Returns
    -------
    dict
        Unsuppressed findings, all warning categories and unfiltered settings.
        Acceptance here tests report semantics rather than scanner execution.
    """
    return {
        "settings": {
            "ignore": [],
            "target_arch": [],
            "target_os": [],
            "severity": None,
            "informational_warnings": ["unmaintained", "unsound", "notice"],
        },
        "vulnerabilities": {"found": False, "count": 0, "list": []},
        "warnings": {},
    }


def test_npm_inventory_binds_every_actual_studio_installation() -> None:
    """Bind every maintained registry entry, including scoped and repeated packages."""
    raw = (ROOT / "studio/frontend/package-lock.json").read_bytes()
    identities = npm_lock_packages(raw)
    lock = json.loads(raw)
    assert len(identities) == len(lock["packages"]) - 1
    assert {row["path"] for row in identities} == set(lock["packages"]) - {""}
    for row in identities:
        entry = lock["packages"][row["path"]]
        assert row["version"] == entry["version"]
        assert row["name"] == entry.get("name", row["path"].rsplit("node_modules/", 1)[-1])


@pytest.mark.parametrize(
    "raw",
    [
        b"[]",
        b"{}",
        b'{"lockfileVersion":true}',
        b'{"lockfileVersion":1,"packages":{}}',
        b'{"lockfileVersion":3,"packages":{}}',
    ],
)
def test_npm_refuses_an_unavailable_lock_inventory(raw: bytes) -> None:
    """Missing lock schema and root identity cannot establish complete package coverage.

    Parameters
    ----------
    raw : bytes
        Malformed or unsupported serialized public lock input.
    """
    with pytest.raises(ValueError):
        npm_lock_packages(raw)


@pytest.mark.parametrize(
    "section,value",
    [
        ("auditReportVersion", False),
        ("auditReportVersion", 1),
        ("metadata", []),
        ("vulnerabilities", []),
    ],
)
def test_npm_refuses_incomplete_report_sections(
    npm_report: dict[str, Any], section: str, value: object
) -> None:
    """Refuse missing report semantics even when no advisory list is populated.

    Parameters
    ----------
    npm_report : dict
        Public response input containing valid severity counters.
    section : str
        Report field whose type or version is invalid.
    value : object
        Untrusted value to reject through the public parser.
    """
    npm_report[section] = value
    with pytest.raises(ValueError):
        npm_report_passed(npm_report, 1)


@pytest.mark.parametrize("value", [False, -1, 0.0, "0", None])
def test_npm_refuses_invalid_severity_counters(npm_report: dict[str, Any], value: object) -> None:
    """Boolean, negative and noninteger counters cannot imitate an audited zero.

    Parameters
    ----------
    npm_report : dict
        Public report input with complete metadata.
    value : object
        Invalid counter supplied to the severity validation boundary.
    """
    npm_report["metadata"]["vulnerabilities"]["total"] = value
    with pytest.raises(ValueError, match="counters"):
        npm_report_passed(npm_report, 1)


@pytest.mark.parametrize("value", [False, 0, 2, None, "1"])
def test_npm_refuses_incomplete_package_counts(npm_report: dict[str, Any], value: object) -> None:
    """Require an integer report count covering the exact validated lock inventory.

    Parameters
    ----------
    npm_report : dict
        Public report input whose advisory counters are clean.
    value : object
        Incorrect or unavailable scanner dependency count.
    """
    npm_report["metadata"]["dependencies"]["total"] = value
    with pytest.raises(ValueError, match="complete lock"):
        npm_report_passed(npm_report, 1)


@pytest.mark.parametrize("severity", ["info", "low", "moderate", "high", "critical"])
def test_npm_blocks_findings_at_every_severity(npm_report: dict[str, Any], severity: str) -> None:
    """Block advisory findings regardless of an npm process exit threshold.

    Parameters
    ----------
    npm_report : dict
        Complete public response input for one locked package.
    severity : str
        Advisory severity whose finding must block acceptance.
    """
    assert npm_report_passed(npm_report, 1) is True
    npm_report["metadata"]["vulnerabilities"][severity] = 1
    with pytest.raises(ValueError, match="inconsistent"):
        npm_report_passed(npm_report, 1)
    npm_report["metadata"]["vulnerabilities"]["total"] = 1
    assert npm_report_passed(npm_report, 1) is False


def test_npm_blocks_nonempty_findings_and_provider_errors(npm_report: dict[str, Any]) -> None:
    """An empty counter cannot clear an advisory object or a registry error.

    Parameters
    ----------
    npm_report : dict
        Complete response input with valid zero counters.
    """
    npm_report["vulnerabilities"] = {"example": {"severity": "low"}}
    assert npm_report_passed(npm_report, 1) is False
    npm_report["vulnerabilities"] = {}
    npm_report["error"] = {"code": "EINVALID"}
    assert npm_report_passed(npm_report, 1) is False


@pytest.mark.parametrize(
    "key,value",
    [
        ("ignore", ["RUSTSEC-EXAMPLE"]),
        ("target_arch", ["x86_64"]),
        ("target_os", ["linux"]),
        ("severity", "high"),
        ("informational_warnings", ["notice"]),
    ],
)
def test_rust_refuses_suppressed_advisory_coverage(
    rust_report: dict[str, Any], key: str, value: object
) -> None:
    """Settings must expose every advisory and all supported warning categories.

    Parameters
    ----------
    rust_report : dict
        Public unfiltered report input with complete finding counters.
    key : str
        Setting that can narrow advisory or informational coverage.
    value : object
        Filtering or suppression setting to reject.
    """
    assert rust_report_passed(rust_report) is True
    rust_report["settings"][key] = value
    with pytest.raises(ValueError):
        rust_report_passed(rust_report)


@pytest.mark.parametrize(
    "key,value",
    [("found", 0), ("count", False), ("count", -1), ("count", 1), ("found", True), ("list", {})],
)
def test_rust_refuses_invalid_finding_counters(
    rust_report: dict[str, Any], key: str, value: object
) -> None:
    """Refuse boolean counters, wrong types and inconsistent finding cardinality.

    Parameters
    ----------
    rust_report : dict
        Public report input whose settings are unsuppressed.
    key : str
        Finding field whose type or consistency is invalid.
    value : object
        Untrusted finding value passed to the public validator.
    """
    rust_report["vulnerabilities"][key] = value
    with pytest.raises(ValueError):
        rust_report_passed(rust_report)


def test_rust_blocks_valid_advisories_and_informational_warnings(
    rust_report: dict[str, Any],
) -> None:
    """A complete advisory or informational warning always blocks acceptance.

    Parameters
    ----------
    rust_report : dict
        Public report input retaining all warning categories.
    """
    rust_report["warnings"] = {"notice": [{}]}
    assert rust_report_passed(rust_report) is False
    rust_report["warnings"] = {}
    rust_report["vulnerabilities"] = {"found": True, "count": 1, "list": [{}]}
    assert rust_report_passed(rust_report) is False


@pytest.mark.parametrize("payload", [None, [], "invalid"])
def test_rust_refuses_nonobject_report_roots(payload: object) -> None:
    """A JSON null, array or string cannot establish a valid Rust audit report.

    Parameters
    ----------
    payload : object
        Untrusted decoded JSON input supplied to the public report boundary.
    """
    with pytest.raises(ValueError, match="JSON root"):
        rust_report_passed(payload)


@pytest.mark.parametrize("warnings", [None, {"notice": False}, {"notice": {}}])
def test_rust_refuses_unavailable_warning_lists(
    rust_report: dict[str, Any], warnings: object
) -> None:
    """Require typed warning lists before accepting a clean advisory counter.

    Parameters
    ----------
    rust_report : dict
        Public response input whose advisory counters and settings are complete.
    warnings : object
        Absent or malformed warning data to refuse through the public parser.
    """
    rust_report["warnings"] = warnings
    with pytest.raises(ValueError, match="warning lists"):
        rust_report_passed(rust_report)
