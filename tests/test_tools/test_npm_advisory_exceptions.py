# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — npm advisory exceptions: dev-only packages, unreleased fixes, review dates

"""An npm advisory is accepted only for a dev-only package with no released fix, named and dated."""

from __future__ import annotations

from datetime import date, timedelta
import json
from pathlib import Path
import tomllib
from typing import Any

import pytest

from tools.security_scan.npm_advisory_exceptions import (
    EXCEPTION_FILE,
    NPM_REGISTRY,
    in_range,
    load_npm_exceptions,
    npm_audit_passed,
    npm_lock_decision,
    osv_npm_decision,
    released_fixes,
)

ROOT = Path(__file__).resolve().parents[2]
TODAY = date(2026, 11, 17)
LOCK = "studio/frontend/package-lock.json"
ADVISORY = "GHSA-fc5q-39xc-vjpf"
URL = f"https://github.com/advisories/{ADVISORY}"


def record(**changes: str) -> dict[str, str]:
    """Return one valid exception record with the given fields replaced."""
    row = {
        "lock": LOCK,
        "package": "picomatch",
        "version": "4.0.7",
        "advisory": ADVISORY,
        "vulnerable_range": "<=4.0.7",
        "reason": "dev_only_fix_unreleased",
        "review_by": "2026-12-01",
        "rationale": "Development-only glob matcher of the build tools; no fix is released.",
    }
    row.update(changes)
    return row


def table(*rows: dict[str, str]) -> bytes:
    """Serialise exception records as the TOML table the module reads."""
    lines: list[str] = []
    for row in rows:
        lines.append("[[exception]]")
        lines.extend(f"{key} = {json.dumps(value)}" for key, value in row.items())
        lines.append("")
    return ("\n".join(lines) or "exception = []\n").encode("utf-8")


def lock(**picomatch: Any) -> bytes:
    """Return a version-3 lock with a dev-only picomatch, a dev-only vite and a production react."""
    entry = {
        "version": "4.0.7",
        "resolved": "https://registry.npmjs.org/picomatch/-/picomatch-4.0.7.tgz",
        "dev": True,
    }
    entry.update(picomatch)
    packages = {
        "": {"name": "studio", "version": "1.0.0"},
        "node_modules/picomatch": entry,
        "node_modules/react": {
            "version": "19.0.0",
            "resolved": "https://registry.npmjs.org/react/-/react-19.0.0.tgz",
        },
        "node_modules/vite": {
            "version": "6.3.5",
            "resolved": "https://registry.npmjs.org/vite/-/vite-6.3.5.tgz",
            "dev": True,
        },
    }
    return json.dumps({"lockfileVersion": 3, "packages": packages}).encode("utf-8")


def report(*advisories: dict[str, Any], total: int = 3) -> dict[str, Any]:
    """Return a complete npm audit report whose findings stem from the given advisories."""
    findings: dict[str, Any] = {}
    for item in advisories:
        finding = findings.setdefault(
            item["name"], {"name": item["name"], "severity": "high", "via": []}
        )
        finding["via"].append(item)
    if "picomatch" in findings:
        findings["vite"] = {"name": "vite", "severity": "high", "via": ["picomatch"]}
    counts = {
        "info": 0,
        "low": 0,
        "moderate": 0,
        "high": len(findings),
        "critical": 0,
        "total": len(findings),
    }
    return {
        "auditReportVersion": 2,
        "vulnerabilities": findings,
        "metadata": {"vulnerabilities": counts, "dependencies": {"total": total}},
    }


def advisory(
    name: str = "picomatch", url: str | None = URL, vulnerable_range: str = "<=4.0.7"
) -> dict[str, Any]:
    """Return one advisory object as npm audit reports it in a finding's ``via`` list."""
    return {
        "source": 1247710,
        "name": name,
        "url": url,
        "severity": "high",
        "range": vulnerable_range,
    }


class Registry:
    """A reader that serves one npm registry document and records each address it is asked for."""

    def __init__(self, *versions: str) -> None:
        """Keep the versions the document lists."""
        self.versions = versions
        self.calls: list[str] = []

    def __call__(self, url: str, limit: int) -> bytes:
        """Return the document, after checking that the caller bounds the read."""
        self.calls.append(url)
        assert limit > 1024
        return json.dumps({"versions": {version: {} for version in self.versions}}).encode("utf-8")


def registry(*versions: str) -> Registry:
    """Return a reader that serves one npm registry document listing ``versions``."""
    return Registry(*versions)


@pytest.fixture
def checkout(tmp_path: Path) -> Path:
    """Return a checkout holding one valid exception record for the Studio lock."""
    (tmp_path / EXCEPTION_FILE).parent.mkdir(parents=True)
    (tmp_path / EXCEPTION_FILE).write_bytes(table(record()))
    return tmp_path


def test_the_committed_record_holds_for_the_committed_studio_lock() -> None:
    """The maintained table loads, and its picomatch record matches the maintained lock exactly."""
    raw = (ROOT / EXCEPTION_FILE).read_bytes()
    # The day before the maintained record's review date, so the test follows each renewal of it.
    on_review = date.fromisoformat(
        tomllib.loads(raw.decode())["exception"][0]["review_by"]
    ) - timedelta(days=1)
    records = load_npm_exceptions(raw, on_review)
    assert [(row["lock"], row["package"], row["version"], row["advisory"]) for row in records] == [
        (LOCK, "picomatch", "4.0.7", ADVISORY)
    ]
    decision = npm_lock_decision(
        ROOT,
        LOCK,
        (ROOT / LOCK).read_bytes(),
        report(advisory(), total=0),
        read=registry("2.3.1", "4.0.6", "4.0.7"),
        today=on_review,
    )
    assert decision["passed"] is True
    installations = decision["accepted"][0]["evidence"]["installations"]
    assert installations and all(
        item == {"path": item["path"], "version": "4.0.7", "dev": True} for item in installations
    )


def test_an_empty_table_is_valid() -> None:
    """A table without records loads as no exception."""
    assert load_npm_exceptions(b"exception = []\n", TODAY) == ()


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        (b"other = []\n", "table is invalid"),
        (b"exception = {}\n", "table is invalid"),
        (
            table({key: value for key, value in record().items() if key != "rationale"}),
            "fields are invalid",
        ),
        (table(record(rationale=" ")), "fields are invalid"),
        (table(record(version="4.0.*")), "fields are invalid"),
        (table(record(lock="/abs/package-lock.json")), "relative package-lock.json"),
        (table(record(lock="studio/../package-lock.json")), "relative package-lock.json"),
        (table(record(lock="studio/frontend/package.json")), "relative package-lock.json"),
        (table(record(lock="studio//frontend/package-lock.json")), "relative package-lock.json"),
        (table(record(package="Pico Match")), "package or version is invalid"),
        (table(record(version="4.0.7-beta.1")), "package or version is invalid"),
        (table(record(reason="fixed_code_present")), "reason or advisory is unsupported"),
        (table(record(advisory="CVE-2026-112366")), "reason or advisory is unsupported"),
        (table(record(vulnerable_range="<4.0.7")), "outside its vulnerable range"),
        (table(record(review_by="2026-11-16")), "requires renewed review"),
        (table(record(review_by="2026-12-02")), "exceeds two weeks"),
        (table(record(), record()), "scope is repeated"),
    ],
)
def test_each_invalid_record_is_refused(raw: bytes, message: str) -> None:
    """A record that widens scope, lacks a field or outlives two weeks fails to load."""
    with pytest.raises(ValueError, match=message):
        load_npm_exceptions(raw, TODAY)


def test_a_record_that_is_not_a_table_is_refused() -> None:
    """An exception entry must be a table of text fields."""
    with pytest.raises(ValueError, match="fields are invalid"):
        load_npm_exceptions(b'exception = ["picomatch"]\n', TODAY)


def test_a_record_with_an_inexact_version_is_refused() -> None:
    """An installed version must be exact before it can be compared with the range."""
    with pytest.raises(ValueError, match="not exact"):
        load_npm_exceptions(table(record(version="v4.0.7")), TODAY)


@pytest.mark.parametrize(
    ("version", "vulnerable_range", "inside"),
    [
        ("4.0.7", "<=4.0.7", True),
        ("4.0.8", "<=4.0.7", False),
        ("4.0.7", "<4.0.7", False),
        ("3.9.99", "<4.0.0", True),
        ("4.0.7+build.5", "<=4.0.7", True),
    ],
)
def test_the_range_bounds_final_releases(version: str, vulnerable_range: str, inside: bool) -> None:
    """``<=`` includes its bound and ``<`` excludes it; build metadata does not order releases."""
    assert in_range(version, vulnerable_range) is inside


@pytest.mark.parametrize(
    ("version", "vulnerable_range", "message"),
    [
        ("4.0.7", ">=4.0.0", "range form is unsupported"),
        ("4.0.7", "<=4.0.7 || >=5.0.0", "range form is unsupported"),
        ("4.0.8-rc.1", "<=4.0.7", "needs final releases"),
        ("latest", "<=4.0.7", "not exact"),
    ],
)
def test_unsupported_ranges_and_versions_are_refused(
    version: str, vulnerable_range: str, message: str
) -> None:
    """Only ``<=X.Y.Z`` and ``<X.Y.Z`` against an exact final release are compared."""
    with pytest.raises(ValueError, match=message):
        in_range(version, vulnerable_range)


def test_released_fixes_are_the_final_releases_outside_the_range() -> None:
    """A newer final release is a fix; a prerelease is not, and the list is in release order."""
    document = json.dumps(
        {"versions": {"4.0.10": {}, "4.0.7": {}, "4.0.9": {}, "5.0.0-beta.1": {}}}
    ).encode()
    assert released_fixes(document, "<=4.0.7") == ["4.0.9", "4.0.10"]


@pytest.mark.parametrize("document", [b"[]", b'{"versions": {}}', b'{"versions": ["4.0.7"]}'])
def test_a_registry_document_without_versions_is_refused(document: bytes) -> None:
    """A registry answer that lists no versions cannot prove that no fix exists."""
    with pytest.raises(ValueError, match="no version list"):
        released_fixes(document, "<=4.0.7")


def test_a_named_dev_only_advisory_without_a_fix_is_accepted(checkout: Path) -> None:
    """The record holds: dev-only installation, matching advisory and range, no released fix."""
    read = registry("4.0.6", "4.0.7")
    decision = npm_lock_decision(checkout, LOCK, lock(), report(advisory()), read=read, today=TODAY)
    assert decision["passed"] is True and decision["blocked"] == [] and decision["unused"] == []
    assert decision["accepted"] == [
        {
            "advisory": ADVISORY,
            "package": "picomatch",
            "reason": "dev_only_fix_unreleased",
            "review_by": "2026-12-01",
            "evidence": {
                "installations": [
                    {"path": "node_modules/picomatch", "version": "4.0.7", "dev": True}
                ],
                "released_fixes": [],
            },
        }
    ]
    assert read.calls == [NPM_REGISTRY + "picomatch"]
    assert len(decision["exception_file_sha256"]) == 64


@pytest.mark.parametrize(
    ("installation", "fixes"),
    [
        ({"dev": False}, ()),
        ({"version": "4.0.6"}, ()),
        ({}, ("4.0.8",)),
    ],
    ids=["production-installation", "other-version", "fix-released"],
)
def test_the_record_does_not_hold_for_production_other_versions_or_released_fixes(
    checkout: Path, installation: dict[str, Any], fixes: tuple[str, ...]
) -> None:
    """A production installation, another installed version or a released fix blocks the advisory."""
    decision = npm_lock_decision(
        checkout,
        LOCK,
        lock(**installation),
        report(advisory()),
        read=registry("4.0.7", *fixes),
        today=TODAY,
    )
    assert decision["passed"] is False and decision["accepted"] == []
    assert decision["blocked"][0]["advisory"] == ADVISORY
    assert decision["blocked"][0]["evidence"]["released_fixes"] == list(fixes)
    assert decision["unused"] == [record()]


def test_a_lock_without_the_package_does_not_hold(checkout: Path) -> None:
    """An advisory on a package the lock does not install cannot be proven development-only."""
    raw = json.loads(lock())
    raw["packages"]["node_modules/picomatch-other"] = raw["packages"].pop("node_modules/picomatch")
    decision = npm_lock_decision(
        checkout,
        LOCK,
        json.dumps(raw).encode(),
        report(advisory()),
        read=registry("4.0.7"),
        today=TODAY,
    )
    assert decision["passed"] is False
    assert decision["blocked"][0]["evidence"]["installations"] == []


@pytest.mark.parametrize(
    "other",
    [
        advisory(url="https://github.com/advisories/GHSA-2222-3333-4444"),
        advisory(vulnerable_range="<4.0.7"),
        advisory(name="micromatch"),
        advisory(url=None),
    ],
    ids=["other-advisory", "other-range", "other-package", "no-url"],
)
def test_every_advisory_the_records_do_not_name_blocks_the_audit(
    checkout: Path, other: dict[str, Any]
) -> None:
    """An advisory that differs in identifier, range or package from every record stays blocking."""
    decision = npm_lock_decision(
        checkout, LOCK, lock(), report(advisory(), other), read=registry("4.0.7"), today=TODAY
    )
    assert decision["passed"] is False
    assert [item["advisory"] for item in decision["accepted"]] == [ADVISORY]
    assert len(decision["blocked"]) == 1


def test_a_record_without_a_finding_blocks_the_audit(checkout: Path) -> None:
    """A stale record is reported as unused and fails the decision until it is removed."""
    decision = npm_lock_decision(
        checkout, LOCK, lock(), report(advisory(name="react")), read=registry("4.0.7"), today=TODAY
    )
    assert decision["passed"] is False and decision["unused"] == [record()]


def test_records_of_other_locks_do_not_apply(tmp_path: Path) -> None:
    """A record names one lock; the same advisory in another lock stays blocking."""
    (tmp_path / EXCEPTION_FILE).parent.mkdir(parents=True)
    (tmp_path / EXCEPTION_FILE).write_bytes(table(record(lock="tools/site/package-lock.json")))
    decision = npm_lock_decision(
        tmp_path, LOCK, lock(), report(advisory()), read=registry("4.0.7"), today=TODAY
    )
    assert decision["passed"] is False and decision["unused"] == [] and decision["accepted"] == []


@pytest.mark.parametrize(
    "via",
    [[], "picomatch", ["unknown-package"], [7]],
    ids=["empty", "not-a-list", "unknown-cause", "not-text"],
)
def test_findings_with_unreadable_causes_are_refused(checkout: Path, via: Any) -> None:
    """Every finding must name advisory objects or other findings as its cause."""
    payload = report(advisory())
    payload["vulnerabilities"]["vite"]["via"] = via
    with pytest.raises(ValueError, match="cause"):
        npm_lock_decision(checkout, LOCK, lock(), payload, read=registry("4.0.7"), today=TODAY)


def test_a_provider_error_is_refused(checkout: Path) -> None:
    """A report carrying an npm error is never judged by the exception table."""
    payload = report(advisory())
    payload["error"] = {"code": "ENOAUDIT"}
    with pytest.raises(ValueError, match="provider error"):
        npm_lock_decision(checkout, LOCK, lock(), payload, read=registry("4.0.7"), today=TODAY)


def test_a_table_changed_during_verification_is_refused(checkout: Path) -> None:
    """The table read at the start must be the table in place when the decision is made."""

    def read(url: str, limit: int) -> bytes:
        (checkout / EXCEPTION_FILE).write_bytes(table(record(review_by="2026-11-30")))
        return json.dumps({"versions": {"4.0.7": {}}}).encode()

    with pytest.raises(ValueError, match="changed during its verification"):
        npm_lock_decision(checkout, LOCK, lock(), report(advisory()), read=read, today=TODAY)


def test_a_scoped_package_is_read_with_its_scope_encoded(checkout: Path) -> None:
    """A scoped package name reaches the registry as one path segment."""
    (checkout / EXCEPTION_FILE).write_bytes(table(record(package="@vite/glob")))
    raw = json.loads(lock())
    raw["packages"]["node_modules/@vite/glob"] = raw["packages"].pop("node_modules/picomatch")
    read = registry("4.0.7")
    decision = npm_lock_decision(
        checkout,
        LOCK,
        json.dumps(raw).encode(),
        report(advisory(name="@vite/glob")),
        read=read,
        today=TODAY,
    )
    assert decision["passed"] is True
    assert read.calls == [NPM_REGISTRY + "@vite%2Fglob"]


def test_a_clean_report_passes_only_with_a_clean_exit(checkout: Path) -> None:
    """With no findings the npm exit status decides; no exception decision is written."""
    clean = report()
    assert npm_audit_passed(checkout, LOCK, lock(), 0, clean, checkout) is True
    assert npm_audit_passed(checkout, LOCK, lock(), 1, clean, checkout) is False
    assert not (checkout / "npm-advisory-exceptions.json").exists()


def test_findings_pass_only_through_a_holding_decision_and_the_findings_exit(
    checkout: Path,
) -> None:
    """Accepted findings need npm's findings status 1, and the decision is kept in the packet."""
    read = registry("4.0.7")
    assert npm_audit_passed(
        checkout, LOCK, lock(), 1, report(advisory()), checkout, read=read, today=TODAY
    )
    kept = json.loads((checkout / "npm-advisory-exceptions.json").read_text(encoding="utf-8"))
    assert kept["passed"] is True and kept["accepted"][0]["advisory"] == ADVISORY
    assert not npm_audit_passed(
        checkout, LOCK, lock(), 0, report(advisory()), checkout, read=read, today=TODAY
    )
    assert not npm_audit_passed(
        checkout, LOCK, lock(dev=False), 1, report(advisory()), checkout, read=read, today=TODAY
    )


def test_an_incomplete_report_is_refused_before_any_exception(checkout: Path) -> None:
    """The complete-report checks of npm_report_passed still come first."""
    with pytest.raises(ValueError, match="complete lock package inventory"):
        npm_audit_passed(
            checkout,
            LOCK,
            lock(),
            1,
            report(advisory(), total=2),
            checkout,
            read=registry("4.0.7"),
            today=TODAY,
        )


def test_counters_without_findings_are_refused(checkout: Path) -> None:
    """A failing report whose findings map is empty cannot pass through the exception table."""
    payload = report()
    payload["metadata"]["vulnerabilities"].update(high=1, total=1)
    with pytest.raises(ValueError, match="no finding to decide"):
        npm_lock_decision(checkout, LOCK, lock(), payload, read=registry("4.0.7"), today=TODAY)


def test_findings_that_name_only_each_other_are_refused(checkout: Path) -> None:
    """A cycle of findings that never reaches an advisory object is not an exception's to judge."""
    payload = report(advisory())
    payload["vulnerabilities"]["a"] = {"name": "a", "via": ["b"]}
    payload["vulnerabilities"]["b"] = {"name": "b", "via": ["a"]}
    payload["metadata"]["vulnerabilities"].update(high=4, total=4)
    with pytest.raises(ValueError, match="reaches no advisory"):
        npm_lock_decision(checkout, LOCK, lock(), payload, read=registry("4.0.7"), today=TODAY)


@pytest.mark.parametrize("total", [1, 3], ids=["fewer", "more"])
def test_counters_that_disagree_with_the_findings_are_refused(checkout: Path, total: int) -> None:
    """The total counter must equal the number of findings the decision reads."""
    payload = report(advisory())
    payload["metadata"]["vulnerabilities"]["total"] = total
    with pytest.raises(ValueError, match="counters disagree"):
        npm_lock_decision(checkout, LOCK, lock(), payload, read=registry("4.0.7"), today=TODAY)


def test_a_report_without_metadata_counters_is_refused(checkout: Path) -> None:
    """A decision called on its own still needs the report's counters."""
    payload = report(advisory())
    del payload["metadata"]
    with pytest.raises(ValueError, match="counters disagree"):
        npm_lock_decision(checkout, LOCK, lock(), payload, read=registry("4.0.7"), today=TODAY)


def osv_affected(events: Any, *, package: str = "picomatch", kind: str = "SEMVER") -> list[Any]:
    """Return an OSV ``affected`` list with one range of ``events`` for an npm package."""
    return [
        {
            "package": {"ecosystem": "npm", "name": package},
            "ranges": [{"type": kind, "events": events}],
        }
    ]


LAST_AFFECTED = [{"introduced": "0"}, {"last_affected": "4.0.7"}]


def osv_report(
    checkout: Path,
    *,
    identifier: str = ADVISORY,
    package: dict[str, str] | None = None,
    affected: Any = None,
    path: str | None = None,
) -> dict[str, Any]:
    """Return an OSV-Scanner report with one finding on the Studio lock of ``checkout``."""
    vulnerability = {
        "id": identifier,
        "aliases": ["CVE-2026-112366"],
        "affected": osv_affected(LAST_AFFECTED) if affected is None else affected,
    }
    return {
        "results": [
            {
                "source": {"path": path or str(checkout / LOCK), "type": "lockfile"},
                "packages": [
                    {
                        "package": package
                        or {"name": "picomatch", "version": "4.0.7", "ecosystem": "npm"},
                        "vulnerabilities": [vulnerability],
                    }
                ],
            }
        ]
    }


@pytest.fixture
def osv_checkout(checkout: Path) -> Path:
    """Return the checkout with the synthetic Studio lock written at its tracked path."""
    (checkout / LOCK).parent.mkdir(parents=True)
    (checkout / LOCK).write_bytes(lock())
    return checkout


def test_osv_accepts_a_named_and_proven_finding(osv_checkout: Path) -> None:
    """OSV's finding passes with the same record, installation proof and registry answer."""
    read = registry("4.0.6", "4.0.7")
    decision = osv_npm_decision(osv_checkout, osv_report(osv_checkout), read=read, today=TODAY)
    assert decision["passed"] is True and decision["blocked"] == []
    assert decision["accepted"] == [
        {
            "id": ADVISORY,
            "package": "picomatch",
            "lock": LOCK,
            "evidence": {
                "installations": [
                    {"path": "node_modules/picomatch", "version": "4.0.7", "dev": True}
                ],
                "released_fixes": [],
            },
        }
    ]
    assert read.calls == [NPM_REGISTRY + "picomatch"]


def test_osv_accepts_a_fixed_bound_for_a_strict_range(osv_checkout: Path) -> None:
    """A record of ``<X.Y.Z`` matches OSV's range to a fixed release X.Y.Z that is not yet published."""
    (osv_checkout / EXCEPTION_FILE).write_bytes(table(record(vulnerable_range="<4.0.8")))
    report_ = osv_report(
        osv_checkout, affected=osv_affected([{"introduced": "0"}, {"fixed": "4.0.8"}])
    )
    decision = osv_npm_decision(osv_checkout, report_, read=registry("4.0.7"), today=TODAY)
    assert decision["passed"] is True


@pytest.mark.parametrize(
    "change",
    [
        {"identifier": "GHSA-2222-3333-4444"},
        {"package": {"name": "picomatch", "version": "4.0.6", "ecosystem": "npm"}},
        {"package": {"name": "picomatch", "version": "4.0.7", "ecosystem": "PyPI"}},
        {"path": "/elsewhere/studio/frontend/package-lock.json"},
        {"affected": osv_affected([{"introduced": "0"}, {"fixed": "4.0.8"}])},
        {"affected": osv_affected(LAST_AFFECTED, package="micromatch")},
        {"affected": osv_affected(LAST_AFFECTED, kind="ECOSYSTEM")},
        {"affected": osv_affected([{"introduced": "1.0.0"}, {"last_affected": "4.0.7"}])},
        {
            "affected": osv_affected(
                [{"introduced": "0"}, {"last_affected": "4.0.7", "fixed": "4.0.8"}]
            )
        },
        {"affected": osv_affected([{"introduced": "0"}, {"limit": "4.0.8"}])},
        {"affected": osv_affected([{"introduced": "0"}])},
        {"affected": [{"package": {"ecosystem": "npm", "name": "picomatch"}, "ranges": []}]},
        {
            "affected": [
                {"package": {"ecosystem": "npm", "name": "picomatch"}, "ranges": ["SEMVER"]}
            ]
        },
        {"affected": [{"package": "picomatch"}, "picomatch"]},
        {"affected": {"package": {"ecosystem": "npm", "name": "picomatch"}}},
    ],
    ids=[
        "other-advisory",
        "other-version",
        "other-ecosystem",
        "outside-checkout",
        "other-range",
        "other-package",
        "not-semver",
        "not-from-zero",
        "two-bounds",
        "limit-only",
        "no-bound",
        "no-range",
        "range-not-object",
        "package-not-object",
        "affected-not-list",
    ],
)
def test_osv_blocks_every_finding_a_record_does_not_name_exactly(
    osv_checkout: Path, change: dict[str, Any]
) -> None:
    """A difference in identifier, package, version, ecosystem, lock or affected range blocks the finding."""
    decision = osv_npm_decision(
        osv_checkout, osv_report(osv_checkout, **change), read=registry("4.0.7"), today=TODAY
    )
    assert (
        decision["passed"] is False and decision["accepted"] == [] and len(decision["blocked"]) == 1
    )


def test_osv_blocks_a_named_finding_whose_record_does_not_hold(osv_checkout: Path) -> None:
    """A production installation found by the same proof blocks the OSV finding too."""
    (osv_checkout / LOCK).write_bytes(lock(dev=False))
    decision = osv_npm_decision(
        osv_checkout, osv_report(osv_checkout), read=registry("4.0.7"), today=TODAY
    )
    assert decision["passed"] is False
    assert decision["blocked"][0]["evidence"]["installations"][0]["dev"] is False


def test_osv_reads_each_lock_once_and_skips_entries_that_are_not_objects(
    osv_checkout: Path,
) -> None:
    """Two findings on one lock share one lock read; non-object entries carry no finding."""
    report_ = osv_report(osv_checkout)
    package = report_["results"][0]["packages"][0]
    package["vulnerabilities"].append(package["vulnerabilities"][0])
    package["vulnerabilities"].append("GHSA-not-an-object")
    report_["results"][0]["packages"].append("not-a-package")
    report_["results"].append("not-a-result")
    decision = osv_npm_decision(osv_checkout, report_, read=registry("4.0.7"), today=TODAY)
    assert decision["passed"] is True and len(decision["accepted"]) == 2


def test_osv_with_nothing_to_accept_does_not_pass(osv_checkout: Path) -> None:
    """A report with no finding gives the OSV exception nothing to pass on."""
    decision = osv_npm_decision(osv_checkout, {"results": []}, read=registry("4.0.7"), today=TODAY)
    assert decision == {
        "passed": False,
        "accepted": [],
        "blocked": [],
        "exception_file_sha256": decision["exception_file_sha256"],
    }


def test_osv_refuses_a_table_changed_during_verification(osv_checkout: Path) -> None:
    """The OSV decision, like the npm one, needs the same table at its start and its end."""

    def read(url: str, limit: int) -> bytes:
        (osv_checkout / EXCEPTION_FILE).write_bytes(table(record(review_by="2026-11-30")))
        return json.dumps({"versions": {"4.0.7": {}}}).encode()

    with pytest.raises(ValueError, match="changed during its verification"):
        osv_npm_decision(osv_checkout, osv_report(osv_checkout), read=read, today=TODAY)
