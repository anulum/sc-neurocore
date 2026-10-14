# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — native Pixi advisory report and full-lock gate regression

"""Exercise native audit responses through the maintained complete-lock gate."""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
from typing import cast

import pytest

from tools.security_scan.locked_dependency_audit import run_locked_audit
from tools.security_scan.locked_dependency_inventory import pixi_package_count
from tools.security_scan.locked_dependency_reports import pixi_report_passed

ROOT = Path(__file__).resolve().parents[2]


def test_pixi_native_cli_and_complete_lock_gate(tmp_path: Path) -> None:
    """Bind actual scanner responses and refuse corrupted coverage or omissions.

    Parameters
    ----------
    tmp_path : Path
        Owned directory for native audit evidence and the clean certificate lock.
    """
    pixi = shutil.which("pixi")
    assert pixi is not None, "Native advisory qualification requires Pixi."
    full_directory = tmp_path / "committed-lock-audit"
    report = run_locked_audit(ROOT, full_directory, "pixi")
    assert report["errors"] == [], report
    assert report["audits"], report
    for number, row in enumerate(report["audits"], 1):
        directory = full_directory / f"lock-{number}"
        if "error_type" in row:
            stderr_path = directory / "stderr.log"
            stderr = stderr_path.read_text() if stderr_path.exists() else "unavailable"
            token = os.environ.get("PIXI_AUDIT_TOKEN")
            if token:
                stderr = stderr.replace(token, "[REDACTED_SECRET]")
            receipt_path = directory / "receipt.json"
            receipt = receipt_path.read_text() if receipt_path.exists() else "unavailable"
            pytest.fail(f"Native Pixi audit failed: {row}; receipt={receipt}; stderr={stderr}")
        payload = json.loads((directory / "stdout.json").read_text())
        receipt = json.loads((directory / "receipt.json").read_text())
        assert receipt["process_reaped"] is True and receipt["timed_out"] is False
        count = pixi_package_count((ROOT / row["path"]).read_bytes())
        decision = json.loads((directory / "exceptions.json").read_text())
        assert row["passed"] is decision["passed"]
        assert len(decision["accepted"]) + len(decision["blocked"]) == len(
            payload["matches"]
        ) + len(payload["not_audited"])
        if pixi_report_passed(payload, count) and receipt["returncode"] == 0:
            assert row["passed"] is (decision["unused"] == [])
        for accepted in decision["accepted"]:
            assert accepted["evidence"]["holds"] is True
        assert len(payload["not_audited"]) + payload["summary"]["audited"] == count

    certificate_directory = tmp_path / "certificate-lock"
    certificate_directory.mkdir()
    manifest = certificate_directory / "pixi.toml"
    manifest.write_text(
        '[workspace]\nname = "sc-neurocore-audit-certificate"\n'
        'channels = ["conda-forge"]\nplatforms = ["linux-64"]\n'
        '[dependencies]\nca-certificates = "==2026.7.22"\n',
        encoding="utf-8",
    )
    environment = os.environ.copy()
    environment["TMPDIR"] = str(tmp_path)
    environment.setdefault("PIXI_CACHE_DIR", str(tmp_path / "pixi-cache"))
    generated = subprocess.run(
        [pixi, "lock", "--manifest-path", str(manifest), "--no-install"],
        cwd=certificate_directory,
        env=environment,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert generated.returncode == 0, generated.stdout + generated.stderr
    certificate_lock = certificate_directory / "pixi.lock"
    count = pixi_package_count(certificate_lock.read_bytes())
    assert count == 1
    command: list[str] = json.loads((full_directory / "lock-1/command.json").read_text())
    command[-1] = str(certificate_lock)
    scanned = subprocess.run(
        command,
        cwd=certificate_directory,
        env=environment,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert scanned.returncode == 0, scanned.stdout + scanned.stderr
    (tmp_path / "certificate-scanner.stdout.json").write_text(scanned.stdout, encoding="utf-8")
    clean: object = json.loads(scanned.stdout)
    assert isinstance(clean, dict)
    assert pixi_report_passed(clean, count) is True

    with pytest.raises(ValueError, match="invalid"):
        pixi_report_passed([], count)
    for invalid_count in (True, 0, -1, 1.0, "1", None):
        with pytest.raises(ValueError, match="invalid"):
            pixi_report_passed(clean, cast(int, invalid_count))
    for section in ("matches", "ignored", "not_audited", "unused_ignores"):
        missing = copy.deepcopy(clean)
        del missing[section]
        with pytest.raises(ValueError, match="lists are unavailable"):
            pixi_report_passed(missing, count)
        present = copy.deepcopy(clean)
        present[section] = [{"package": "ca-certificates", "version": "2026.7.22"}]
        counter_names = {
            "matches": "vulnerable",
            "ignored": "ignored",
            "not_audited": "not_audited",
        }
        if section in counter_names:
            present["summary"][counter_names[section]] = 1
        if section == "not_audited":
            present["summary"]["audited"] = 0
        assert pixi_report_passed(present, count) is False
    missing_summary = copy.deepcopy(clean)
    del missing_summary["summary"]
    with pytest.raises(ValueError, match="counters are unavailable"):
        pixi_report_passed(missing_summary, count)
    for counter in ("audited", "vulnerable", "ignored", "not_audited"):
        for value in (True, -1, 1.0, "0", None):
            invalid = copy.deepcopy(clean)
            invalid["summary"][counter] = value
            with pytest.raises(ValueError, match="counter is invalid"):
                pixi_report_passed(invalid, count)
    with pytest.raises(ValueError, match="complete lock"):
        pixi_report_passed(clean, count + 1)
    for counter in ("vulnerable", "ignored", "not_audited"):
        inconsistent = copy.deepcopy(clean)
        inconsistent["summary"][counter] = 1
        if counter == "not_audited":
            inconsistent["summary"]["audited"] = 0
        with pytest.raises(ValueError, match="disagree"):
            pixi_report_passed(inconsistent, count)
