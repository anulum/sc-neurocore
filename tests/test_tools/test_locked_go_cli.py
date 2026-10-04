# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Real Go identity refusal through the audit CLI

"""Exercise actual Go module parsing and refuse incomplete advisory identities."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("condition", ["replacement", "missing-go", "invalid-checksum"])
def test_public_cli_refuses_incomplete_go_identities(tmp_path: Path, condition: str) -> None:
    """Refuse actual Go identities whose complete advisory coverage is unavailable.

    Parameters
    ----------
    tmp_path : Path
        Fresh real checkout and retained public CLI audit packet.
    condition : str
        Module replacement, missing standard-library version or invalid checksum.

    Notes
    -----
    The real Go parser consumes a copy of the maintained module manifest.
    No subprocess response or advisory provider is substituted. All cases
    refuse before submitting an OSV query and leave source inputs intact.
    """
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    shutil.copytree(ROOT / "requirements", checkout / "requirements")
    source = (ROOT / "src/sc_neurocore/accel/go/go.mod").read_text()
    if condition == "replacement":
        source += "\nreplace example.invalid/locked => ./unavailable\n"
    elif condition == "missing-go":
        source = "\n".join(line for line in source.splitlines() if not line.startswith("go "))
    else:
        (checkout / "go.sum").write_text("example.invalid/locked v1.0.0 invalid-checksum\n")
    (checkout / "go.mod").write_text(source)
    subprocess.run(
        ["git", "init", "--initial-branch=main", str(checkout)],
        check=True,
        capture_output=True,
        timeout=30,
    )
    subprocess.run(["git", "add", "."], cwd=checkout, check=True, capture_output=True, timeout=30)
    target = tmp_path / "packet"
    process = subprocess.run(
        [
            sys.executable,
            "-B",
            "-m",
            "tools.security_scan.locked_dependency_audit",
            "--repo-root",
            str(checkout),
            "--ecosystem",
            "go",
            "--output-dir",
            str(target),
        ],
        cwd=ROOT,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "GOTOOLCHAIN": "local"},
        capture_output=True,
        text=True,
        timeout=45,
        check=False,
    )
    assert process.returncode == 1
    report = json.loads((target / "summary.json").read_text())
    assert report["passed"] is False and report["errors"] == []
    assert report["audits"] == [{"path": "go.mod", "passed": False, "error_type": "ValueError"}]
    assert not (target / "lock-1").exists()
    parser = json.loads((target / "lock-1-identities/receipt.json").read_text())
    assert parser == {"returncode": 0, "timed_out": False, "process_reaped": True}
    assert (checkout / "go.mod").read_text() == source
