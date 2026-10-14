# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Native locked Pixi environment audit contracts

"""Exercise the complete installed Conda inventory and native Syft/Grype gate."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
from typing import Any

import pytest

from tools.security_scan.locked_dependency_audit import run_locked_audit
from tools.security_scan.locked_dependency_inventory import pixi_package_count

ROOT = Path(__file__).resolve().parents[2]
LOCK = "src/sc_neurocore/accel/mojo/pixi.lock"
NativeCondaPacket = tuple[dict[str, Any], dict[str, Any], dict[str, Any], Path, Path]


@pytest.fixture(scope="session")
def native_pixi_audit(tmp_path_factory: pytest.TempPathFactory) -> NativeCondaPacket:
    """Run the maintained public audit with real tools and retain its evidence.

    Parameters
    ----------
    tmp_path_factory : pytest.TempPathFactory
        Session-owned directory for actual native process packets.

    Returns
    -------
    tuple
        Syft report, Grype report, decision, installed prefix and packet directory.
    """
    for executable in ("pixi", "syft", "grype"):
        assert shutil.which(executable), f"Native Conda qualification requires {executable}."
    target = tmp_path_factory.mktemp("native-conda") / "packet"
    report = run_locked_audit(ROOT, target, "pixi")
    if report["passed"] is not True:
        diagnostics = "\n".join(
            f"{path.parent.name}: {path.read_text()[-4000:]}" for path in target.rglob("stderr.log")
        )
        pytest.fail(f"Native Conda audit failed: {report}; stderr={diagnostics}")
    packet = target / "lock-1"
    syft = json.loads((packet / "syft/stdout.json").read_text())
    grype = json.loads((packet / "grype/stdout.json").read_text())
    decision = json.loads((packet / "summary.json").read_text())
    prefix = Path(syft["source"]["metadata"]["path"])
    return syft, grype, decision, prefix, packet


def test_pixi_native_cli_and_complete_lock_gate(native_pixi_audit: NativeCondaPacket) -> None:
    """Bind every installed package and finding to native process evidence.

    Parameters
    ----------
    native_pixi_audit : tuple
        Actual complete-lock audit reports and installed environment.
    """
    syft, grype, decision, _, packet = native_pixi_audit
    count = pixi_package_count((ROOT / LOCK).read_bytes())
    assert decision["passed"] is True and decision["package_inventory_complete"] is True
    assert decision["package_count"] == len(syft["artifacts"]) == count > 0
    assert decision["advisory_matching"] == "NVD/CPE" and decision["limitations"]
    assert len(decision["accepted"]) + len(decision["blocked"]) == len(grype["matches"])
    assert decision["blocked"] == decision["unused"] == []
    assert all(row["evidence"]["holds"] is True for row in decision["accepted"])
    for name in ("install", "environment", "syft", "grype"):
        receipt = json.loads((packet / name / "receipt.json").read_text())
        assert receipt == {"returncode": 0, "timed_out": False, "process_reaped": True}
        assert json.loads((packet / name / "command.json").read_text())
