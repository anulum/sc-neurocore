# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li

from __future__ import annotations

from pathlib import Path
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "examples/dm04_synthesis_report_reader.py"


def test_dm04_reads_committed_reports() -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--dir", str(REPO / "hdl/reports")],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "vivado_util_xc7z020_100mhz.rpt" in result.stdout
    assert "slice_luts: 990" in result.stdout
    assert "not a live synthesis run" in result.stdout


def test_dm04_fails_closed_on_malformed_json(tmp_path: Path) -> None:
    (tmp_path / "yosys_bad.json").write_text("{not-json", encoding="utf-8")
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--dir", str(tmp_path)],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 1
    assert "unreadable: JSONDecodeError" in result.stdout
