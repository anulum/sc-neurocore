# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li

from __future__ import annotations

import csv
import os
from pathlib import Path
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "examples/dm02_sc_error_sweep.py"


def test_dm02_runs_hh_derived_sc_sweep(tmp_path: Path) -> None:
    csv_path = tmp_path / "sweep.csv"
    png_path = tmp_path / "sweep.png"
    env = os.environ | {"PYTHONPATH": str(REPO / "src"), "MPLBACKEND": "Agg"}
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--csv", str(csv_path), "--png", str(png_path)],
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    with csv_path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert [int(row["bitstream_length"]) for row in rows] == [
        64,
        128,
        256,
        512,
        1024,
        2048,
        4096,
        8192,
    ]
    assert all(float(row["abs_error"]) >= 0.0 for row in rows)
    assert len({row["proxy_value"] for row in rows}) == 1
    assert png_path.stat().st_size > 1000
