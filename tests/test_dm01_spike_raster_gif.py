# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "examples/dm01_spike_raster_gif.py"


def test_dm01_uses_explicit_writer_capability_not_broad_fallback() -> None:
    source = SCRIPT.read_text(encoding="utf-8")

    assert 'animation.writers.is_available("pillow")' in source
    assert "except Exception" not in source
    assert "HodgkinHuxleyNeuron" in source


def test_dm01_generates_real_animation(tmp_path: Path) -> None:
    output = tmp_path / "hh.gif"
    env = os.environ | {"PYTHONPATH": str(REPO / "src"), "MPLBACKEND": "Agg"}
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--n-steps",
            "400",
            "--frames",
            "3",
            "--out",
            str(output),
        ],
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert output.is_file()
    assert output.stat().st_size > 1000
