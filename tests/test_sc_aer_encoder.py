# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Actual RTL AER encoder priority and pending-bit contracts

"""Compile canonical AER RTL and exercise complete masks, addresses and reset timing."""

from __future__ import annotations

from pathlib import Path
import subprocess

import pytest


@pytest.mark.parametrize("neurons", [4, 128])
@pytest.mark.parametrize("timestamp_width", [8, 16])
def test_encoder_emits_each_pending_address_once_with_clocked_priority(
    tmp_path: Path, neurons: int, timestamp_width: int
) -> None:
    """Real RTL drains each bit once, preserves merge/coalescing policy and resets queued state."""
    root = Path(__file__).resolve().parents[1]
    image = tmp_path / "encoder.vvp"
    compiled = subprocess.run(
        [
            "iverilog",
            "-g2012",
            "-Wall",
            "-s",
            "tb_sc_aer_encoder",
            f"-Ptb_sc_aer_encoder.N_NEURONS={neurons}",
            f"-Ptb_sc_aer_encoder.TIMESTAMP_WIDTH={timestamp_width}",
            "-o",
            str(image),
            str(root / "hdl/tb_sc_aer_encoder.v"),
            str(root / "hdl/sc_aer_encoder.v"),
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert compiled.returncode == 0, compiled.stdout + compiled.stderr
    simulated = subprocess.run(
        ["vvp", str(image)], capture_output=True, text=True, timeout=30, check=False
    )
    assert simulated.returncode == 0, simulated.stdout + simulated.stderr
    assert f"PASS N={neurons} cases={neurons + 20}" in simulated.stdout
    assert (
        f"MEASUREMENT N={neurons} full_burst_events={neurons} "
        f"service_cycles={neurons} capture_cycles=1" in simulated.stdout
    )
