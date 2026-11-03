# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — AXI configuration and complete dense-layer measurement lifecycle

"""Exercise the actual top-level AXI bus, neuron datapath and firing-rate bank."""

import shutil
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SOURCES = (
    "sc_bitstream_encoder.v",
    "sc_bitstream_synapse.v",
    "sc_dotproduct_to_current.v",
    "sc_lif_neuron.v",
    "sc_axil_cfg.v",
    "sc_dense_layer_core.v",
    "sc_firing_rate_bank.v",
    "sc_neurocore_top.v",
)
# Per-command test governance allowance; execution resources belong to the runner.
COMMAND_TIMEOUT_SECONDS = 120


@pytest.mark.parametrize(
    ("bench", "summary"),
    [
        ("tb_sc_neurocore_top", "tb_sc_neurocore_top: 6 passed, 0 failed"),
        ("tb_sc_neurocore_top_lifecycle", "AXI LIFECYCLE PASS"),
    ],
)
def test_top_level_axi_measurement(tmp_path: Path, bench: str, summary: str) -> None:
    """Complete configuration and repeated measurements through native RTL ports.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Test-owned directory for the compiled circuit and native logs.
    bench : str
        Maintained full-system or lifecycle testbench module.
    summary : str
        Successful terminal marker emitted after every bench assertion.
    """
    compiler = shutil.which("iverilog")
    runtime = shutil.which("vvp")
    assert compiler is not None, "Top-level RTL acceptance requires Icarus Verilog."
    assert runtime is not None, "Top-level RTL acceptance requires vvp."
    executable = tmp_path / (bench + ".vvp")
    compilation = subprocess.run(
        [
            compiler,
            "-g2012",
            "-Wall",
            "-s",
            bench,
            "-o",
            str(executable),
            str(ROOT / "hdl" / (bench + ".v")),
            *[str(ROOT / "hdl" / name) for name in SOURCES],
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=COMMAND_TIMEOUT_SECONDS,
        check=False,
    )
    (tmp_path / "compilation.log").write_text(compilation.stdout + compilation.stderr)
    assert compilation.returncode == 0, compilation.stderr
    assert compilation.stderr == "", compilation.stderr
    execution = subprocess.run(
        [runtime, str(executable)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=COMMAND_TIMEOUT_SECONDS,
        check=False,
    )
    (tmp_path / "execution.log").write_text(execution.stdout + execution.stderr)
    assert execution.returncode == 0, execution.stdout + execution.stderr
    assert execution.stderr == "", execution.stderr
    assert "[FAIL]" not in execution.stdout, execution.stdout
    assert summary in execution.stdout, execution.stdout
