# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — fixed-point LIF public co-simulation CLI tests

"""Exercise the advertised LIF CLI against real handwritten RTL observations."""

from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from scripts.cosim_gen_and_check import check_results, generate_stimuli

REPO_ROOT = Path(__file__).resolve().parents[1]
CLI = REPO_ROOT / "scripts/cosim_gen_and_check.py"


def _run_cli(directory: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    """Run the installed Python interpreter against the actual public script."""
    return subprocess.run(
        [sys.executable, str(CLI), *arguments],
        cwd=directory,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )


def test_cosim_cli_real_rtl_and_invalid_traces(tmp_path: Path) -> None:
    """Accept actual RTL matches and refuse damaged observations at the CLI.

    Parameters
    ----------
    tmp_path : Path
        Isolated directory for stimuli, compiled RTL and result observations.
    """
    compiler = shutil.which("iverilog")
    simulator = shutil.which("vvp")
    assert compiler is not None, "The LIF CLI regression requires Icarus Verilog."
    assert simulator is not None, "The LIF CLI regression requires vvp."

    generated = _run_cli(tmp_path, "--generate")
    assert generated.returncode == 0, generated.stdout + generated.stderr
    stimuli = tmp_path / "stimuli.txt"
    expected = tmp_path / "results_expected.txt"
    original_stimuli = stimuli.read_bytes()
    original_expected = expected.read_bytes()
    assert len(original_stimuli.splitlines()) == 1000
    assert len(original_expected.splitlines()) == 1000
    executable = tmp_path / "tb_lif"
    compiled = subprocess.run(
        [
            compiler,
            "-s",
            "tb_sc_lif_neuron",
            "-o",
            str(executable),
            str(REPO_ROOT / "hdl/sc_lif_neuron.v"),
            str(REPO_ROOT / "hdl/tb_sc_lif_neuron.v"),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert compiled.returncode == 0, compiled.stdout + compiled.stderr
    simulated = subprocess.run(
        [simulator, str(executable)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert simulated.returncode == 0, simulated.stdout + simulated.stderr
    result_path = tmp_path / "results_verilog.txt"
    actual = result_path.read_bytes()
    rows = actual.decode("ascii").splitlines(keepends=True)
    assert len(rows) == 1000
    assert any(row.split()[0] == "1" for row in rows)
    matched = _run_cli(tmp_path, "--generate", "--check", "--steps", "1000", "--seed", "42")
    assert matched.returncode == 0, matched.stdout + matched.stderr
    assert "SUCCESS: All 1000 steps match bit-exactly!" in matched.stdout
    assert stimuli.read_bytes() == original_stimuli
    assert expected.read_bytes() == original_expected

    damaged: list[tuple[str, bytes]] = [
        ("empty", b""),
        ("one correct row", rows[0].encode("ascii")),
        ("short trace", "".join(rows[:-1]).encode("ascii")),
        ("extra row", actual + rows[-1].encode("ascii")),
    ]
    for bad_row in (
        "\n",
        "0\n",
        "0 0 extra\n",
        "x 0\n",
        "0 z\n",
        "2 0\n",
        "0 1_0\n",
        "0 1.0\n",
        "0 0x10\n",
        "0 1e2\n",
        "0 +\n",
        "0 １２\n",
        "0 " + "9" * 10000 + "\n",
    ):
        corrupted = rows.copy()
        corrupted[17] = bad_row
        damaged.append((repr(bad_row), "".join(corrupted).encode("utf-8")))
    for name, content in damaged:
        result_path.write_bytes(content)
        refused = _run_cli(tmp_path, "--check")
        assert refused.returncode == 1, (name, refused.stdout, refused.stderr)
        assert "SUCCESS" not in refused.stdout, name
        assert "Traceback" not in refused.stderr, name

    mismatched = rows.copy()
    for index in range(12):
        spike, potential = rows[index].split()
        mismatched[index] = f"{1 - int(spike)} {potential}\n"
    result_path.write_text("".join(mismatched), encoding="ascii")
    refused = _run_cli(tmp_path, "--check")
    assert refused.returncode == 1
    assert "FAILURE: 12/1000 mismatches found." in refused.stdout
    assert refused.stdout.count("Mismatch at step") == 10

    result_path.unlink()
    missing = _run_cli(tmp_path, "--check")
    assert missing.returncode == 1
    assert "Cannot read the ASCII Verilog trace" in missing.stdout
    assert "Traceback" not in missing.stderr
    result_path.mkdir()
    unreadable = _run_cli(tmp_path, "--check")
    assert unreadable.returncode == 1
    assert "Traceback" not in unreadable.stderr
    result_path.rmdir()
    result_path.write_bytes(actual)
    for arguments in (("--steps", "0"), ("--steps", "-1"), ("--seed", "-1")):
        invalid = _run_cli(tmp_path, "--check", *arguments)
        assert invalid.returncode == 2
        assert "SUCCESS" not in invalid.stdout
        assert "Traceback" not in invalid.stderr
    help_result = _run_cli(tmp_path)
    assert help_result.returncode == 0
    assert "Quick start:" in help_result.stdout
    assert not check_results([], str(result_path))
    for steps, seed in ((0, 42), (-1, 42), (1, -1)):
        with pytest.raises(ValueError, match="steps must be positive and seed must be nonnegative"):
            generate_stimuli(steps, seed)
