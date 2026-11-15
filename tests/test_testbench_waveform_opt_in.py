# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Generated testbench waveform opt-in tests

"""A generated testbench writes its VCD waveform only when the run asks for it with ``+vcd``.

The waveform goes to the simulator's working directory. Written by default, every co-simulation
that a test or a tool runs from a checkout left a ``tb_<module>.vcd`` there. These tests compile
a generated testbench with Icarus Verilog and run it twice in a private directory, and read what
``compile --testbench`` tells its user.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from sc_neurocore.cli import main
from sc_neurocore.compiler.equation_compiler import generate_testbench
from sc_neurocore.neurons.universal_dsl import UniversalNeuron

# The shared ``tests.cosim_runtime.HAS_IVERILOG`` checks ``iverilog`` only; these tests also run
# the compiled image with ``vvp``.
HAS_ICARUS = shutil.which("iverilog") is not None and shutil.which("vvp") is not None
MODULE = "sc_lif_waveform_probe"


def _testbench_and_rtl() -> tuple[str, str]:
    """Return a generated testbench and the RTL it drives, for the LIF schema.

    Returns
    -------
    tuple of str
        The testbench source and the Verilog module source.
    """
    neuron = UniversalNeuron.from_schema("lif")
    bench = generate_testbench(neuron.to_equation_neuron(), module_name=MODULE, n_steps=20)
    return bench, neuron.to_verilog(module_name=MODULE)


def _run(work: Path, *plusargs: str) -> subprocess.CompletedProcess[str]:
    """Compile the testbench in ``work`` and run it there with the given plusargs.

    Parameters
    ----------
    work : Path
        A private directory, also the simulator's working directory.
    *plusargs : str
        Plusargs passed to ``vvp`` after the compiled image.

    Returns
    -------
    subprocess.CompletedProcess
        The simulation's exit status and output.
    """
    bench, rtl = _testbench_and_rtl()
    (work / f"{MODULE}.v").write_text(rtl, encoding="utf-8")
    (work / f"tb_{MODULE}.v").write_text(bench, encoding="utf-8")
    subprocess.run(
        ["iverilog", "-g2012", "-o", "sim", f"{MODULE}.v", f"tb_{MODULE}.v"],
        cwd=work,
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    return subprocess.run(
        ["vvp", "sim", *plusargs], cwd=work, check=True, capture_output=True, text=True, timeout=120
    )


def test_the_dump_is_guarded_by_the_vcd_plusarg() -> None:
    """``$dumpfile`` and ``$dumpvars`` sit inside a ``$test$plusargs("vcd")`` block."""
    lines = [line.strip() for line in _testbench_and_rtl()[0].splitlines()]
    guard = lines.index('if ($test$plusargs("vcd")) begin')
    assert lines[guard + 1] == f'$dumpfile("tb_{MODULE}.vcd");'
    assert lines[guard + 2] == f"$dumpvars(0, tb_{MODULE});"
    assert lines[guard + 3] == "end"
    assert sum(line.startswith("$dump") for line in lines) == 2


@pytest.mark.skipif(not HAS_ICARUS, reason="Icarus Verilog (iverilog, vvp) is not installed")
def test_a_run_without_the_plusarg_writes_no_waveform(tmp_path: Path) -> None:
    """The simulation completes and leaves no ``.vcd`` file in its working directory."""
    completed = _run(tmp_path)
    assert "Simulation complete" in completed.stdout
    assert sorted(path.name for path in tmp_path.glob("*.vcd")) == []


@pytest.mark.skipif(not HAS_ICARUS, reason="Icarus Verilog (iverilog, vvp) is not installed")
def test_the_plusarg_writes_the_waveform(tmp_path: Path) -> None:
    """``vvp sim +vcd`` writes ``tb_<module>.vcd`` with the testbench's signals."""
    completed = _run(tmp_path, "+vcd")
    assert "Simulation complete" in completed.stdout
    waveform = tmp_path / f"tb_{MODULE}.vcd"
    assert waveform.is_file()
    assert "$enddefinitions" in waveform.read_text(encoding="utf-8")


def test_compile_testbench_says_how_to_ask_for_the_waveform(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """``compile --testbench`` prints the run command and the ``+vcd`` plusarg for the waveform."""
    output = tmp_path / "out"
    argv = ["sc-neurocore", "compile", "dv/dt = I", "--init", "v=0", "--testbench"]
    with patch("sys.argv", [*argv, "-o", str(output), "--module-name", "probe"]):
        assert main() == 0
    printed = capsys.readouterr().out.splitlines()
    assert "  tb_probe.v — simulation testbench" in printed
    assert "  Run: iverilog -o sim probe.v tb_probe.v && vvp sim" in printed
    assert "  Add +vcd (vvp sim +vcd) to also write the waveform tb_probe.vcd" in printed
    assert not list(output.glob("*.vcd"))
