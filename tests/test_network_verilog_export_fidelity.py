# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — network Verilog export: real RTL that steps as the models step

"""``export_verilog`` writes RTL that elaborates and spikes as the Python neurons do.

Reproducers of RIGOR review 272: the top instantiated an ``sc_lif_array`` that
no file defined, every LIF-family model and parameter set produced the same
file, ``Network()`` wrote a superfluous port comma and an empty population
raised a raw ``IndexError`` after creating the directory. The exported neuron
modules come from the equation compiler; their spike trains are compared with
the Python neurons on dyadic parameters, where the Q8.8 arithmetic is exact.
Icarus cases skip without it.
"""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from sc_neurocore.exceptions import SCHardwareError
from sc_neurocore.network.export import export_verilog
from sc_neurocore.network.network import Network
from sc_neurocore.network.population import Population

_HAVE_ICARUS = shutil.which("iverilog") is not None and shutil.which("vvp") is not None


def _rtl_trains(top: Path, currents: list[float], steps: int) -> list[list[int]]:
    """Simulate a one-population top at constant per-neuron currents; return spikes per step."""
    n = len(currents)
    packed = 0
    for k, current in enumerate(currents):
        packed |= (round(current * 256) & 0xFFFF) << (16 * k)
    tb = top.parent / "tb.v"
    tb.write_text(
        "\n".join(
            [
                "`timescale 1ns/1ps",
                "module tb;",
                f"reg clk; reg rst_n; wire [{n - 1}:0] s;",
                f"sc_network_top uut (.clk(clk), .rst_n(rst_n), .current_0({16 * n}'h{packed:x}), .spike_0(s));",
                "initial clk = 0; always #5 clk = ~clk;",
                "integer k;",
                "initial begin",
                "  rst_n = 0; #23; rst_n = 1;",
                f'  for (k = 0; k < {steps}; k = k + 1) begin @(posedge clk); #1; $display("%b", s); end',
                "  $finish;",
                "end",
                "endmodule",
            ]
        )
    )
    sim = top.parent / "sim"
    subprocess.run(
        ["iverilog", "-g2012", "-o", str(sim), str(top), str(tb)],
        check=True,
        capture_output=True,
        timeout=120,
    )
    out = subprocess.run(["vvp", str(sim)], capture_output=True, text=True, timeout=120).stdout
    rows = [
        line.strip()
        for line in out.splitlines()
        if set(line.strip()) <= {"0", "1"} and line.strip()
    ]
    return [[int(bit) for bit in reversed(row)] for row in rows]


def _python_trains(pop: Population, currents: list[float], steps: int) -> list[list[int]]:
    """Step every neuron at its constant current and return spikes per step."""
    return [[int(neuron.step(c)) for neuron, c in zip(pop.neurons, currents)] for _ in range(steps)]


@pytest.mark.skipif(not _HAVE_ICARUS, reason="Icarus Verilog not available")
def test_stochastic_lif_export_equals_the_model(tmp_path: Path) -> None:
    """Leak 1/2 and currents of few bits: every voltage is exact in Q8.8, so the trains are equal.

    0.75 crosses on its second step and resets to an exact cycle, 1.0 crosses at once,
    and 0.25 settles near 0.5, far below the threshold in both. (A current whose fixed
    point is the threshold itself, such as 0.5, is not exact: the RTL floors the leak
    product and reaches 1.0 where the float approaches it from below.)
    """
    params = {"tau_mem": 2.0, "dt": 1.0, "v_threshold": 1.0}
    currents = [0.75, 1.0, 0.25]
    top = Path(
        export_verilog(Network(Population("StochasticLIFNeuron", 3, params=params)), str(tmp_path))
    )
    rtl = _rtl_trains(top, currents, 40)
    expected = _python_trains(Population("StochasticLIFNeuron", 3, params=params), currents, 40)
    assert rtl == expected
    assert np.asarray(expected).sum() > 0


@pytest.mark.skipif(not _HAVE_ICARUS, reason="Icarus Verilog not available")
def test_sc_lapicque_export_elaborates_and_spikes(tmp_path: Path) -> None:
    """The SC LIF profile (exact flow, quantised decay) elaborates and spikes as often as the model."""
    currents = [1.5, 3.0]
    top = Path(export_verilog(Network(Population("SCLapicqueLIFNeuron", 2)), str(tmp_path)))
    rtl = np.asarray(_rtl_trains(top, currents, 60))
    expected = np.asarray(_python_trains(Population("SCLapicqueLIFNeuron", 2), currents, 60))
    assert rtl.sum() > 0
    assert np.abs(rtl.sum(axis=0) - expected.sum(axis=0)).max() <= 1


def test_parameters_reach_the_rtl() -> None:
    """Two tau_mem values give two different files; the parameter is not collapsed."""
    from sc_neurocore.network.export import _emit_top

    files = {
        tau: _emit_top(Network(Population("StochasticLIFNeuron", 1, params={"tau_mem": tau})), "t")
        for tau in (5.0, 50.0)
    }
    assert files[5.0] != files[50.0]


@pytest.mark.parametrize(
    "build",
    [
        lambda: Network(),
        lambda: Network(Population("SCLapicqueLIFNeuron", 0)),
        lambda: Network(Population("LapicqueNeuron", 2)),
        lambda: Network(Population("StochasticLIFNeuron", 2, params={"noise_std": 0.1})),
    ],
)
def test_refusals_write_nothing(build: Callable[[], Network], tmp_path: Path) -> None:
    """Empty network, empty population, the 1907 source profile and noise: refused, no directory."""
    with pytest.raises(SCHardwareError):
        export_verilog(build(), str(tmp_path / "out"))
    assert not (tmp_path / "out").exists()


@pytest.mark.skipif(shutil.which("yosys") is None, reason="Yosys not available")
def test_yosys_elaborates_a_two_population_export(tmp_path: Path) -> None:
    """Yosys hierarchy -check finds every module; proc and check -assert pass."""
    network = Network(Population("SCLapicqueLIFNeuron", 2), Population("StochasticLIFNeuron", 3))
    top = export_verilog(network, str(tmp_path))
    script = f"read_verilog -sv {top}; hierarchy -check -top sc_network_top; proc; check -assert"
    result = subprocess.run(
        ["yosys", "-q", "-p", script], capture_output=True, text=True, timeout=300
    )
    assert result.returncode == 0, result.stdout + result.stderr
