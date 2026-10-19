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
from typing import Any

import numpy as np
import pytest

from sc_neurocore.exceptions import SCHardwareError
from sc_neurocore.network.export import export_verilog
from sc_neurocore.network.network import Network
from sc_neurocore.network.population import Population

_HAVE_ICARUS = shutil.which("iverilog") is not None and shutil.which("vvp") is not None


def _rtl_trains(
    top: Path, currents: list[float], steps: int, width: int = 16, fraction: int = 8
) -> list[list[int]]:
    """Simulate a one-population top at constant per-neuron currents; return spikes per step."""
    n = len(currents)
    packed = 0
    for k, current in enumerate(currents):
        packed |= (round(current * (1 << fraction)) & ((1 << width) - 1)) << (width * k)
    tb = top.parent / "tb.v"
    tb.write_text(
        "\n".join(
            [
                "`timescale 1ns/1ps",
                "module tb;",
                f"reg clk; reg rst_n; wire [{n - 1}:0] s;",
                f"sc_network_top uut (.clk(clk), .rst_n(rst_n), .current_0({width * n}'h{packed:x}), .spike_0(s));",
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


def test_parameters_reach_the_rtl(tmp_path: Path) -> None:
    """Two tau_mem values give two different files; the parameter is not collapsed."""
    files = {
        tau: Path(
            export_verilog(
                Network(Population("StochasticLIFNeuron", 1, params={"tau_mem": tau})),
                str(tmp_path / str(tau)),
            )
        ).read_text()
        for tau in (5.0, 50.0)
    }
    assert files[5.0] != files[50.0]


# RIGOR review 277: Q8.8 holds [-128, 127.99609375]; threshold 300 was written as 44 and a
# gain of 256 as 0, so the RTL spiked where the model did not and the other way round.
_OUT_OF_RANGE = [
    ({"tau_mem": 2.0, "dt": 1.0, "v_threshold": 300.0}, 100.0, "v_threshold=300.0"),
    ({"tau_mem": 2.0, "dt": 1.0, "v_threshold": 1.0, "resistance": 256.0}, 0.5, "gain=256.0"),
]


@pytest.mark.parametrize(("params", "current", "named"), _OUT_OF_RANGE)
def test_a_value_outside_q8_8_is_refused_before_writing(
    params: dict[str, float], current: float, named: str, tmp_path: Path
) -> None:
    """The out-of-range parameter is named and nothing is written."""
    network = Network(Population("StochasticLIFNeuron", 1, params=params))
    with pytest.raises(SCHardwareError, match=named):
        export_verilog(network, str(tmp_path / "out"))
    assert not (tmp_path / "out").exists()


def test_a_parameter_that_rounds_to_zero_is_refused(tmp_path: Path) -> None:
    """tau_mem 1000 gives a leak of 0.001, below one Q8.8 step: the leak would vanish."""
    network = Network(Population("StochasticLIFNeuron", 1, params={"tau_mem": 1000.0}))
    with pytest.raises(SCHardwareError, match="leak=0.001"):
        export_verilog(network, str(tmp_path / "out"))
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize(
    ("data_width", "fraction"), [(16, 0), (16, 16), (16, 20), (True, 8), (16.0, 8)]
)
def test_an_invalid_format_is_refused(data_width: Any, fraction: Any, tmp_path: Path) -> None:
    """The format needs integer widths with 0 < fraction < data_width."""
    network = Network(Population("StochasticLIFNeuron", 1))
    with pytest.raises(SCHardwareError, match="data_width|fraction"):
        export_verilog(network, str(tmp_path / "out"), data_width=data_width, fraction=fraction)
    assert not (tmp_path / "out").exists()


@pytest.mark.skipif(not _HAVE_ICARUS, reason="Icarus Verilog not available")
@pytest.mark.parametrize(
    ("params", "current", "width", "fraction", "spikes"),
    [
        (_OUT_OF_RANGE[0][0], 100.0, 32, 16, 0),
        (_OUT_OF_RANGE[1][0], 0.5, 32, 16, 16),
        ({"tau_mem": 2.0, "dt": 1.0, "v_threshold": 120.0}, 100.0, 16, 8, 8),
        ({"tau_mem": 2.0, "dt": 1.0, "v_threshold": 1.0, "resistance": 2.0}, 0.5, 16, 8, 16),
    ],
)
def test_a_format_that_holds_the_model_steps_as_the_model(
    params: dict[str, float],
    current: float,
    width: int,
    fraction: int,
    spikes: int,
    tmp_path: Path,
) -> None:
    """Q16.16 holds both refused models and the RTL equals Python; Q8.8 controls agree too."""
    top = Path(
        export_verilog(
            Network(Population("StochasticLIFNeuron", 1, params=params)),
            str(tmp_path),
            data_width=width,
            fraction=fraction,
        )
    )
    rtl = _rtl_trains(top, [current], 16, width, fraction)
    expected = _python_trains(Population("StochasticLIFNeuron", 1, params=params), [current], 16)
    assert rtl == expected
    assert sum(row[0] for row in expected) == spikes


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
