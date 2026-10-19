# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — the adaptive-precision compile honours every lowering option it is given

"""``--adaptive-precision`` compiles both datapaths with the requested modes.

The CLI accepted ``--adaptive-precision --random-precision exact`` and wrote the
standard ``randn`` unit (RIGOR NEURO-EXACT-ADAPTIVE-CLI-01: 17 of 21 HP codes
differed from the exact mode at Q3.29). The API now takes ``function_lowering``
and ``random_precision``, compiles the LP and HP datapaths with them and records
them in the manifest, and the CLI forwards both. Each datapath of the adaptive
output must be exactly the module ``compile_to_verilog`` emits for the same
options (whose exact-mode codes ``test_exact_randn`` checks against the
reference in Icarus), and options the wrapper cannot honour are refused.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from sc_neurocore.compiler.adaptive_runtime_precision import compile_adaptive_precision
from sc_neurocore.compiler.verilog_compiler import compile_to_verilog
from sc_neurocore.neurons.equation_builder import EquationNeuron, from_equations

_REPO = Path(__file__).resolve().parents[1]


def _neuron() -> EquationNeuron:
    """Return v = randn() as the CLI builds it from its arguments."""
    return from_equations(
        "dv/dt = randn()", init={"v": 0.0}, dt=1.0, method="map", rng_seed=2026, random_stream=0
    )


def _cli(*extra: str) -> tuple[int, str]:
    """Run the real compile command for v = randn(); return its code and the Verilog."""
    with tempfile.TemporaryDirectory() as d:
        done = subprocess.run(
            [
                sys.executable,
                "-m",
                "sc_neurocore.cli",
                "compile",
                "dv/dt = randn()",
                "--method",
                "map",
                "--init",
                "v=0",
                "--dt",
                "1.0",
                "--rng-seed",
                "2026",
                "--adaptive-precision",
                "--hp-width",
                "32",
                "--hp-frac",
                "29",
                "-o",
                d,
                *extra,
            ],
            capture_output=True,
            text=True,
            timeout=300,
            cwd=_REPO,
            env={"PYTHONPATH": str(_REPO / "src"), "PATH": ""},
        )
        path = Path(d) / "sc_equation_neuron.v"
        return done.returncode, path.read_text() if path.exists() else done.stdout


@pytest.mark.parametrize("precision", ["exact", "standard"])
def test_the_cli_compiles_both_datapaths_in_the_requested_mode(precision: str) -> None:
    """LP (Q8.8) and HP (Q3.29) are the modules compile_to_verilog emits for that mode."""
    code, verilog = _cli("--random-precision", precision)
    assert code == 0
    for suffix, width, fraction in (("lp", 16, 8), ("hp", 32, 29)):
        own = compile_to_verilog(
            _neuron(), f"sc_equation_neuron_{suffix}", width, fraction, random_precision=precision
        )
        other = compile_to_verilog(
            _neuron(),
            f"sc_equation_neuron_{suffix}",
            width,
            fraction,
            random_precision="standard" if precision == "exact" else "exact",
        )
        assert own in verilog and other not in verilog
    manifest = re.search(r"Adaptive Precision Manifest: (\{.*\})", verilog)
    assert manifest is not None
    assert json.loads(manifest.group(1))["random_precision"] == precision


def test_the_api_forwards_the_function_lowering() -> None:
    """function_lowering reaches both datapaths and the manifest."""
    neuron = from_equations("dv/dt = -v + tanh(I)", init={"v": 0.0}, dt=0.5)
    verilog = compile_adaptive_precision(neuron, module_name="ad", function_lowering="v2")
    for suffix, width, fraction in (("lp", 16, 8), ("hp", 32, 16)):
        module = compile_to_verilog(neuron, f"ad_{suffix}", width, fraction, function_lowering="v2")
        assert module in verilog
    assert '"function_lowering":"v2"' in verilog


def test_the_cli_refuses_a_pipeline_it_cannot_honour() -> None:
    """The wrapper times single-cycle datapaths: --pipeline with --adaptive-precision is refused."""
    code, output = _cli("--pipeline", "2")
    assert code == 1
    assert "--adaptive-precision does not support --pipeline" in output
