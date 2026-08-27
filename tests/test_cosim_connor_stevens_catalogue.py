# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — committed Connor-Stevens RTL evidence

from __future__ import annotations

from pathlib import Path
import re
import shutil
import subprocess
import tempfile

import pytest

from sc_neurocore.compiler.testbench_gen import generate_testbench
from sc_neurocore.neurons.universal_dsl import UniversalNeuron
from tests.cosim_reference_connor_stevens import _connor_stevens_hand_spike_count

ROOT = Path(__file__).resolve().parents[1]
RTL = ROOT / "hdl/formal/catalogue/sc_connor_stevens.v"
MODULE = "sc_connor_stevens"


def _compiler_rtl() -> str:
    rtl = UniversalNeuron.from_schema("connor_stevens").to_verilog(
        module_name=MODULE,
        data_width=32,
        fraction=16,
    )
    return rtl if rtl.endswith("\n") else rtl + "\n"


def test_committed_rtl_is_the_current_q1616_compiler_lowering() -> None:
    assert RTL.read_text(encoding="utf-8") == _compiler_rtl()


@pytest.mark.skipif(shutil.which("iverilog") is None, reason="iverilog is required")
def test_committed_q1616_rtl_preserves_the_bounded_event_band() -> None:
    macro_steps = 20
    substeps = 100
    current = 100.0
    source_events = _connor_stevens_hand_spike_count(macro_steps, current)
    universal = UniversalNeuron.from_schema("connor_stevens")
    schema_events = sum(universal.step(I=current) for _ in range(macro_steps))
    testbench = generate_testbench(
        universal.to_equation_neuron(),
        module_name=MODULE,
        n_steps=macro_steps * substeps,
        input_current=current,
        data_width=32,
        fraction=16,
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        work = Path(tmpdir)
        testbench_path = work / "tb.v"
        output_path = work / "tb"
        testbench_path.write_text(testbench, encoding="utf-8")
        subprocess.run(
            ["iverilog", "-g2012", "-o", str(output_path), str(RTL), str(testbench_path)],
            check=True,
            capture_output=True,
            text=True,
            timeout=60,
        )
        completed = subprocess.run(
            ["vvp", str(output_path)],
            check=True,
            capture_output=True,
            text=True,
            timeout=60,
        )
    match = re.search(r"(\d+) spikes", completed.stdout)
    assert match is not None, completed.stdout
    rtl_events = int(match.group(1))
    assert 1 < source_events < macro_steps
    assert schema_events == source_events
    assert abs(rtl_events - source_events) <= 1


@pytest.mark.skipif(shutil.which("yosys") is None, reason="Yosys is required")
def test_yosys_synthesises_committed_rtl() -> None:
    completed = subprocess.run(
        [
            "yosys",
            "-q",
            "-p",
            f"read_verilog {RTL}; synth -top {MODULE} -run begin:coarse; check; stat",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert completed.returncode == 0, completed.stderr
