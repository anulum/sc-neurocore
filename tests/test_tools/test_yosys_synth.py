# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Native synthesis resource report contracts

"""Check synthesis status, source closures and native resource table formats."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest


def _load_tool() -> Any:
    """Load the production synthesis command without starting synthesis.

    Returns
    -------
    Any
        Command module exposing the resource reader and CLI entry point.
    """
    repo_root = Path(__file__).resolve().parents[2]
    tool_path = repo_root / "tools" / "yosys_synth.py"
    spec = importlib.util.spec_from_file_location("yosys_synth", tool_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_yosys_synth_allow_skips_rejects_zero_synthesis_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Refuse an artifact run when every native synthesis attempt fails.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Scoped command arguments, discovery and native-result substitutions.
    """
    tool = _load_tool()

    monkeypatch.setattr(
        tool.shutil, "which", lambda name: "/usr/bin/yosys" if name == "yosys" else None
    )
    monkeypatch.setattr(tool, "preprocess_hdl", lambda _module: [])
    monkeypatch.setattr(tool, "MODULES", ["sc_lif_neuron"])
    monkeypatch.setattr(
        tool,
        "run_synth",
        lambda module, sources: tool.SynthResult(
            module, 0, 0, 0, 0, False, "synthesis timed out (120s)"
        ),
    )

    monkeypatch.setattr(sys, "argv", ["yosys_synth.py"])
    assert tool.main() == 1

    monkeypatch.setattr(sys, "argv", ["yosys_synth.py", "--allow-skips"])
    assert tool.main() == 1


def test_yosys_synth_allow_skips_accepts_partial_synthesis_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Retain partial results only under the explicit artifact skip policy.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Scoped discovery and one successful plus one failed synthesis result.
    """
    tool = _load_tool()

    monkeypatch.setattr(
        tool.shutil,
        "which",
        lambda name: f"/usr/bin/{name}" if name in {"yosys", "sv2v"} else None,
    )
    monkeypatch.setattr(tool, "preprocess_hdl", lambda _module: [])
    monkeypatch.setattr(tool, "MODULES", ["working", "unsupported"])

    def _run_synth(module: str, _sources: list[Path]) -> Any:
        if module == "working":
            return tool.SynthResult(module, 1, 2, 0, 0, True)
        return tool.SynthResult(module, 0, 0, 0, 0, False, "unsupported")

    monkeypatch.setattr(tool, "run_synth", _run_synth)
    monkeypatch.setattr(sys, "argv", ["yosys_synth.py", "--allow-skips"])
    assert tool.main() == 0


def test_yosys_synth_uses_top_specific_dependency_closures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Collect exactly the HDL dependencies of each requested top.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Scoped discovery without an external SystemVerilog converter.
    """
    tool = _load_tool()
    monkeypatch.setattr(tool.shutil, "which", lambda _name: None)

    encoder_sources = tool.preprocess_hdl("sc_bitstream_encoder")
    assert [source.name for source in encoder_sources] == ["sc_bitstream_encoder.v"]

    top_sources = tool.preprocess_hdl("sc_neurocore_top")
    assert [source.stem for source in top_sources] == [
        "sc_bitstream_encoder",
        "sc_bitstream_synapse",
        "sc_dotproduct_to_current",
        "sc_lif_neuron",
        "sc_axil_cfg",
        "sc_dense_layer_core",
        "sc_firing_rate_bank",
        "sc_neurocore_top",
    ]


def test_yosys_synth_parses_cell_name_before_count_from_last_stat_block() -> None:
    """Read only the final legacy table and aggregate each FPGA cell family."""
    tool = _load_tool()
    output = """
1. Printing statistics.
     LUT6                          999
2. Printing statistics.
     LUT2                           20
     LUT3                           16
     LUT4                            1
     LUT6                           44
     FDCE                           17
     FDPE                           16
     RAMB36E1                        2
     DSP48E1                         3
"""

    assert tool.parse_stat_output(output) == {
        "luts": 81,
        "ffs": 33,
        "bram": 2,
        "dsp": 3,
    }


@pytest.mark.parametrize(
    ("output", "luts"),
    [
        (
            # Final sc_lif_neuron table from native Yosys 0.33.
            "3. Printing statistics.\n"
            "     DSP48E1                         2\n"
            "     FDCE                           49\n"
            "     LUT2                           21\n"
            "     LUT3                            3\n"
            "     LUT4                           35\n"
            "     LUT5                            2\n"
            "     LUT6                           48\n",
            109,
        ),
        (
            # Same source and synth_xilinx command under official Yosys 0.70.
            "3. Printing statistics.\n"
            "        2   DSP48E1\n"
            "       49   FDCE\n"
            "       25   LUT2\n"
            "       22   LUT3\n"
            "       36   LUT4\n"
            "        3   LUT5\n"
            "       40   LUT6\n",
            126,
        ),
    ],
)
def test_yosys_synth_reads_both_native_lif_resource_tables(output: str, luts: int) -> None:
    """Preserve measured LIF resources across the native table format change.

    Parameters
    ----------
    output : str
        Resource rows captured from the two native synthesis versions.
    luts : int
        Independently summed LUT count for that measured netlist.
    """
    assert _load_tool().parse_stat_output(output) == {
        "luts": luts,
        "ffs": 49,
        "bram": 0,
        "dsp": 2,
    }
