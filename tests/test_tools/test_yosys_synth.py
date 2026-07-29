# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest


def _load_tool() -> Any:
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
