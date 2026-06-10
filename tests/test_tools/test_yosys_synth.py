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


def test_yosys_synth_allow_skips_preserves_artifact_mode_exit_contract(monkeypatch) -> None:
    tool = _load_tool()

    monkeypatch.setattr(
        tool.shutil, "which", lambda name: "/usr/bin/yosys" if name == "yosys" else None
    )
    monkeypatch.setattr(tool, "preprocess_hdl", lambda: [])
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
    assert tool.main() == 0
