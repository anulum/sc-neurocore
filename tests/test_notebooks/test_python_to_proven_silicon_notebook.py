# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — The Python-to-RTL notebook runs, and its RTL declares what it reads

"""Notebook 27 runs from its first code cell to its last, and its RTL reads only declared signals.

Measured before this was written: the notebook compiled the threshold
``v > V_th`` and the reset ``v = V_reset`` with parameters that held neither
name, and set them as the Python model's constants after the compile. The RTL
read two nets it never declared. Since the RTL compile refuses a name it has
no signal for, the cell stopped there. No test ran the notebook.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib
from matplotlib import pyplot
import pytest

REPO = Path(__file__).resolve().parents[2]
NOTEBOOK = REPO / "notebooks/27_python_to_proven_silicon.ipynb"


def test_the_python_to_rtl_notebook_code_executes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every code cell runs in order, and the RTL declares the threshold and reset levels.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Closes each figure where the notebook shows it, since no display exists here.
    """
    matplotlib.use("Agg")
    monkeypatch.setattr(pyplot, "show", lambda *args, **kwargs: pyplot.close("all"))
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    namespace: dict[str, Any] = {"__name__": "__notebook_test__"}
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            exec(compile("".join(cell["source"]), str(NOTEBOOK), "exec"), namespace)
    verilog = namespace["verilog"]
    assert {"V_th", "V_reset"} <= set(namespace["neuron"].parameters)
    assert "parameter signed [15:0] P_V_TH = 16'sd52736," in verilog
    assert "parameter signed [15:0] P_V_RESET = 16'sd48896" in verilog
    assert "module proven_lif" in verilog
