# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Smoke tests for public examples

"""Smoke tests for public examples that are expected to run in the base package."""

from pathlib import Path
import runpy


def test_example_02_runs(capsys):
    example = Path(__file__).resolve().parents[1] / "examples" / "02_sc_neuron_layer.py"
    runpy.run_path(str(example), run_name="__main__")
    out = capsys.readouterr().out
    assert "Dense Layer Demo" in out
    assert "Average firing rates" in out
