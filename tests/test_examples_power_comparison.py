# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Toggle-count example states only what it measures

"""Keep examples/power_comparison.py's summary derived from its own measurements.

Its printed conclusion once said that clock-driven toggles stay constant and
that the event-driven model makes about 100 times fewer toggles at 0.1 %
activity, while its own rows showed both counts falling with activity and a
38-fold reduction, and it labelled a toggle reduction as a power saving.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import ModuleType

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "examples" / "power_comparison.py"


def _load_example(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """Load the example as a module; its dataclass needs the module registered."""
    name = "power_comparison_example"
    spec = importlib.util.spec_from_file_location(name, MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


def test_summary_reports_the_measured_counts_and_reduction_range(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every figure in the summary is read from the rows it summarises."""
    example = _load_example(monkeypatch)
    rows = example.compare(n_cycles=20_000)
    lines = example.conclusion(rows, n_cycles=20_000)
    text = " ".join(lines)

    first, last = rows[0], rows[-1]
    assert f"clock-driven {first.clock_toggles:,} -> {last.clock_toggles:,}" in text
    assert f"event-driven {first.event_toggles:,} -> {last.event_toggles:,}" in text
    reductions = [row.reduction for row in rows]
    assert f"{min(reductions):.1f}x to {max(reductions):.1f}x fewer toggles" in text
    assert "not a power measurement" in text


def test_clock_driven_toggles_follow_activity_in_this_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The clock-driven model's register toggles fall with input activity; they are not constant."""
    rows = _load_example(monkeypatch).compare(n_cycles=20_000)

    clock = [row.clock_toggles for row in rows]
    assert clock == sorted(clock, reverse=True)
    assert clock[0] > clock[-1]
    assert all(row.event_toggles < row.clock_toggles for row in rows)


def test_printed_report_names_toggles_not_power(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The report labels its figures as toggle counts and never as power savings."""
    _load_example(monkeypatch).main()
    out = capsys.readouterr().out

    assert "Fewer toggles:" in out
    assert "Power savings" not in out
    assert "constant regardless" not in out
    assert "not a power measurement" in out
