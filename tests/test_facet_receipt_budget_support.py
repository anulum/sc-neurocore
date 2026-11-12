# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — The time budget of the recorder's real-process tests

"""Bind the one rule by which the recorder's real-process tests bound a command.

The recorder's tests start real processes and give each a budget measured on the
host: a baseline command is timed and multiplied. An earlier form of the rule
multiplied by the number of processors, so a host with four processors received
a fifth of the margin of a host with twenty-four, and a hosted runner timed out
on a command that had not hung. These tests hold the rule in one place: the
margin does not depend on how many processors the host has.
"""

from __future__ import annotations

import ast
import math
import os
from pathlib import Path

import pytest

from tests import facet_receipt_budget_support as support
from tests.facet_receipt_budget_support import (
    BUDGET_CEILING_SECONDS,
    BUDGET_FACTOR,
    command_budget,
    host_pressure,
    poll_interval,
    usable_processors,
)

_TESTS = Path(__file__).resolve().parent
# The six files whose budgets come from the rule. Their number may grow, never fall.
_BOUND_FILES = (
    "test_facet_receipt_checkout.py",
    "test_facet_receipt_cli.py",
    "test_facet_receipt_posix.py",
    "test_facet_receipt_posix_tools.py",
    "test_facet_receipt_process.py",
    "test_facet_receipt_provenance.py",
)


def _host(monkeypatch: pytest.MonkeyPatch, processors: int, load: float) -> None:
    """Present a host with the given usable processors and one-minute load.

    The five-minute and fifteen-minute values are zero, so a rule that read one of
    them would see an idle host.
    """
    monkeypatch.setattr(os, "sched_getaffinity", lambda _pid: set(range(processors)), raising=False)
    monkeypatch.setattr(os, "getloadavg", lambda: (load, 0.0, 0.0), raising=False)


def test_the_factor_is_the_margin_the_tests_were_qualified_with() -> None:
    """The factor is the one the earlier rule gave on the 24-processor reference host."""
    assert BUDGET_FACTOR == 25
    assert BUDGET_CEILING_SECONDS == 4 * 60 * 60


@pytest.mark.parametrize("processors", [1, 2, 4, 24, 96])
def test_an_idle_host_gets_the_same_budget_whatever_its_processor_count(
    monkeypatch: pytest.MonkeyPatch, processors: int
) -> None:
    """A small host is not given a smaller margin than a large one.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Presents the host.
    processors : int
        The number of processors the process may use.
    """
    _host(monkeypatch, processors, 0.0)
    assert usable_processors() == processors
    assert host_pressure() == 1.0
    assert command_budget(0.72) == 0.72 * 25
    assert poll_interval(0.72) == 0.72 / 25


@pytest.mark.parametrize(
    ("processors", "load", "pressure"),
    [(4, 8.0, 2.0), (4, 4.0, 1.0), (4, 2.0, 1.0), (24, 36.0, 1.5), (1, 3.0, 3.0)],
)
def test_load_above_the_processors_widens_the_budget_and_never_narrows_it(
    monkeypatch: pytest.MonkeyPatch, processors: int, load: float, pressure: float
) -> None:
    """Pressure is the load per usable processor, and at least one.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Presents the host.
    processors : int
        The number of processors the process may use.
    load : float
        The one-minute load average.
    pressure : float
        The expected pressure.
    """
    _host(monkeypatch, processors, load)
    assert host_pressure() == pressure
    assert command_budget(0.5) == 0.5 * 25 * pressure
    assert poll_interval(0.5) == 0.5 / 25


def test_the_budget_stops_at_its_ceiling(monkeypatch: pytest.MonkeyPatch) -> None:
    """A slow baseline cannot ask for more than the ceiling, the default or a given one."""
    _host(monkeypatch, 4, 0.0)
    assert command_budget(1000.0) == BUDGET_CEILING_SECONDS
    assert command_budget(1000.0, ceiling=1800.0) == 1800.0
    assert command_budget(2.0, ceiling=1800.0) == 50.0


def test_a_platform_without_an_affinity_call_counts_its_processors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without ``sched_getaffinity`` the processor count is used, and one when it is unknown."""
    monkeypatch.delattr(os, "sched_getaffinity", raising=False)
    monkeypatch.setattr(os, "cpu_count", lambda: 6)
    assert usable_processors() == 6
    monkeypatch.setattr(os, "cpu_count", lambda: None)
    assert usable_processors() == 1


def test_a_platform_without_a_load_average_has_pressure_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without ``getloadavg`` nothing is known about the load, and the budget is not widened."""
    monkeypatch.delattr(os, "sched_getaffinity", raising=False)
    monkeypatch.delattr(os, "getloadavg", raising=False)
    monkeypatch.setattr(os, "cpu_count", lambda: 4)
    assert host_pressure() == 1.0
    assert command_budget(0.4) == 0.4 * 25


@pytest.mark.parametrize("baseline", [0.0, -0.5, math.inf, math.nan])
@pytest.mark.parametrize("rule", [command_budget, poll_interval])
def test_a_baseline_that_is_not_a_positive_time_is_refused(baseline: float, rule: object) -> None:
    """A clock that did not advance, or a value that is no time, gives no budget.

    Parameters
    ----------
    baseline : float
        What stands where a measured time is expected.
    rule : object
        The function that is asked.
    """
    assert callable(rule)
    with pytest.raises(ValueError) as caught:
        rule(baseline)
    assert str(caught.value) == (
        f"the measured baseline must be a positive, finite time in seconds, got {baseline!r}"
    )


@pytest.mark.parametrize("ceiling", [0.0, -1.0, math.inf, math.nan])
def test_a_ceiling_that_is_not_a_positive_time_is_refused(ceiling: float) -> None:
    """The ceiling is a time too.

    Parameters
    ----------
    ceiling : float
        What stands where the ceiling is expected.
    """
    with pytest.raises(ValueError) as caught:
        command_budget(1.0, ceiling=ceiling)
    assert str(caught.value) == (
        f"the budget ceiling must be a positive, finite time in seconds, got {ceiling!r}"
    )


def _host_readings(tree: ast.AST) -> list[tuple[int, str]]:
    """Return every place a syntax tree reads the host's processors or load itself."""
    names = {"sched_getaffinity", "getloadavg", "cpu_count"}
    return sorted(
        (node.lineno, node.attr)
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute) and node.attr in names
    )


@pytest.mark.parametrize("name", _BOUND_FILES)
def test_the_recorder_tests_take_their_budget_from_the_rule(name: str) -> None:
    """Each bound file calls the rule and computes no budget from the processor count itself.

    Parameters
    ----------
    name : str
        A test file of the recorder that starts real processes.
    """
    source = (_TESTS / name).read_text(encoding="utf-8")
    tree = ast.parse(source)
    local_names = {
        alias.asname or alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module == "tests.facet_receipt_budget_support"
        for alias in node.names
        if alias.name == "command_budget"
    }
    assert len(local_names) == 1, name
    calls = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert local_names <= calls, name
    # What is left of the host's own readings is what a test states as a fact of the
    # run (the outcomes a stress test expects, an evidence record), never a budget.
    allowed = {
        "test_facet_receipt_posix_tools.py": ["sched_getaffinity"],
        "test_facet_receipt_provenance.py": ["sched_getaffinity"],
    }
    assert [attr for _line, attr in _host_readings(tree)] == allowed.get(name, []), name


def test_the_rule_itself_reads_the_host_in_one_place_each() -> None:
    """The support module is where the processors and the load are read."""
    tree = ast.parse(Path(support.__file__).read_text(encoding="utf-8"))
    assert [attr for _line, attr in _host_readings(tree)] == [
        "sched_getaffinity",
        "cpu_count",
        "getloadavg",
    ]
