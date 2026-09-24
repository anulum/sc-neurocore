# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Execution-lane distribution contract tests

"""A lane is reported present only when this installation holds its resources."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import types
from typing import cast

import pytest

from sc_neurocore.runtime_lanes import ACCEL_ROOT, CONTRACT, LaneStatus, lane_statuses
from tests.cli_test_support import run_cli


def _by_lane(statuses: tuple[LaneStatus, ...]) -> dict[str, LaneStatus]:
    return {status.lane: status for status in statuses}


def _installed(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


def test_the_contract_names_every_lane_once_with_its_distribution() -> None:
    """Python is bundled, Rust is a separate package, the rest need a checkout."""
    assert [(contract.lane, contract.distribution) for contract in CONTRACT] == [
        ("python", "bundled"),
        ("rust", "optional-package"),
        ("julia", "source-checkout"),
        ("go", "source-checkout"),
        ("mojo", "source-checkout"),
    ]
    assert all(contract.requirement for contract in CONTRACT)


def test_an_installation_without_native_resources_reports_what_is_missing(
    tmp_path: Path,
) -> None:
    """With no kernel sources or libraries, only Python and an installed engine remain."""
    statuses = _by_lane(lane_statuses(accel_root=tmp_path))

    assert statuses["python"].resources_present
    assert statuses["rust"].resources_present == _installed("sc_neurocore_engine")
    for lane in ("julia", "go", "mojo"):
        status = statuses[lane]
        assert not status.resources_present
        assert "source checkout" in status.detail
    assert "Julia kernel sources" in statuses["julia"].detail
    assert "No built Go shared libraries" in statuses["go"].detail
    assert "No built Mojo shared libraries" in statuses["mojo"].detail


def test_native_resources_count_only_where_the_lane_keeps_them(tmp_path: Path) -> None:
    """Kernel sources and built libraries are found in their own lane directories."""
    (tmp_path / "julia" / "neurons").mkdir(parents=True)
    (tmp_path / "julia" / "neurons" / "kernel.jl").write_text("# kernel\n", encoding="utf-8")
    (tmp_path / "go" / "alpha").mkdir(parents=True)
    (tmp_path / "go" / "alpha" / "libalpha.so").write_bytes(b"\x7fELF")
    # A Mojo library filed under the Go lane is not a Mojo library.
    (tmp_path / "mojo").mkdir()
    (tmp_path / "mojo" / "libalpha.so").mkdir()

    statuses = _by_lane(lane_statuses(accel_root=tmp_path))

    assert statuses["julia"].resources_present == _installed("juliacall")
    assert statuses["go"].resources_present
    assert statuses["go"].detail == "Built Go shared libraries are present."
    assert not statuses["mojo"].resources_present


def test_this_checkout_reports_the_resources_it_holds() -> None:
    """The default root is the installed accel package, read as files."""
    statuses = _by_lane(lane_statuses())

    assert ACCEL_ROOT.name == "accel" and ACCEL_ROOT.is_dir()
    assert statuses["julia"].resources_present == (
        any((ACCEL_ROOT / "julia").rglob("*.jl")) and _installed("juliacall")
    )
    assert statuses["go"].resources_present == any((ACCEL_ROOT / "go").rglob("*.so"))
    assert statuses["mojo"].resources_present == any((ACCEL_ROOT / "mojo").rglob("*.so"))


def test_a_blocked_or_specless_module_does_not_break_the_report(tmp_path: Path) -> None:
    """``None`` in ``sys.modules`` means absent; a loaded module without a spec is present."""
    name = "sc_neurocore_engine"
    had = name in sys.modules
    saved = sys.modules.get(name)
    try:
        # Python's own import-blocking entry: the module is not importable.
        sys.modules[name] = cast(types.ModuleType, None)
        assert not _by_lane(lane_statuses(accel_root=tmp_path))["rust"].resources_present
        sys.modules[name] = types.ModuleType(name)
        assert _by_lane(lane_statuses(accel_root=tmp_path))["rust"].resources_present
    finally:
        if had:
            sys.modules[name] = cast(types.ModuleType, saved)
        else:
            sys.modules.pop(name, None)


def test_info_prints_every_lane_as_the_contract_reports_it(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The installed-runtime report carries each lane's state and requirement."""
    assert run_cli("info") == 0
    out = capsys.readouterr().out
    assert "Execution lanes:" in out
    for status in lane_statuses():
        state = "present" if status.resources_present else "absent"
        assert f"  {status.lane} ({status.distribution}): {state}. {status.detail}" in out
