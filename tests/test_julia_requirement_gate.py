# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia requirement gate contract

"""The Julia bridge gate skips only on genuine absence and never masks CI.

Enforced here: the helper's behaviours (present, genuinely absent,
present-but-broken, broken dependency, CI requirement), the hosted-CI
export parsed from the workflow YAML, and a pinned inventory of the
Julia-lane parity modules that must carry a direct unconditional
module-body gate call before any other statement executes Julia-lane
code. Kernel-level failures after a healthy bridge import (missing
``.jl`` files, Julia-side load errors) are deliberately outside this
gate and stay hard failures in the parity tests themselves; the
cortical-column parity module skips on a production loader flag and is
tracked as a recorded boundary rather than gated here.
"""

from __future__ import annotations

import ast
from pathlib import Path
import sys

import pytest
import yaml

from tests.julia_requirement import JULIA_BRIDGE_MODULE, require_julia

_ROOT = Path(__file__).resolve().parents[1]

# Pinned inventory of Julia-lane modules gated by require_julia().
_GATED_JULIA_FILES = (
    "tests/test_adaptive_threshold_if_julia_parity.py",
    "tests/test_alpha_julia_parity.py",
    "tests/test_gpfa_julia_parity.py",
    "tests/test_julia_rk4_neuron_parity.py",
    "tests/test_phi_estimation_julia_parity.py",
    "tests/test_resonate_and_fire_julia_parity.py",
    "tests/test_spike_stats_dimensionality_julia_parity.py",
    "tests/test_spike_stats_sorting_quality_julia_parity.py",
)


def test_require_julia_returns_the_bridge_when_present() -> None:
    pytest.importorskip(JULIA_BRIDGE_MODULE)
    module = require_julia()
    assert module.__name__ == JULIA_BRIDGE_MODULE


def test_require_julia_skips_when_the_bridge_is_genuinely_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("SC_NEUROCORE_REQUIRE_JULIA", raising=False)
    with pytest.raises(pytest.skip.Exception):
        require_julia("fake_julia_bridge_that_does_not_exist")


def test_require_julia_hard_fails_on_a_present_but_broken_bridge(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    package = tmp_path / "fake_julia_bridge_broken"
    package.mkdir()
    (package / "__init__.py").write_text(
        'raise ImportError("bridge present but failed to load")', encoding="utf-8"
    )
    monkeypatch.delenv("SC_NEUROCORE_REQUIRE_JULIA", raising=False)
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, "fake_julia_bridge_broken", raising=False)
    with pytest.raises(ImportError, match="failed to load"):
        require_julia("fake_julia_bridge_broken")


def test_require_julia_hard_fails_on_a_broken_transitive_dependency(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    package = tmp_path / "fake_julia_bridge_broken_dep"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "runtime.py").write_text(
        "import dependency_that_is_not_installed_anywhere", encoding="utf-8"
    )
    monkeypatch.delenv("SC_NEUROCORE_REQUIRE_JULIA", raising=False)
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, "fake_julia_bridge_broken_dep", raising=False)
    with pytest.raises(ModuleNotFoundError, match="dependency_that_is_not_installed"):
        require_julia("fake_julia_bridge_broken_dep.runtime")


def test_require_julia_hard_fails_when_ci_requires_the_bridge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SC_NEUROCORE_REQUIRE_JULIA", "1")
    with pytest.raises(ModuleNotFoundError):
        require_julia("fake_julia_bridge_that_does_not_exist")


def test_hosted_ci_exports_the_julia_requirement_at_workflow_level() -> None:
    workflow = yaml.safe_load((_ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8"))
    assert workflow["env"]["SC_NEUROCORE_REQUIRE_JULIA"] == "1"


def _unconditional_gate_line(tree: ast.Module) -> int | None:
    """First top-level statement whose value IS a require_julia call."""
    for statement in tree.body:
        value: ast.expr | None = None
        if isinstance(statement, (ast.Expr, ast.Assign, ast.AnnAssign)):
            value = statement.value
        if not isinstance(value, ast.Call):
            continue
        func = value.func
        named = (isinstance(func, ast.Name) and func.id == "require_julia") or (
            isinstance(func, ast.Attribute) and func.attr == "require_julia"
        )
        if named:
            return statement.lineno
    return None


def test_every_pinned_julia_module_carries_an_unconditional_gate() -> None:
    """Dominance: a direct module-body require_julia call must exist."""
    for relative in _GATED_JULIA_FILES:
        tree = ast.parse((_ROOT / relative).read_text(encoding="utf-8"), filename=relative)
        assert _unconditional_gate_line(tree) is not None, (
            f"{relative} lost its unconditional module-body require_julia gate"
        )


def test_pinned_inventory_equals_the_derived_require_julia_set() -> None:
    """Exact set equality: silent additions fail exactly like removals."""
    derived = set()
    for path in sorted((_ROOT / "tests").rglob("test_*.py")):
        if path.name == Path(__file__).name:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        if _unconditional_gate_line(tree) is not None:
            derived.add(str(path.relative_to(_ROOT)))
    assert derived == set(_GATED_JULIA_FILES), (
        "require_julia-gated set drifted from the pinned inventory; "
        f"unpinned additions: {sorted(derived - set(_GATED_JULIA_FILES))}; "
        f"missing from tree: {sorted(set(_GATED_JULIA_FILES) - derived)}"
    )


def test_no_ad_hoc_juliacall_availability_probes_remain_in_tests() -> None:
    """The retired find_spec('juliacall') skip idiom must not reappear."""
    offenders = []
    for path in sorted((_ROOT / "tests").rglob("*.py")):
        if path.name in (Path(__file__).name, "julia_requirement.py"):
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        if "_JULIA_AVAILABLE" in text:
            offenders.append(str(path.relative_to(_ROOT)))
    assert offenders == [], f"retired _JULIA_AVAILABLE probe reappeared in: {offenders}"
