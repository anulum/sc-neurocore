# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — engine and Julia availability skips go through the named gates

"""Engine and Julia availability skips go through the named gates, never ad hoc.

``tests/engine_requirement.py`` and ``tests/julia_requirement.py`` skip only when the engine,
the Julia bridge or the ``julia`` command is genuinely absent, and hosted CI, which exports
``SC_NEUROCORE_REQUIRE_ENGINE=1`` and ``SC_NEUROCORE_REQUIRE_JULIA=1``, turns that absence into
a failure. An ad hoc ``pytest.importorskip``, ``skipif`` or ``pytest.skip`` bypasses both rules:
it still skips in CI and it hides a broken engine as a skip. This module checks the executable
gate's three behaviours, then sweeps every test source for the ad hoc idioms.
"""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace
from typing import TypeGuard

import pytest

from tests import julia_requirement
from tests.julia_requirement import require_julia_executable

_ROOT = Path(__file__).resolve().parents[1]
_GATE_MODULES = frozenset({"tests/engine_requirement.py", "tests/julia_requirement.py"})
_ENGINE = "sc_neurocore_engine"
_BRIDGE = "juliacall"
#: Known gaps, each with the TODO row that closes it. The evolution-substrate fixture runs
#: Julia from a fixed Juliaup path; hosted CI installs Julia without Juliaup and does not
#: instantiate that Julia project, so routing it through the gate would turn its skip red.
_KNOWN_GAPS = {
    "tests/test_evo_substrate/multilang_parity_support.py": "SCN-EVO-JULIA-PROJECT-NOT-INSTANTIATED-IN-CI",
}


def _which(found: str | None) -> SimpleNamespace:
    """Return a stand-in for ``shutil`` whose ``which`` reports ``found``."""

    def which(_name: str) -> str | None:
        """Report the configured lookup result."""
        return found

    return SimpleNamespace(which=which)


def test_the_executable_gate_returns_the_julia_on_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """A Julia on ``PATH`` is returned, whatever the run requires."""
    monkeypatch.setattr(julia_requirement, "shutil", _which("/opt/julia/bin/julia"))
    monkeypatch.setenv("SC_NEUROCORE_REQUIRE_JULIA", "1")
    assert require_julia_executable() == "/opt/julia/bin/julia"


def test_the_executable_gate_skips_an_absent_julia_outside_ci(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without the CI requirement an absent ``julia`` skips, naming the command."""
    monkeypatch.setattr(julia_requirement, "shutil", _which(None))
    monkeypatch.delenv("SC_NEUROCORE_REQUIRE_JULIA", raising=False)
    with pytest.raises(pytest.skip.Exception, match="'julia' is not on PATH"):
        require_julia_executable()


def test_the_executable_gate_fails_an_absent_julia_when_ci_requires_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With ``SC_NEUROCORE_REQUIRE_JULIA=1`` an absent ``julia`` is a failure, not a skip."""
    monkeypatch.setattr(julia_requirement, "shutil", _which(None))
    monkeypatch.setenv("SC_NEUROCORE_REQUIRE_JULIA", "1")
    with pytest.raises(RuntimeError, match="SC_NEUROCORE_REQUIRE_JULIA=1 requires it"):
        require_julia_executable()


def _is_skip_call(node: ast.AST) -> TypeGuard[ast.Call]:
    """Return whether ``node`` calls ``pytest.skip``."""
    return isinstance(node, ast.Call) and ast.unparse(node.func) == "pytest.skip"


def _mentions_julia_lookup(node: ast.AST) -> bool:
    """Return whether an expression looks ``julia`` up on PATH or reads a Julia lane flag."""
    text = ast.unparse(node)
    return "which('julia')" in text or "_HAS_JULIA_NEURONS" in text


def _imports_native(body: list[ast.stmt]) -> bool:
    """Return whether a ``try`` body imports the engine or the Julia bridge."""
    for statement in body:
        for node in ast.walk(statement):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            if any(name.split(".")[0] in (_ENGINE, _BRIDGE) for name in names):
                return True
    return False


def ad_hoc_availability_skips(source: str) -> list[int]:
    """Return the lines of the ad hoc engine or Julia availability skips in a test source.

    Parameters
    ----------
    source : str
        Python source of one test module.

    Returns
    -------
    list of int
        Line numbers of ``pytest.importorskip`` calls on the engine or the bridge, of ``skipif``
        marks conditioned on a Julia lookup or an engine flag, and of ``pytest.skip`` calls that
        follow a Julia lookup or an engine or bridge import failure.
    """
    tree = ast.parse(source)
    lines: list[int] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = ast.unparse(node.func)
            first = node.args[0] if node.args else None
            if (
                name == "pytest.importorskip"
                and isinstance(first, ast.Constant)
                and isinstance(first.value, str)
                and first.value.split(".")[0] in (_ENGINE, _BRIDGE)
            ):
                lines.append(node.lineno)
            elif name.endswith("mark.skipif") and first is not None:
                condition = ast.unparse(first)
                if _mentions_julia_lookup(first) or "HAS_ENGINE" in condition:
                    lines.append(node.lineno)
        elif isinstance(node, ast.If) and _mentions_julia_lookup(node.test):
            lines.extend(inner.lineno for inner in ast.walk(node) if _is_skip_call(inner))
        elif isinstance(node, ast.Try) and _imports_native(node.body):
            for handler in node.handlers:
                lines.extend(inner.lineno for inner in ast.walk(handler) if _is_skip_call(inner))
    return sorted(set(lines))


@pytest.mark.parametrize(
    ("label", "source"),
    [
        ("engine importorskip", 'import pytest\npytest.importorskip("sc_neurocore_engine")\n'),
        (
            "compiled importorskip",
            'import pytest\ne = pytest.importorskip("sc_neurocore_engine.ir")\n',
        ),
        ("bridge importorskip", 'import pytest\njc = pytest.importorskip("juliacall")\n'),
        (
            "which skipif",
            'import pytest, shutil\n@pytest.mark.skipif(shutil.which("julia") is None, reason="x")\n'
            "def test_a():\n    pass\n",
        ),
        (
            "engine flag skipif",
            'import pytest\n@pytest.mark.skipif(not HAS_ENGINE, reason="x")\ndef test_a():\n    pass\n',
        ),
        (
            "which then skip",
            'import pytest, shutil\ndef test_a():\n    if shutil.which("julia") is None:\n'
            '        pytest.skip("x")\n',
        ),
        (
            "lane flag then skip",
            'import pytest\ndef test_a(mod):\n    if not mod._HAS_JULIA_NEURONS:\n        pytest.skip("x")\n',
        ),
        (
            "import failure then skip",
            "import pytest\ndef test_a():\n    try:\n        from sc_neurocore_engine import f\n"
            '    except ImportError:\n        pytest.skip("x")\n',
        ),
    ],
)
def test_the_sweep_finds_each_ad_hoc_idiom(label: str, source: str) -> None:
    """Every idiom the gates replaced is reported."""
    assert ad_hoc_availability_skips(source), label


def test_the_sweep_leaves_gated_and_unrelated_skips_alone() -> None:
    """Gate calls, other optional packages and logic skips are not reported."""
    source = (
        "import pytest\n"
        "from tests.engine_requirement import require_engine\n"
        "from tests.julia_requirement import require_julia, require_julia_executable\n"
        "require_engine()\n"
        'h5py = pytest.importorskip("h5py")\n'
        "def test_a(name):\n"
        "    julia = require_julia_executable()\n"
        "    bridge = require_julia()\n"
        '    if name == "x":\n'
        '        pytest.skip("x is currently Python-only")\n'
    )
    assert ad_hoc_availability_skips(source) == []


def test_no_test_module_skips_on_engine_or_julia_absence_ad_hoc() -> None:
    """Outside the two gate modules no test source skips on engine or Julia absence by itself."""
    offenders: dict[str, list[int]] = {}
    for path in sorted((_ROOT / "tests").rglob("*.py")):
        relative = str(path.relative_to(_ROOT))
        if relative in _GATE_MODULES:
            continue
        found = ad_hoc_availability_skips(path.read_text(encoding="utf-8"))
        if found:
            offenders[relative] = found
    assert offenders == {}, f"ad hoc engine or Julia availability skips: {offenders}"


def test_each_known_gap_is_still_open_and_named() -> None:
    """A recorded gap still holds its fixed-path Julia skip; once routed through the gate it must leave the list."""
    for relative, row in _KNOWN_GAPS.items():
        source = (_ROOT / relative).read_text(encoding="utf-8")
        assert ".juliaup" in source and "pytest.skip(" in source, (
            f"{relative} no longer has the gap that {row} records; remove it from _KNOWN_GAPS"
        )
        assert "require_julia_executable" not in source, relative
