# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Engine requirement gate contract

"""The engine import gate skips only on genuine absence and never masks CI.

Enforced here: the helper's four behaviours (present, genuinely
absent, present-but-broken, broken dependency), the hosted-CI
requirement export parsed from the workflow YAML, an AST order-aware
sweep proving every module-level engine import sits behind a gate call
that precedes it, a pinned inventory of the gated binding files, and
isolated subprocess proofs that an engine-less environment collects
the binding modules as skips while a broken extension stays a hard
collection error.
"""

from __future__ import annotations

import ast
import os
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

from tests.engine_requirement import ENGINE_EXTENSION_MODULE, require_engine

_ROOT = Path(__file__).resolve().parents[1]

# Pinned inventory of binding-test modules gated by require_engine().
# A file may leave this tuple only by deliberately changing its guard
# (for example to pytest.importorskip) — never by silently dropping it.
_GUARDED_BINDING_FILES = (
    "tests/test_adc_to_spike_engine_binding.py",
    "tests/test_adex_engine_binding.py",
    "tests/test_bitstream_engine_binding.py",
    "tests/test_brunel_wang_engine_binding.py",
    "tests/test_cazelles_map_engine_binding.py",
    "tests/test_chialvo_map_engine_binding.py",
    "tests/test_coba_lif_engine_binding.py",
    "tests/test_compte_wm_engine_binding.py",
    "tests/test_cordiv_engine_bindings.py",
    "tests/test_cortical_inject_engine_binding.py",
    "tests/test_courage_nekorkin_map_engine_binding.py",
    "tests/test_dcls_engine_binding.py",
    "tests/test_ei_network_engine_binding.py",
    "tests/test_engine_v3_thread_pool_configuration.py",
    "tests/test_ermentrout_kopell_map_engine_binding.py",
    "tests/test_escape_rate_engine_binding.py",
    "tests/test_evo_substrate/test_engine_bindings.py",
    "tests/test_fault_engine_bindings.py",
    "tests/test_fitzhugh_nagumo_engine_binding.py",
    "tests/test_fitzhugh_rinzel_engine_binding.py",
    "tests/test_fixed_point_lif_engine_binding.py",
    "tests/test_glif_engine_binding.py",
    "tests/test_hindmarsh_rose_engine_binding.py",
    "tests/test_ibarz_tanaka_map_engine_binding.py",
    "tests/test_iqif_engine_binding.py",
    "tests/test_izhikevich2007_engine_binding.py",
    "tests/test_izhikevich_engine_binding.py",
    "tests/test_lapicque_engine_binding.py",
    "tests/test_lgssm_engine_binding.py",
    "tests/test_mat_engine_binding.py",
    "tests/test_mckean_engine_binding.py",
    "tests/test_medvedev_map_engine_binding.py",
    "tests/test_mihalas_niebur_engine_binding.py",
    "tests/test_mixed_dense_engine_binding.py",
    "tests/test_network_runner_engine_binding.py",
    "tests/test_non_resetting_lif_engine_binding.py",
    "tests/test_ollivier_ricci_engine_binding.py",
    "tests/test_optimizer/test_engine_bindings.py",
    "tests/test_partition_engine_binding.py",
    "tests/test_pernarowski_engine_binding.py",
    "tests/test_phi_engine_bindings.py",
    "tests/test_ping_engine_binding.py",
    "tests/test_poisson_engine_binding.py",
    "tests/test_predictive_coding_engine_bindings.py",
    "tests/test_rulkov_map_engine_binding.py",
    "tests/test_sc_inference_engine_binding.py",
    "tests/test_terman_wang_engine_binding.py",
    "tests/test_wilson_cowan_engine_binding.py",
    "tests/test_wilson_hr_engine_binding.py",
)


def _is_engine_name(name: str) -> bool:
    return name == "sc_neurocore_engine" or name.startswith("sc_neurocore_engine.")


def _call_imports_engine_module(node: ast.Call) -> bool:
    """Return whether a call is importlib.import_module('sc_neurocore_engine…')."""
    func = node.func
    named = (isinstance(func, ast.Attribute) and func.attr == "import_module") or (
        isinstance(func, ast.Name) and func.id == "import_module"
    )
    if not named or not node.args:
        return False
    argument = node.args[0]
    return (
        isinstance(argument, ast.Constant)
        and isinstance(argument.value, str)
        and _is_engine_name(argument.value)
    )


def _call_is_gate(node: ast.Call) -> bool:
    """Return whether a call is require_engine(...) or an engine importorskip."""
    func = node.func
    if isinstance(func, ast.Name) and func.id == "require_engine":
        return True
    if isinstance(func, ast.Attribute) and func.attr == "require_engine":
        return True
    return (
        isinstance(func, ast.Attribute)
        and func.attr == "importorskip"
        and bool(node.args)
        and isinstance(node.args[0], ast.Constant)
        and isinstance(node.args[0].value, str)
        and _is_engine_name(node.args[0].value)
    )


class _ModuleLevelEngineScan:
    """Module-import-time engine usage: gate line and unguarded imports.

    Walks only code that executes while the module is imported (module
    body, class bodies, module-level ``try``/``if``/``with`` blocks) and
    never descends into function bodies. An engine import inside a
    ``try`` whose handlers catch ``ImportError``/``ModuleNotFoundError``
    (or broader) counts as guarded by construction.
    """

    def __init__(self) -> None:
        self.gate_line: int | None = None
        self.engine_imports: list[tuple[int, bool]] = []

    def _record_gate(self, lineno: int) -> None:
        self.gate_line = lineno if self.gate_line is None else min(self.gate_line, lineno)

    @staticmethod
    def _try_guards_imports(node: ast.Try) -> bool:
        for handler in node.handlers:
            if handler.type is None:
                return True
            names = []
            for expression in (
                handler.type.elts if isinstance(handler.type, ast.Tuple) else [handler.type]
            ):
                if isinstance(expression, ast.Name):
                    names.append(expression.id)
                elif isinstance(expression, ast.Attribute):
                    names.append(expression.attr)
            if any(
                name in ("ImportError", "ModuleNotFoundError", "Exception", "BaseException")
                for name in names
            ):
                return True
        return False

    def visit(self, statements: list[ast.stmt], guarded: bool) -> None:
        for statement in statements:
            if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if isinstance(statement, ast.Import):
                if any(_is_engine_name(alias.name) for alias in statement.names):
                    self.engine_imports.append((statement.lineno, guarded))
                continue
            if isinstance(statement, ast.ImportFrom):
                if statement.module is not None and _is_engine_name(statement.module):
                    self.engine_imports.append((statement.lineno, guarded))
                continue
            if isinstance(statement, ast.Try):
                inner = guarded or self._try_guards_imports(statement)
                self.visit(statement.body, inner)
                for handler in statement.handlers:
                    self.visit(handler.body, guarded)
                self.visit(statement.orelse, guarded)
                self.visit(statement.finalbody, guarded)
                continue
            if isinstance(statement, ast.If):
                self.visit(statement.body, guarded)
                self.visit(statement.orelse, guarded)
                continue
            if isinstance(statement, ast.With):
                self.visit(statement.body, guarded)
                continue
            if isinstance(statement, ast.ClassDef):
                self.visit(statement.body, guarded)
                continue
            for node in ast.walk(statement):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                    break
                if isinstance(node, ast.Call):
                    if _call_is_gate(node):
                        self._record_gate(node.lineno)
                    elif _call_imports_engine_module(node):
                        self.engine_imports.append((node.lineno, guarded))


def _module_level_engine_analysis(path: Path) -> tuple[int | None, int | None]:
    """Return (first gate lineno, first UNGUARDED module-level engine import)."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    scan = _ModuleLevelEngineScan()
    scan.visit(tree.body, False)
    unguarded = [lineno for lineno, guarded in scan.engine_imports if not guarded]
    return scan.gate_line, (min(unguarded) if unguarded else None)


def test_require_engine_returns_the_compiled_extension_when_present() -> None:
    module = require_engine()
    assert module.__name__ == ENGINE_EXTENSION_MODULE


def test_require_engine_skips_when_the_extension_is_genuinely_absent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A pure-Python package without the compiled extension must skip."""
    package = tmp_path / "fake_engine_pkg_absent_ext"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    monkeypatch.delenv("SC_NEUROCORE_REQUIRE_ENGINE", raising=False)
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, "fake_engine_pkg_absent_ext", raising=False)
    with pytest.raises(pytest.skip.Exception):
        require_engine("fake_engine_pkg_absent_ext.fake_extension")


def test_require_engine_skips_when_the_package_itself_is_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("SC_NEUROCORE_REQUIRE_ENGINE", raising=False)
    with pytest.raises(pytest.skip.Exception):
        require_engine("fake_engine_pkg_that_does_not_exist.fake_extension")


def test_require_engine_hard_fails_on_a_present_but_broken_extension(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A present extension that cannot load must never be skipped away."""
    package = tmp_path / "fake_engine_pkg_broken_ext"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "fake_extension.py").write_text(
        'raise ImportError("extension present but failed to load")', encoding="utf-8"
    )
    monkeypatch.delenv("SC_NEUROCORE_REQUIRE_ENGINE", raising=False)
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, "fake_engine_pkg_broken_ext", raising=False)
    with pytest.raises(ImportError, match="failed to load"):
        require_engine("fake_engine_pkg_broken_ext.fake_extension")


def test_require_engine_hard_fails_on_a_broken_transitive_dependency(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """An absent third-party dependency is a broken environment, not a skip."""
    package = tmp_path / "fake_engine_pkg_broken_dep"
    package.mkdir()
    (package / "__init__.py").write_text(
        "import dependency_that_is_not_installed_anywhere", encoding="utf-8"
    )
    monkeypatch.delenv("SC_NEUROCORE_REQUIRE_ENGINE", raising=False)
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, "fake_engine_pkg_broken_dep", raising=False)
    with pytest.raises(ModuleNotFoundError, match="dependency_that_is_not_installed"):
        require_engine("fake_engine_pkg_broken_dep.fake_extension")


def test_require_engine_hard_fails_when_ci_requires_the_engine(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SC_NEUROCORE_REQUIRE_ENGINE", "1")
    with pytest.raises(ModuleNotFoundError):
        require_engine("fake_engine_pkg_that_does_not_exist.fake_extension")


def test_hosted_ci_exports_the_engine_requirement_at_workflow_level() -> None:
    """The workflow-level env block must force the hard-fail path in CI."""
    workflow = yaml.safe_load((_ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8"))
    assert workflow["env"]["SC_NEUROCORE_REQUIRE_ENGINE"] == "1"


def test_every_pinned_binding_file_gates_before_its_engine_import() -> None:
    """Order-aware: the gate call must precede any module-level engine import."""
    assert len(_GUARDED_BINDING_FILES) == 49
    for relative in _GUARDED_BINDING_FILES:
        gate_line, engine_line = _module_level_engine_analysis(_ROOT / relative)
        assert gate_line is not None, f"{relative} lost its require_engine gate"
        if engine_line is not None:
            assert gate_line < engine_line, (
                f"{relative} gates on line {gate_line} AFTER importing the engine "
                f"on line {engine_line}"
            )


def test_no_test_module_imports_the_engine_before_a_gate() -> None:
    """AST sweep: every module-level engine import repo-wide sits behind a gate."""
    offenders = []
    for path in sorted((_ROOT / "tests").rglob("test_*.py")):
        gate_line, engine_line = _module_level_engine_analysis(path)
        if engine_line is None:
            continue
        if gate_line is None or gate_line > engine_line:
            offenders.append(f"{path.relative_to(_ROOT)}:{engine_line}")
    assert offenders == [], f"module-level engine imports without a preceding gate: {offenders}"


def _subprocess_collect(
    test_file: str, *, shadow: str, require: bool
) -> subprocess.CompletedProcess[str]:
    """Collect one binding module with the engine shadowed away.

    The repository root ``conftest.py`` prepends ``bridge/`` to
    ``sys.path``, so path shadowing alone cannot hide the installed
    extension; instead an early-loaded ``-p`` plugin injects the shadow
    package into ``sys.modules`` before any conftest runs.
    """
    env = dict(os.environ)
    env.pop("SC_NEUROCORE_REQUIRE_ENGINE", None)
    env["PYTHONPATH"] = os.pathsep.join([shadow, str(_ROOT / "src")])
    if require:
        env["SC_NEUROCORE_REQUIRE_ENGINE"] = "1"
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-p",
            "engine_shadow_plugin",
            "--collect-only",
            "-q",
            "--no-header",
            test_file,
        ],
        cwd=_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )


def _shadow_package(tmp_path: Path, *, broken_extension: bool) -> str:
    """Build a sys.modules-injected engine shadow without a working extension."""
    package = tmp_path / "shadow_pkg"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    if broken_extension:
        (package / "sc_neurocore_engine.py").write_text(
            'raise ImportError("extension present but failed to load")', encoding="utf-8"
        )
    (tmp_path / "engine_shadow_plugin.py").write_text(
        "import pathlib\n"
        "import sys\n"
        "import types\n"
        "\n"
        'package = types.ModuleType("sc_neurocore_engine")\n'
        'package.__path__ = [str(pathlib.Path(__file__).parent / "shadow_pkg")]\n'
        'sys.modules["sc_neurocore_engine"] = package\n',
        encoding="utf-8",
    )
    return str(tmp_path)


def test_engine_absent_environment_collects_binding_module_as_skip(tmp_path: Path) -> None:
    """Isolated proof: without the extension the module skips at collection."""
    shadow = _shadow_package(tmp_path, broken_extension=False)
    completed = _subprocess_collect(
        "tests/test_mckean_engine_binding.py", shadow=shadow, require=False
    )
    # Exit code 5 is pytest's "no tests collected" — the whole module
    # skipped cleanly; anything else is a collection failure.
    assert completed.returncode == 5, completed.stdout + completed.stderr
    assert "no tests collected" in completed.stdout
    assert "error" not in (completed.stdout + completed.stderr).lower()


def test_engine_broken_extension_is_a_hard_collection_error(tmp_path: Path) -> None:
    """Isolated proof: a present-but-broken extension must fail collection."""
    shadow = _shadow_package(tmp_path, broken_extension=True)
    completed = _subprocess_collect(
        "tests/test_mckean_engine_binding.py", shadow=shadow, require=False
    )
    assert completed.returncode != 0
    assert "failed to load" in completed.stdout + completed.stderr


def test_engine_absent_with_ci_requirement_is_a_hard_collection_error(
    tmp_path: Path,
) -> None:
    """Isolated proof: CI's requirement turns absence into a collection error."""
    shadow = _shadow_package(tmp_path, broken_extension=False)
    completed = _subprocess_collect(
        "tests/test_mckean_engine_binding.py", shadow=shadow, require=True
    )
    assert completed.returncode != 0
    assert "ModuleNotFoundError" in completed.stdout + completed.stderr
