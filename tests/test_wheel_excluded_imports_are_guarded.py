# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Imports of wheel-excluded packages are guarded

"""Hold every import of a package the wheel leaves out behind the source-checkout refusal.

``pyproject.toml`` keeps some research packages out of the wheel. A shipped module that imports
one of them has two allowed forms:

- **inside the function that needs it,** after a call of
  :func:`sc_neurocore._source_checkout.require_source_checkout_package` that names the package;
- **at module level,** when the whole module needs the package: inside a ``try`` whose
  ``except ModuleNotFoundError`` calls the refusal and re-raises.

Any other form makes an installed wheel fail with a bare ``ModuleNotFoundError`` that does not
say why. The rule is read from the syntax tree of every shipped module, so these do not count:

- a refusal placed after its import;
- a refusal that names another package;
- a refusal in another function.

``importlib.import_module`` with a constant name counts as an import. Imports under
``if TYPE_CHECKING:`` do not run, so they are exempt. The subprocess test in
``test_wheel_shipped_subpackages_import.py`` calls the guarded features with the excluded
packages blocked; this file holds the rule for every site, called or not.
"""

from __future__ import annotations

import ast
from collections import Counter
import subprocess
import tomllib
from dataclasses import dataclass
from pathlib import Path

import pytest

from tools.benchmark_evidence_gate import tracked_files

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
GUARD = "require_source_checkout_package"

# Identify every baseline import by source and target, retaining multiplicity.
REQUIRED_EXCLUDED_IMPORTS = Counter(
    {
        ("sc_neurocore/bridges/photonic_codesign.py", "sc_neurocore.optics.photonic_emitter"): 1,
        ("sc_neurocore/experimental/physics_routes.py", "sc_neurocore.physics.heat"): 1,
        ("sc_neurocore/identity/decoder.py", "sc_neurocore.analysis"): 3,
        ("sc_neurocore/identity/substrate.py", "sc_neurocore.analysis"): 1,
        ("sc_neurocore/network/monitor.py", "sc_neurocore.analysis.spike_stats"): 1,
        ("sc_neurocore/utils/bitstreams.py", "sc_neurocore.chaos.rng"): 1,
    }
)


def excluded_packages(pyproject: Path) -> tuple[str, ...]:
    """Return the packages ``pyproject.toml`` keeps out of the wheel.

    Parameters
    ----------
    pyproject : Path
        The project's ``pyproject.toml``.

    Returns
    -------
    tuple of str
        The excluded package names, without the ``.*`` patterns, sorted.
    """
    text = pyproject.read_text(encoding="utf-8")
    find = tomllib.loads(text)["tool"]["setuptools"]["packages"]["find"]
    return tuple(sorted({entry.removesuffix(".*") for entry in find["exclude"]}))


def _within(name: str, package: str) -> bool:
    """Return whether ``name`` is ``package`` or a module inside it."""
    return name == package or name.startswith(package + ".")


@dataclass(frozen=True)
class ExcludedImport:
    """One import of an excluded package found in a shipped module.

    Attributes
    ----------
    line : int
        The line of the import.
    target : str
        The excluded module the import loads.
    guarded : bool
        Whether a refusal that names the package runs before the import.
    """

    line: int
    target: str
    guarded: bool


def _call_name(call: ast.Call) -> str | None:
    """Return the called function's simple name, if it has one."""
    if isinstance(call.func, ast.Name):
        return call.func.id
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    return None


def _string_constants(tree: ast.Module) -> dict[str, str]:
    """Return the module-level ``NAME = "text"`` assignments."""
    constants: dict[str, str] = {}
    for statement in tree.body:
        if (
            isinstance(statement, ast.Assign)
            and isinstance(statement.value, ast.Constant)
            and isinstance(statement.value.value, str)
        ):
            for target in statement.targets:
                if isinstance(target, ast.Name):
                    constants[target.id] = statement.value.value
    return constants


def _guarded_package(statement: ast.stmt, constants: dict[str, str]) -> str | None:
    """Return the package a statement's source-checkout refusal names, if it is one."""
    if not isinstance(statement, ast.Expr) or not isinstance(statement.value, ast.Call):
        return None
    call = statement.value
    if _call_name(call) != GUARD or not call.args:
        return None
    first = call.args[0]
    if isinstance(first, ast.Constant) and isinstance(first.value, str):
        return first.value
    if isinstance(first, ast.Name):
        return constants.get(first.id)
    return None


def _loaded_modules(node: ast.AST, package: str) -> list[str]:
    """Return the absolute module names an import statement or call may load."""
    if isinstance(node, ast.Import):
        return [alias.name for alias in node.names]
    if isinstance(node, ast.ImportFrom):
        parts = package.split(".") if node.level else []
        if node.level > 1:
            parts = parts[: len(parts) - (node.level - 1)]
        base = ".".join([*parts, *([node.module] if node.module else [])])
        return [base, *(f"{base}.{alias.name}" for alias in node.names)]
    if (
        isinstance(node, ast.Call)
        and _call_name(node) == "import_module"
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and isinstance(node.args[0].value, str)
    ):
        return [node.args[0].value]
    return []


def _covers(guard: str | None, target: str, excluded: tuple[str, ...]) -> bool:
    """Return whether a refusal for ``guard`` covers an import of ``target``."""
    return (
        guard is not None
        and _within(target, guard)
        and any(_within(guard, package) for package in excluded)
    )


def _is_type_checking(test: ast.expr) -> bool:
    """Return whether an ``if`` test is ``TYPE_CHECKING`` or ``typing.TYPE_CHECKING``."""
    return (isinstance(test, ast.Name) and test.id == "TYPE_CHECKING") or (
        isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING"
    )


def _handler_refuses(
    handler: ast.ExceptHandler, target: str, excluded: tuple[str, ...], constants: dict[str, str]
) -> bool:
    """Return whether an ``except`` clause refuses for ``target`` before its first bare ``raise``.

    A refusal placed after the ``raise`` never runs, so it does not count.
    """
    caught = handler.type
    names = list(caught.elts) if isinstance(caught, ast.Tuple) else [caught] if caught else []
    if not any(
        isinstance(name, ast.Name) and name.id in {"ModuleNotFoundError", "ImportError"}
        for name in names
    ):
        return False
    refused = False
    for statement in handler.body:
        if isinstance(statement, ast.Raise) and statement.exc is None:
            return refused
        if _covers(_guarded_package(statement, constants), target, excluded):
            refused = True
    return False


_BLOCK_FIELDS = ("body", "orelse", "finalbody")
_Handlers = tuple[list[ast.ExceptHandler], ...]


def scan_source(
    source: str, module: str, excluded: tuple[str, ...], *, is_package: bool = False
) -> list[ExcludedImport]:
    """Return every import of an excluded package in one module's source, with its guard state.

    A refusal guards the statements after it in its own block and in the blocks nested in them,
    within one function; a function body starts with no refusal. An import in a ``try`` body is
    also guarded when one of its ``except ModuleNotFoundError`` clauses refuses and re-raises.

    Parameters
    ----------
    source : str
        The module's source text.
    module : str
        The module's dotted name, to resolve relative imports.
    excluded : tuple of str
        The packages the wheel leaves out.
    is_package : bool
        Whether the source is a package's ``__init__``.

    Returns
    -------
    list of ExcludedImport
        One entry per import statement or ``import_module`` call that loads an excluded module,
        in source order, except those under ``if TYPE_CHECKING:``.
    """
    tree = ast.parse(source)
    constants = _string_constants(tree)
    package = module if is_package else module.rpartition(".")[0]
    found: list[ExcludedImport] = []

    def record(node: ast.AST, guards: list[str], handlers: _Handlers) -> None:
        target = next(
            (
                name
                for name in _loaded_modules(node, package)
                if any(_within(name, excluded_package) for excluded_package in excluded)
            ),
            None,
        )
        if target is None:
            return
        guarded = any(_covers(guard, target, excluded) for guard in guards) or any(
            _handler_refuses(handler, target, excluded, constants)
            for clauses in handlers
            for handler in clauses
        )
        found.append(ExcludedImport(getattr(node, "lineno", 0), target, guarded))

    def visit_block(block: list[ast.stmt], guards: list[str], handlers: _Handlers) -> None:
        local = list(guards)
        for statement in block:
            visit_statement(statement, local, handlers)
            guard = _guarded_package(statement, constants)
            if guard is not None:
                local.append(guard)

    def visit_statement(statement: ast.stmt, guards: list[str], handlers: _Handlers) -> None:
        if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
            visit_block(statement.body, [], ())
            return
        if isinstance(statement, ast.If) and _is_type_checking(statement.test):
            visit_block(statement.orelse, guards, handlers)
            return
        if isinstance(statement, (ast.Try, ast.TryStar)):
            visit_block(statement.body, guards, (*handlers, statement.handlers))
            for handler in statement.handlers:
                visit_block(handler.body, guards, handlers)
            visit_block(statement.orelse, guards, handlers)
            visit_block(statement.finalbody, guards, handlers)
            return
        if isinstance(statement, ast.Match):
            for node in ast.walk(statement.subject):
                record(node, guards, handlers)
            for case in statement.cases:
                visit_block(case.body, guards, handlers)
            return
        blocks = [
            value
            for field, value in ast.iter_fields(statement)
            if field in _BLOCK_FIELDS and isinstance(value, list)
        ]
        if not blocks:
            for node in ast.walk(statement):
                record(node, guards, handlers)
            return
        for field, value in ast.iter_fields(statement):
            if field in _BLOCK_FIELDS:
                continue
            for item in value if isinstance(value, list) else [value]:
                if isinstance(item, ast.AST):
                    for node in ast.walk(item):
                        record(node, guards, handlers)
        for nested in blocks:
            visit_block(nested, guards, handlers)

    visit_block(tree.body, [], ())
    return found


def shipped_module_paths(src: Path) -> list[Path]:
    """Return the Python files of ``sc_neurocore`` under ``src`` that the repository ships.

    In a Git work tree whose top is the parent of ``src``, these are the files Git tracks, so an
    ignored environment inside the package (an installed Mojo toolchain with its own Python
    library, say) is never read. Elsewhere, a packaged or temporary tree, every file is walked.

    Parameters
    ----------
    src : Path
        The source root that holds ``sc_neurocore``.

    Returns
    -------
    list of Path
        The modules, sorted.
    """
    tracked = tracked_files(src.parent, f"{src.name}/sc_neurocore")
    if tracked is None:
        return sorted((src / "sc_neurocore").rglob("*.py"))
    return sorted(src.parent / name for name in tracked if name.endswith(".py"))


def scan_shipped_modules(src: Path, excluded: tuple[str, ...]) -> dict[str, list[ExcludedImport]]:
    """Return the excluded-package imports of every shipped module under ``src``, by file.

    Parameters
    ----------
    src : Path
        The source root that holds ``sc_neurocore``.
    excluded : tuple of str
        The packages the wheel leaves out.

    Returns
    -------
    dict of str to list of ExcludedImport
        Files relative to ``src`` with at least one such import.
    """
    result: dict[str, list[ExcludedImport]] = {}
    for path in shipped_module_paths(src):
        parts = path.relative_to(src).with_suffix("").parts
        is_package = parts[-1] == "__init__"
        module = ".".join(parts[:-1] if is_package else parts)
        if any(_within(module, package) for package in excluded):
            continue
        found = scan_source(
            path.read_text(encoding="utf-8"), module, excluded, is_package=is_package
        )
        if found:
            result[path.relative_to(src).as_posix()] = found
    return result


EXCLUDED = excluded_packages(REPO_ROOT / "pyproject.toml")


def test_the_excluded_packages_are_read_from_pyproject() -> None:
    """The rule's package list is the wheel's own exclusion list, not a copy of it."""
    assert {
        "sc_neurocore.analysis",
        "sc_neurocore.chaos",
        "sc_neurocore.optics",
        "sc_neurocore.physics",
        "sc_neurocore.world_model",
    } <= set(EXCLUDED)


def test_every_excluded_import_in_a_shipped_module_is_guarded() -> None:
    """No shipped module imports an excluded package without the refusal before it."""
    scanned = scan_shipped_modules(SRC, EXCLUDED)
    observed = Counter(
        (path, entry.target) for path, entries in scanned.items() for entry in entries
    )
    assert observed >= REQUIRED_EXCLUDED_IMPORTS, REQUIRED_EXCLUDED_IMPORTS - observed
    unguarded = [
        f"{path}:{entry.line} imports {entry.target}"
        for path, entries in scanned.items()
        for entry in entries
        if not entry.guarded
    ]
    assert unguarded == []


def test_an_ignored_environment_inside_the_package_is_not_read(tmp_path: Path) -> None:
    """Only tracked modules are scanned: an untracked file that does not parse is never read."""
    package = tmp_path / "src" / "sc_neurocore"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text('"""Shipped."""\n', encoding="utf-8")
    environment = package / "accel" / ".pixi" / "lib"
    environment.mkdir(parents=True)
    (environment / "site.py").write_text("print 'not this Python'\n", encoding="utf-8")
    (tmp_path / ".gitignore").write_text(".pixi/\n", encoding="utf-8")
    for command in (["init", "-q"], ["add", "."]):
        subprocess.run(["git", *command], cwd=tmp_path, check=True, capture_output=True)
    assert shipped_module_paths(tmp_path / "src") == [package / "__init__.py"]
    assert scan_shipped_modules(tmp_path / "src", ("sc_neurocore.chaos",)) == {}


_ANALYSIS_GUARD = 'require_source_checkout_package("sc_neurocore.analysis", "f")\n'


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        pytest.param(
            "def f():\n    " + _ANALYSIS_GUARD + "    from sc_neurocore.analysis import x\n",
            [(3, "sc_neurocore.analysis", True)],
            id="guard-before-import",
        ),
        pytest.param(
            "def f():\n    from sc_neurocore.analysis import x\n    " + _ANALYSIS_GUARD,
            [(2, "sc_neurocore.analysis", False)],
            id="guard-after-import",
        ),
        pytest.param(
            'def f():\n    require_source_checkout_package("sc_neurocore.physics", "f")\n'
            "    from sc_neurocore.analysis import x\n",
            [(3, "sc_neurocore.analysis", False)],
            id="guard-for-another-package",
        ),
        pytest.param(
            'def f():\n    require_source_checkout_package("sc_neurocore.utils", "f")\n'
            "    from sc_neurocore.analysis import x\n",
            [(3, "sc_neurocore.analysis", False)],
            id="guard-for-a-shipped-package",
        ),
        pytest.param(
            "def g():\n    "
            + _ANALYSIS_GUARD
            + "def f():\n    from sc_neurocore.analysis import x\n",
            [(4, "sc_neurocore.analysis", False)],
            id="guard-in-another-function",
        ),
        pytest.param(
            '_A = "sc_neurocore.analysis"\ndef f():\n    require_source_checkout_package(_A, "f")\n'
            "    from sc_neurocore.analysis.spike_stats import y\n",
            [(4, "sc_neurocore.analysis.spike_stats", True)],
            id="guard-through-a-module-constant",
        ),
        pytest.param(
            "def f(flag):\n    "
            + _ANALYSIS_GUARD
            + "    if flag:\n        import sc_neurocore.analysis\n",
            [(4, "sc_neurocore.analysis", True)],
            id="guard-in-an-enclosing-block",
        ),
        pytest.param(
            "from sc_neurocore.optics import y\n",
            [(1, "sc_neurocore.optics", False)],
            id="module-level-import",
        ),
        pytest.param(
            "try:\n    from sc_neurocore.optics import y\nexcept ModuleNotFoundError:\n"
            '    require_source_checkout_package("sc_neurocore.optics", "m")\n    raise\n',
            [(2, "sc_neurocore.optics", True)],
            id="module-level-try-refuses-and-reraises",
        ),
        pytest.param(
            "try:\n    from sc_neurocore.optics import y\nexcept ModuleNotFoundError:\n"
            '    require_source_checkout_package("sc_neurocore.optics", "m")\n',
            [(2, "sc_neurocore.optics", False)],
            id="module-level-try-without-reraise",
        ),
        pytest.param(
            "try:\n    from sc_neurocore.optics import y\nexcept ModuleNotFoundError:\n    raise\n"
            '    require_source_checkout_package("sc_neurocore.optics", "m")\n',
            [(2, "sc_neurocore.optics", False)],
            id="module-level-try-reraises-before-refusing",
        ),
        pytest.param(
            "try:\n    from sc_neurocore.optics import y\nexcept ModuleNotFoundError:\n    y = None\n",
            [(2, "sc_neurocore.optics", False)],
            id="module-level-try-swallows",
        ),
        pytest.param(
            "from typing import TYPE_CHECKING\nif TYPE_CHECKING:\n    from sc_neurocore.optics import y\n",
            [],
            id="type-checking-is-exempt",
        ),
        pytest.param(
            "def f():\n    from ..world_model.spike_predictor import p\n",
            [(2, "sc_neurocore.world_model.spike_predictor", False)],
            id="relative-import-resolved",
        ),
        pytest.param(
            'import importlib\ndef f():\n    return importlib.import_module("sc_neurocore.chaos.rng")\n',
            [(3, "sc_neurocore.chaos.rng", False)],
            id="import-module-call",
        ),
        pytest.param(
            "from sc_neurocore import analysis\n",
            [(1, "sc_neurocore.analysis", False)],
            id="excluded-package-imported-by-name",
        ),
        pytest.param(
            "def f():\n    from sc_neurocore.utils import bitstreams\n",
            [],
            id="shipped-import-not-reported",
        ),
    ],
)
def test_the_reader_judges_each_form(source: str, expected: list[tuple[int, str, bool]]) -> None:
    """Each allowed and forbidden form of an excluded import is judged as the rule says."""
    found = scan_source(source, "sc_neurocore.spike_codec.module", EXCLUDED)
    assert [(entry.line, entry.target, entry.guarded) for entry in found] == expected
