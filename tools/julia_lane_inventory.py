# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia lane inventory

"""Classify every tracked Julia source by its declarations, its Python loaders and whether Julia can parse it.

The package's Julia tree mixes kernels that maintained Python code loads with research mirrors and machine
transcripts of Python originals, and many of the transcripts are not Julia at all. This inventory answers, by rule,
which file is which. It is written to ``docs/_generated/julia_lane_inventory.json`` and checked for drift.

Three facts are recorded for each tracked ``.jl`` file under ``src/sc_neurocore``, each as its own flag:

``authoritative``
    The file is listed in ``AUTHORITATIVE_JULIA_ENTRYPOINTS`` of ``accel/julia/__init__.py``.
``loaded``
    A tracked Python module under ``src/sc_neurocore``, ``benchmarks`` or ``tools`` builds the file's exact path.
``mirror``
    The file matches one of ``NON_AUTHORITATIVE_JULIA_MIRROR_GLOBS``, segment by segment (``*`` never crosses ``/``).

One class is derived from them by this precedence: ``authoritative_entrypoint``, ``loaded_by_python``,
``declared_mirror``, ``undeclared``. The flags stay in the record, so a loaded file under a mirror glob remains
visible.

Each file also records whether Julia parses it: ``Meta.parseall`` reads the text, every ``:error`` and
``:incomplete`` node of the result counts as a refusal, and nothing is evaluated. The record names the Julia version.

Every file that Julia does not parse is a transcript, and the record names the Python module it copies, so it is
the worklist for replacing transcripts with real code. The module is found by rule, never guessed:

``header``
    The header says ``Julia acceleration for <module>``, and ``src/sc_neurocore/<module>.py`` (or the package
    ``<module>/__init__.py``) is tracked.
``header_underscore_dir``
    The header's first segment joins two directories with ``_`` (``adapters_holonomic`` for ``adapters/holonomic``),
    and exactly one way of reading one ``_`` as ``/`` names a tracked module.
``ambiguous``
    Several such modules exist; all are listed as candidates and none is chosen.
``none``
    The header names no module, or no tracked module matches.

The modules searched are the tracked ``.py`` files of ``src/sc_neurocore`` outside ``accel/julia``. Each transcript
records ``verbatim_line_share`` as ``"n/d"``: of its ``d`` non-empty, non-comment lines, ``n`` appear, stripped, as a
stripped line of the chosen module. A file Julia parses records the rule ``not_a_transcript``.

Loader paths are found by reading each Python module's AST.

- **What is followed.** Every string constant ending in ``.jl`` is followed outward through the expressions that
  build a path: ``os.path.join``, ``dirname``, ``abspath``, ``normpath`` and ``realpath``; ``Path(...)``, ``/``,
  ``.parent``, ``.parents[i]``, ``.with_name`` and ``.resolve()``; ``str(...)``; and calls to small module-level
  helpers such as ``_accel_path(*parts)``.
- **How it is evaluated.** The outermost such expression is evaluated with ``__file__`` bound to the module's path.
  It may use module-level and function-local assignments, imported module constants and helper functions.
- **The three outcomes.**
  - A path that names a tracked ``.jl`` file is ``resolved``. Only a resolved reference makes a file ``loaded``.
  - A path inside the repository that names no tracked file is ``missing``.
  - A literal that is itself the repository-relative path of a tracked ``.jl`` file, such as an entry in a
    benchmark's list of source files, is ``named``. It names the file without building a path that loads it.
  - Anything else is ``unresolved``, listed with its module and line and never dropped. That covers names collected
    in a list and joined later, a name completed with ``.jl`` (an f-string, ``name + ".jl"``,
    ``.with_suffix(".jl")``), a path outside the repository, and an expression the rules do not cover.
- **What is not a loader.** The package's own declaration tuples, patterns containing ``*``, the bare suffix
  ``.jl``, and text containing a space are declarations or prose.

Only files Git tracks are read, so an untracked file or an ignored directory changes nothing. The root must be the top
of a Git work tree, a tracked file missing from it is refused, and a tree without tracked Python modules is refused.

Parse verdicts need Julia.

- **``--write``** parses every file and refuses to run without Julia.
- **``--check`` with Julia** parses every file again.
- **``--check`` without Julia** carries each file's verdict from the committed record when the file's sha256 is
  unchanged. A changed or new file is drift. Everything else (the tracked files, the declarations, the loader
  references and the classes) is recomputed on every check.
- **The output.** The check prints which mode ran. ``--require-julia`` makes an absent Julia a refusal.

Usage::

    python tools/julia_lane_inventory.py --write
    python tools/julia_lane_inventory.py --check [--require-julia]
    python tools/julia_lane_inventory.py --summary
"""

from __future__ import annotations

import argparse
import ast
import fnmatch
import hashlib
import json
import posixpath
import re
import shutil
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import TypedDict

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from tools.mojo_lane_inventory import string_tuples, tracked

DEFAULT_OUTPUT = "docs/_generated/julia_lane_inventory.json"
PACKAGE = "src/sc_neurocore"
JULIA_ROOT = "src/sc_neurocore/accel/julia"
DECLARATIONS = "src/sc_neurocore/accel/julia/__init__.py"
LOADER_DIRECTORIES = ("src/sc_neurocore", "benchmarks", "tools")
CLASSES = ("authoritative_entrypoint", "loaded_by_python", "declared_mirror", "undeclared")
STATUSES = ("resolved", "named", "missing", "unresolved")
#: The rules that name the Python module a transcript copies, in the order they are tried.
TRANSCRIPTION_RULES = ("header", "header_underscore_dir", "ambiguous", "none")
_JULIA_HEADER = re.compile(r"Julia acceleration for (\S+)")
_HEADER_LINES = 40
PARSE_METHOD = "Meta.parseall; every :error and :incomplete node is a refusal; nothing is evaluated"
_DECLARATION_NAMES = ("AUTHORITATIVE_JULIA_ENTRYPOINTS", "NON_AUTHORITATIVE_JULIA_MIRROR_GLOBS")
_MAX_DEPTH = 24

#: Julia program that parses each repository-relative path read from standard input, one line per file.
JULIA_PARSER = r"""
println("VERSION\t", VERSION)
for rel in eachline(stdin)
    accepted = try
        stack = Any[Meta.parseall(read(rel, String); filename=rel)]
        clean = true
        while !isempty(stack)
            node = pop!(stack)
            if node isa Expr
                if node.head === :error || node.head === :incomplete
                    clean = false
                    break
                end
                append!(stack, node.args)
            end
        end
        clean
    catch
        false
    end
    println(accepted ? "PARSES\t" : "REFUSED\t", rel)
end
"""

#: One Julia source and the facts the inventory records about it.
Source = TypedDict(
    "Source",
    {
        "path": str,
        "sha256": str,
        "parses": bool,
        "authoritative": bool,
        "loaded": bool,
        "mirror": bool,
        "class": str,
        "loaders": list[str],
        "transcribed_module": str | None,
        "transcription_rule": str,
        "verbatim_line_share": str | None,
        "transcription_candidates": list[str],
    },
)


class Transcription(TypedDict):
    """The Python module a transcript copies, the rule that names it and the evidence."""

    transcribed_module: str | None
    transcription_rule: str
    verbatim_line_share: str | None
    transcription_candidates: list[str]


#: The record of a file that Julia parses.
NOT_A_TRANSCRIPT: Transcription = {
    "transcribed_module": None,
    "transcription_rule": "not_a_transcript",
    "verbatim_line_share": None,
    "transcription_candidates": [],
}


class Reference(TypedDict):
    """One ``.jl`` path a Python module builds, and what it resolves to."""

    python: str
    line: int
    status: str
    target: str | None


class Julia(TypedDict):
    """The parser that produced the verdicts."""

    version: str
    parse_method: str


class Inventory(TypedDict):
    """The whole inventory as it is committed."""

    generator: str
    julia: Julia
    counts: dict[str, int]
    sources: list[Source]
    references: list[Reference]


#: A parser takes the root and the repository-relative paths and returns the Julia version and a verdict per path.
Parser = Callable[[Path, Sequence[str]], tuple[str, dict[str, bool]]]


class JuliaUnavailable(RuntimeError):
    """No ``julia`` program can be started, or it did not answer for every file."""


class _Unresolved(Exception):
    """An expression lies outside the path rules."""


def parse_with_julia(root: Path, paths: Sequence[str]) -> tuple[str, dict[str, bool]]:
    """Parse each file with the ``julia`` program on ``PATH``, evaluating nothing.

    Parameters
    ----------
    root : Path
        The repository root; the paths are relative to it.
    paths : sequence of str
        Repository-relative ``.jl`` paths.

    Returns
    -------
    tuple
        The Julia version and, for each path, whether it parses.

    Raises
    ------
    JuliaUnavailable
        When no ``julia`` program is found, it fails, or its answer does not cover every path exactly once.
    """
    program = shutil.which("julia")
    if program is None:
        raise JuliaUnavailable("no julia program on PATH; parse verdicts cannot be computed here")
    completed = subprocess.run(
        [program, "--startup-file=no", "--history-file=no", "-O0", "-e", JULIA_PARSER],
        input="".join(f"{path}\n" for path in paths),
        capture_output=True,
        text=True,
        cwd=root,
        check=False,
    )
    if completed.returncode != 0:
        raise JuliaUnavailable(
            f"julia exited {completed.returncode}: {completed.stderr.strip()[:300]}"
        )
    version = ""
    verdicts: dict[str, bool] = {}
    for line in completed.stdout.splitlines():
        tag, _, value = line.partition("\t")
        if tag == "VERSION":
            version = value
        elif tag in ("PARSES", "REFUSED"):
            if value in verdicts:
                raise JuliaUnavailable(f"julia answered twice for {value}")
            verdicts[value] = tag == "PARSES"
    if not version or set(verdicts) != set(paths):
        raise JuliaUnavailable("julia did not answer for every file")
    return version, verdicts


def declarations(root: Path) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return the package's authoritative Julia entrypoints and its mirror globs.

    Parameters
    ----------
    root : Path
        The repository root.

    Returns
    -------
    tuple
        The entrypoints and the globs, relative to ``accel/julia``; both empty without the declaration module.
    """
    path = root / DECLARATIONS
    if not path.is_file():
        return (), ()
    found = string_tuples(path)
    return found.get(_DECLARATION_NAMES[0], ()), found.get(_DECLARATION_NAMES[1], ())


def _module_name(relative: str) -> str | None:
    """Return the dotted module name of a tracked ``src/`` or top-level Python file."""
    parts = relative.removesuffix(".py").split("/")
    if parts[0] == "src":
        parts = parts[1:]
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts) if parts else None


class _Module:
    """One parsed Python module: its assignments, functions and imports."""

    def __init__(self, root: Path, relative: str, tree: ast.Module) -> None:
        self.relative = relative
        self.file = str(root / relative)
        self.tree = tree
        name = _module_name(relative) or ""
        self.package = name if relative.endswith("__init__.py") else name.rpartition(".")[0]
        self.assigns: dict[str, list[ast.expr]] = {}
        self.functions: dict[str, ast.FunctionDef] = {}
        self.imports: dict[str, tuple[str, str | None]] = {}
        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                self.functions[node.name] = node
            elif isinstance(node, (ast.Assign, ast.AnnAssign)) and node.value is not None:
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for target in targets:
                    if isinstance(target, ast.Name):
                        self.assigns.setdefault(target.id, []).append(node.value)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.asname:
                        self.imports[alias.asname] = (alias.name, None)
                    else:
                        head = alias.name.split(".")[0]
                        self.imports[head] = (head, None)
            elif isinstance(node, ast.ImportFrom):
                base = self._absolute(node.module, node.level)
                for alias in node.names:
                    self.imports[alias.asname or alias.name] = (base, alias.name)

    def _absolute(self, module: str | None, level: int) -> str:
        """Return the absolute dotted name of an import, resolving its relative level."""
        if level == 0:
            return module or ""
        parts = self.package.split(".")
        anchor = parts[: len(parts) - (level - 1)] if level > 1 else parts
        return ".".join([*anchor, *([module] if module else [])])


class _Function:
    """A module-level helper whose body is assignments and one return."""

    def __init__(self, module: _Module, node: ast.FunctionDef) -> None:
        self.module = module
        self.node = node


class _Evaluator:
    """Evaluate path expressions of the tracked Python modules."""

    _OS_PATH = ("join", "dirname", "abspath", "normpath", "realpath")
    _PATH_METHODS = ("with_name", "resolve", "joinpath")

    def __init__(self, root: Path, modules: Mapping[str, _Module]) -> None:
        self.root = root
        self.modules = modules
        self.by_name = {_module_name(relative): module for relative, module in modules.items()}

    def module_object(self, dotted: str) -> object:
        """Return the value an imported dotted module stands for."""
        if dotted in ("os", "os.path"):
            return ("os.path",)
        if dotted == "pathlib":
            return ("pathlib",)
        module = self.by_name.get(dotted)
        if module is None:
            raise _Unresolved(dotted)
        return module

    def lookup(self, name: str, module: _Module, scope: dict[str, object], depth: int) -> object:
        """Return the value of a name in a function scope, then the module, then its imports."""
        if name in scope:
            return scope[name]
        if name == "__file__":
            return module.file
        if name in module.assigns:
            values = {
                repr(self.evaluate(value, module, {}, depth + 1)) for value in module.assigns[name]
            }
            if len(values) != 1:
                raise _Unresolved(name)
            return self.evaluate(module.assigns[name][0], module, {}, depth + 1)
        if name in module.functions:
            return _Function(module, module.functions[name])
        if name in module.imports:
            source, attribute = module.imports[name]
            if attribute is None:
                return self.module_object(source)
            if source == "pathlib" and attribute == "Path":
                return ("Path",)
            if source == "os" and attribute == "path":
                return ("os.path",)
            if source == "os.path" and attribute in self._OS_PATH:
                return (f"os.path.{attribute}",)
            submodule = self.by_name.get(f"{source}.{attribute}")
            if submodule is not None:
                return submodule
            target = self.by_name.get(source)
            if target is None or target is module:
                raise _Unresolved(name)
            return self.lookup(attribute, target, {}, depth + 1)
        if name == "str":
            return ("str",)
        raise _Unresolved(name)

    def evaluate(
        self, node: ast.expr, module: _Module, scope: dict[str, object], depth: int = 0
    ) -> object:
        """Return the value of one expression under the path rules.

        Raises
        ------
        _Unresolved
            When the expression lies outside the rules.
        """
        if depth > _MAX_DEPTH:
            raise _Unresolved("depth")
        if isinstance(node, ast.Constant) and isinstance(node.value, (str, int)):
            return node.value
        if isinstance(node, ast.Name):
            return self.lookup(node.id, module, scope, depth)
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
            return posixpath.join(
                self._text(self.evaluate(node.left, module, scope, depth + 1)),
                self._text(self.evaluate(node.right, module, scope, depth + 1)),
            )
        if isinstance(node, ast.Attribute):
            return self._attribute(
                self.evaluate(node.value, module, scope, depth + 1), node.attr, depth
            )
        if isinstance(node, ast.Subscript):
            value = self.evaluate(node.value, module, scope, depth + 1)
            index = self.evaluate(node.slice, module, scope, depth + 1)
            if isinstance(value, tuple) and value[:1] == ("parents",) and isinstance(index, int):
                path = self._text(value[1])
                for _ in range(index + 1):
                    path = posixpath.dirname(path)
                return path
            raise _Unresolved("subscript")
        if isinstance(node, ast.Call):
            return self._call(node, module, scope, depth)
        raise _Unresolved(type(node).__name__)

    def _arguments(
        self, node: ast.Call, module: _Module, scope: dict[str, object], depth: int
    ) -> list[object]:
        """Return the evaluated positional arguments of a call, expanding starred tuples."""
        if node.keywords:
            raise _Unresolved("keywords")
        values: list[object] = []
        for argument in node.args:
            if isinstance(argument, ast.Starred):
                expanded = self.evaluate(argument.value, module, scope, depth + 1)
                if not (isinstance(expanded, tuple) and expanded[:1] == ("args",)):
                    raise _Unresolved("starred")
                values.extend(expanded[1:])
            else:
                values.append(self.evaluate(argument, module, scope, depth + 1))
        return values

    def _call(
        self, node: ast.Call, module: _Module, scope: dict[str, object], depth: int
    ) -> object:
        """Return the value of a call to a path function, a path method or a module helper."""
        function = self.evaluate(node.func, module, scope, depth + 1)
        arguments = self._arguments(node, module, scope, depth)
        if isinstance(function, tuple):
            kind = function[0]
            if kind in ("Path", "str") and arguments:
                return posixpath.join(*(self._text(item) for item in arguments))
            if kind == "os.path.join" and arguments:
                return posixpath.join(*(self._text(item) for item in arguments))
            if (
                kind in ("os.path.abspath", "os.path.normpath", "os.path.realpath")
                and len(arguments) == 1
            ):
                return posixpath.normpath(self._text(arguments[0]))
            if kind == "os.path.dirname" and len(arguments) == 1:
                return posixpath.dirname(self._text(arguments[0]))
            if kind == "method" and function[1] == "resolve" and not arguments:
                return posixpath.normpath(self._text(function[2]))
            if kind == "method" and function[1] == "with_name" and len(arguments) == 1:
                return posixpath.join(
                    posixpath.dirname(self._text(function[2])), self._text(arguments[0])
                )
            if kind == "method" and function[1] == "joinpath" and arguments:
                return posixpath.join(
                    self._text(function[2]), *(self._text(item) for item in arguments)
                )
        if isinstance(function, _Function):
            return self._inline(function, arguments, depth)
        raise _Unresolved("call")

    def _attribute(self, value: object, attribute: str, depth: int) -> object:
        """Return an attribute of an evaluated value under the path rules."""
        if isinstance(value, tuple) and value[0] == "os.path" and attribute == "path":
            return ("os.path",)
        if isinstance(value, tuple) and value[0] == "os.path" and attribute in self._OS_PATH:
            return (f"os.path.{attribute}",)
        if isinstance(value, tuple) and value[0] == "pathlib" and attribute == "Path":
            return ("Path",)
        if isinstance(value, _Module):
            return self.lookup(attribute, value, {}, depth + 1)
        if isinstance(value, str):
            if attribute == "parent":
                return posixpath.dirname(value)
            if attribute == "parents":
                return ("parents", value)
            if attribute in self._PATH_METHODS:
                return ("method", attribute, value)
        raise _Unresolved(f"attribute {attribute}")

    def _inline(self, function: _Function, arguments: list[object], depth: int) -> object:
        """Evaluate a helper whose body is assignments and one final return, with its arguments bound."""
        node = function.node
        body = list(node.body)
        if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
            body = body[1:]
        if not body or not isinstance(body[-1], ast.Return) or body[-1].value is None:
            raise _Unresolved("helper body")
        parameters = [argument.arg for argument in node.args.args]
        if node.args.kwonlyargs or node.args.kwarg or node.args.defaults:
            raise _Unresolved("helper signature")
        if len(arguments) < len(parameters) or (
            len(arguments) > len(parameters) and node.args.vararg is None
        ):
            raise _Unresolved("helper arguments")
        scope: dict[str, object] = dict(zip(parameters, arguments, strict=False))
        if node.args.vararg is not None:
            scope[node.args.vararg.arg] = ("args", *arguments[len(parameters) :])
        for statement in body[:-1]:
            if not (
                isinstance(statement, ast.Assign)
                and len(statement.targets) == 1
                and isinstance(statement.targets[0], ast.Name)
            ):
                raise _Unresolved("helper statement")
            scope[statement.targets[0].id] = self.evaluate(
                statement.value, function.module, scope, depth + 1
            )
        return self.evaluate(body[-1].value, function.module, scope, depth + 1)

    @staticmethod
    def _text(value: object) -> str:
        """Return a value that must be a path or a string."""
        if not isinstance(value, str):
            raise _Unresolved("not a path")
        return value


def _parents(tree: ast.Module) -> dict[ast.AST, ast.AST]:
    """Return each node's parent."""
    found: dict[ast.AST, ast.AST] = {}
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            found[child] = parent
    return found


def _wraps_a_path(parent: ast.AST, child: ast.AST) -> bool:
    """Return whether ``parent`` is an expression that a path built by ``child`` may continue into."""
    return (
        (isinstance(parent, ast.BinOp) and isinstance(parent.op, ast.Div))
        or (isinstance(parent, ast.Call) and (child in parent.args or parent.func is child))
        or isinstance(parent, ast.Starred)
        or (isinstance(parent, ast.Attribute) and parent.attr in _Evaluator._PATH_METHODS)
    )


def _path_expression(
    literal: ast.AST, parents: Mapping[ast.AST, ast.AST], builds_path: Callable[[ast.Call], bool]
) -> ast.AST:
    """Return the outermost expression that builds a path around a ``.jl`` literal."""
    node = literal
    while node in parents and _wraps_a_path(parents[node], node):
        node = parents[node]
    # A call that merely consumes the path (an ``include`` or ``open``) is not part of it.
    while isinstance(node, ast.Call) and not builds_path(node):
        inner = [argument for argument in node.args if _contains(argument, literal)]
        if len(inner) != 1:
            break
        node = inner[0]
    return node


def _path_builder(evaluator: _Evaluator, module: _Module) -> Callable[[ast.Call], bool]:
    """Return a test of whether a call in ``module`` builds a path rather than consuming one."""

    def builds_path(node: ast.Call) -> bool:
        function = node.func
        if isinstance(function, ast.Attribute):
            return function.attr in (*_Evaluator._OS_PATH, *_Evaluator._PATH_METHODS, "Path")
        if isinstance(function, ast.Name):
            if function.id in ("Path", "str"):
                return True
            try:
                return isinstance(evaluator.lookup(function.id, module, {}, 1), _Function)
            except _Unresolved:
                return False
        return False

    return builds_path


def _contains(node: ast.AST, target: ast.AST) -> bool:
    """Return whether ``target`` lies inside ``node``."""
    return any(child is target for child in ast.walk(node))


def _enclosing_function(
    node: ast.AST, parents: Mapping[ast.AST, ast.AST]
) -> ast.FunctionDef | None:
    """Return the innermost function that contains a node."""
    while node in parents:
        node = parents[node]
        if isinstance(node, ast.FunctionDef):
            return node
    return None


def _local_scope(
    function: ast.FunctionDef | None, evaluator: _Evaluator, module: _Module
) -> dict[str, object]:
    """Return the names a function assigns exactly once that evaluate under the rules."""
    if function is None:
        return {}
    assigned: dict[str, list[ast.expr]] = {}
    for node in ast.walk(function):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    assigned.setdefault(target.id, []).append(node.value)
    single = {name: values[0] for name, values in assigned.items() if len(values) == 1}
    scope: dict[str, object] = {}
    progress = True
    while progress:
        progress = False
        for name, value in single.items():
            if name in scope:
                continue
            try:
                scope[name] = evaluator.evaluate(value, module, scope, 1)
            except _Unresolved:
                continue
            progress = True
    return scope


def _reference_anchor(
    node: ast.AST, parents: Mapping[ast.AST, ast.AST], skipped: set[int]
) -> ast.expr | None:
    """Return the node a ``.jl`` reference is anchored on, or ``None`` when ``node`` is not one.

    A string constant ending in ``.jl`` anchors itself, unless it is a declaration, a pattern or prose. A name
    completed with ``.jl`` anchors on the expression that completes it: an f-string, ``name + ".jl"`` or
    ``.with_suffix(".jl")``. Those are never evaluated as constants, so they are listed rather than dropped.
    """
    if not (isinstance(node, ast.Constant) and isinstance(node.value, str)):
        return None
    value = node.value
    if not value.endswith(".jl") or id(node) in skipped:
        return None
    parent = parents.get(node)
    if isinstance(parent, ast.JoinedStr):
        return parent
    if isinstance(parent, ast.BinOp) and isinstance(parent.op, ast.Add) and parent.right is node:
        return parent
    if (
        isinstance(parent, ast.Call)
        and isinstance(parent.func, ast.Attribute)
        and parent.func.attr == "with_suffix"
    ):
        return parent
    if value == ".jl" or "*" in value or any(character.isspace() for character in value):
        return None
    return node


def _declaration_literals(tree: ast.Module) -> set[int]:
    """Return the ids of the string constants inside the package's own declaration tuples."""
    found: set[int] = set()
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)) and node.value is not None:
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(
                isinstance(target, ast.Name) and target.id in _DECLARATION_NAMES
                for target in targets
            ):
                found.update(id(child) for child in ast.walk(node.value))
    return found


def loader_modules(root: Path) -> dict[str, _Module]:
    """Return the tracked Python modules that may load Julia files, parsed.

    Parameters
    ----------
    root : Path
        The repository root.

    Returns
    -------
    dict
        Each repository-relative path and its parsed module.

    Raises
    ------
    ValueError
        When the root is not the top of a Git work tree, a tracked file is missing, or no Python module is tracked.
    """
    modules: dict[str, _Module] = {}
    for directory in LOADER_DIRECTORIES:
        for path in tracked(root, directory, ".py"):
            relative = path.relative_to(root).as_posix()
            modules[relative] = _Module(root, relative, ast.parse(path.read_text(encoding="utf-8")))
    if not modules:
        raise ValueError("no tracked Python module was found; loader paths cannot be resolved")
    return modules


def references(root: Path, julia_sources: set[str]) -> list[Reference]:
    """Return every ``.jl`` path the tracked Python modules build, resolved under the rules.

    Parameters
    ----------
    root : Path
        The repository root.
    julia_sources : set of str
        The tracked ``.jl`` files, repository-relative.

    Returns
    -------
    list of Reference
        One record per ``.jl`` literal outside the declarations, sorted by module and line.
    """
    root = root.resolve()
    modules = loader_modules(root)
    evaluator = _Evaluator(root, modules)
    prefix = str(root) + "/"
    found: list[Reference] = []
    for relative, module in sorted(modules.items()):
        parents = _parents(module.tree)
        skipped = _declaration_literals(module.tree) if relative == DECLARATIONS else set()
        anchors: dict[int, ast.expr] = {}
        for node in ast.walk(module.tree):
            anchor = _reference_anchor(node, parents, skipped)
            if anchor is not None:
                anchors.setdefault(id(anchor), anchor)
        for literal in sorted(anchors.values(), key=lambda item: (item.lineno, item.col_offset)):
            expression = _path_expression(literal, parents, _path_builder(evaluator, module))
            scope = _local_scope(_enclosing_function(literal, parents), evaluator, module)
            status, target = "unresolved", None
            value: object = None
            if isinstance(expression, ast.expr):
                try:
                    value = evaluator.evaluate(expression, module, scope)
                except _Unresolved:
                    value = None
            if isinstance(value, str) and value.startswith(prefix):
                candidate = posixpath.normpath(value)[len(prefix) :]
                status, target = (
                    ("resolved", candidate)
                    if candidate in julia_sources
                    else ("missing", candidate)
                )
            elif (
                expression is literal
                and isinstance(literal, ast.Constant)
                and literal.value in julia_sources
            ):
                status, target = "named", literal.value
            found.append(
                {"python": relative, "line": literal.lineno, "status": status, "target": target}
            )
    return found


def _glob_matches(path: str, pattern: str) -> bool:
    """Return whether a relative path matches a glob segment by segment, so that ``*`` never crosses ``/``."""
    parts, wanted = path.split("/"), pattern.split("/")
    return len(parts) == len(wanted) and all(
        fnmatch.fnmatchcase(part, glob) for part, glob in zip(parts, wanted, strict=True)
    )


def package_modules(root: Path) -> set[str]:
    """Return the tracked Python modules of the package outside ``accel/julia``.

    Parameters
    ----------
    root : Path
        The repository root.

    Returns
    -------
    set of str
        Repository-relative paths.
    """
    return {
        path.relative_to(root).as_posix()
        for path in tracked(root, PACKAGE, ".py")
        if not path.relative_to(root).as_posix().startswith(JULIA_ROOT + "/")
    }


def _module_paths(name: str, modules: set[str]) -> list[str]:
    """Return the tracked module paths a dotted-by-slash module name can mean."""
    return sorted(
        candidate
        for candidate in (f"{PACKAGE}/{name}.py", f"{PACKAGE}/{name}/__init__.py")
        if candidate in modules
    )


def transcription(text: str, modules: set[str], read: Callable[[str], set[str]]) -> Transcription:
    """Return the Python module a transcript copies, by the stated rules.

    Parameters
    ----------
    text : str
        The whole transcript.
    modules : set of str
        :func:`package_modules` of the repository.
    read : callable
        Returns the stripped lines of a repository-relative module.

    Returns
    -------
    Transcription
        The module (or ``None``), the rule, the verbatim line share ``"n/d"`` for a chosen module, and the
        candidates of an ambiguous name.
    """
    lines = [line.strip() for line in text.splitlines()]
    body = [line for line in lines if line and not line.startswith("#")]

    def chosen(module: str, rule: str) -> Transcription:
        """Return the record of a transcript whose module a rule names."""
        held = read(module)
        return {
            "transcribed_module": module,
            "transcription_rule": rule,
            "verbatim_line_share": f"{sum(1 for line in body if line in held)}/{len(body)}",
            "transcription_candidates": [],
        }

    nothing: Transcription = {
        "transcribed_module": None,
        "transcription_rule": "none",
        "verbatim_line_share": None,
        "transcription_candidates": [],
    }
    match = _JULIA_HEADER.search("\n".join(text.splitlines()[:_HEADER_LINES]))
    if match is None:
        return nothing
    name = match.group(1)
    direct = _module_paths(name, modules)
    if len(direct) == 1:
        return chosen(direct[0], "header")
    head, _, rest = name.partition("/")
    candidates = sorted(
        {
            path
            for index, character in enumerate(head)
            if character == "_"
            for path in _module_paths(
                head[:index] + "/" + head[index + 1 :] + (f"/{rest}" if rest else ""), modules
            )
        }
        | set(direct)
    )
    if len(candidates) == 1:
        return chosen(candidates[0], "header_underscore_dir")
    if candidates:
        return {
            "transcribed_module": None,
            "transcription_rule": "ambiguous",
            "verbatim_line_share": None,
            "transcription_candidates": candidates,
        }
    return nothing


def classify(authoritative: bool, loaded: bool, mirror: bool) -> str:
    """Return the class a file's three flags give, by the stated precedence.

    Parameters
    ----------
    authoritative, loaded, mirror : bool
        The file's flags.

    Returns
    -------
    str
        One of :data:`CLASSES`.
    """
    if authoritative:
        return "authoritative_entrypoint"
    if loaded:
        return "loaded_by_python"
    if mirror:
        return "declared_mirror"
    return "undeclared"


def build(root: Path, verdicts: Mapping[str, bool], julia_version: str) -> Inventory:
    """Return the whole inventory of the repository at ``root`` with the given parse verdicts.

    Parameters
    ----------
    root : Path
        The repository root.
    verdicts : mapping
        Whether each tracked ``.jl`` file parses, by repository-relative path.
    julia_version : str
        The Julia version that produced the verdicts.

    Returns
    -------
    Inventory
        The parser, the counts, the sources and the loader references.

    Raises
    ------
    ValueError
        When the tracked files and the verdicts do not cover the same paths, or the tree is refused.
    """
    julia = root / JULIA_ROOT
    paths = tracked(root, PACKAGE, ".jl")
    relatives = [path.relative_to(root).as_posix() for path in paths]
    if set(relatives) != set(verdicts):
        raise ValueError("the parse verdicts do not cover exactly the tracked Julia files")
    authoritative, mirrors = declarations(root)
    found = references(root, set(relatives))
    loaders: dict[str, set[str]] = {}
    for reference in found:
        if reference["status"] == "resolved" and reference["target"] is not None:
            loaders.setdefault(reference["target"], set()).add(reference["python"])
    modules = package_modules(root)
    stripped: dict[str, set[str]] = {}

    def read(module: str) -> set[str]:
        """Return a package module's stripped lines, read once."""
        if module not in stripped:
            text = (root / module).read_text(encoding="utf-8")
            stripped[module] = {line.strip() for line in text.splitlines()}
        return stripped[module]

    sources: list[Source] = []
    for path, relative in zip(paths, relatives, strict=True):
        inside = path.relative_to(julia).as_posix() if path.is_relative_to(julia) else None
        copied = (
            NOT_A_TRANSCRIPT
            if verdicts[relative]
            else transcription(path.read_text(encoding="utf-8"), modules, read)
        )
        flags = (
            inside is not None and inside in authoritative,
            relative in loaders,
            inside is not None and any(_glob_matches(inside, pattern) for pattern in mirrors),
        )
        sources.append(
            {
                "path": relative,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "parses": bool(verdicts[relative]),
                "authoritative": flags[0],
                "loaded": flags[1],
                "mirror": flags[2],
                "class": classify(*flags),
                "loaders": sorted(loaders.get(relative, set())),
                "transcribed_module": copied["transcribed_module"],
                "transcription_rule": copied["transcription_rule"],
                "verbatim_line_share": copied["verbatim_line_share"],
                "transcription_candidates": list(copied["transcription_candidates"]),
            }
        )
    counts: dict[str, int] = {
        "sources": len(sources),
        "parses": sum(1 for source in sources if source["parses"]),
        "does_not_parse": sum(1 for source in sources if not source["parses"]),
    }
    for name in CLASSES:
        members = [source for source in sources if source["class"] == name]
        counts[name] = len(members)
        counts[f"{name}_does_not_parse"] = sum(1 for source in members if not source["parses"])
    counts["loaded_under_a_mirror_glob"] = sum(
        1 for source in sources if source["loaded"] and source["mirror"]
    )
    for rule in (*TRANSCRIPTION_RULES, "not_a_transcript"):
        counts[f"transcription_rule_{rule}"] = sum(
            1 for source in sources if source["transcription_rule"] == rule
        )
    counts["references"] = len(found)
    for status in STATUSES:
        counts[f"references_{status}"] = sum(
            1 for reference in found if reference["status"] == status
        )
    return {
        "generator": "tools/julia_lane_inventory.py",
        "julia": {"version": julia_version, "parse_method": PARSE_METHOD},
        "counts": counts,
        "sources": sources,
        "references": found,
    }


def carried_verdicts(root: Path, record: Inventory) -> tuple[dict[str, bool], list[str]]:
    """Return the committed verdicts of files whose sha256 is unchanged, and the files that need a parse.

    Parameters
    ----------
    root : Path
        The repository root.
    record : Inventory
        The committed inventory.

    Returns
    -------
    tuple
        The carried verdicts, and the tracked files that are new or changed since the record.
    """
    committed = {source["path"]: source for source in record["sources"]}
    verdicts: dict[str, bool] = {}
    changed: list[str] = []
    for path in tracked(root, PACKAGE, ".jl"):
        relative = path.relative_to(root).as_posix()
        source = committed.get(relative)
        if source is not None and source["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest():
            verdicts[relative] = source["parses"]
        else:
            changed.append(relative)
    return verdicts, changed


def render(inventory: Inventory) -> str:
    """Return the inventory as its committed text.

    Parameters
    ----------
    inventory : Inventory
        The inventory from :func:`build`.

    Returns
    -------
    str
        Two-space indented JSON with one trailing newline.
    """
    return json.dumps(inventory, indent=2, ensure_ascii=True) + "\n"


def main(argv: Sequence[str] | None = None, parser: Parser = parse_with_julia) -> int:
    """Write, check or summarise the inventory.

    Parameters
    ----------
    argv : sequence of str, optional
        Command-line arguments; ``sys.argv[1:]`` when omitted.
    parser : callable, optional
        The Julia parser; tests replace it.

    Returns
    -------
    int
        ``0`` on success, ``1`` when ``--check`` finds the file absent, stale or a changed file it cannot parse
        here, ``2`` when the tree is refused or Julia is required and absent.
    """
    options = argparse.ArgumentParser(
        description="Classify Julia sources by declaration, loader and parse."
    )
    options.add_argument(
        "--write", action="store_true", help="parse every file and write the inventory"
    )
    options.add_argument("--check", action="store_true", help="fail when the file has drifted")
    options.add_argument("--require-julia", action="store_true", help="refuse when Julia is absent")
    options.add_argument("--summary", action="store_true", help="print the counts")
    options.add_argument("--root", type=Path, default=REPO_ROOT)
    options.add_argument("--output", type=Path, default=None)
    arguments = options.parse_args(argv)
    root: Path = arguments.root
    output: Path = arguments.output if arguments.output is not None else root / DEFAULT_OUTPUT
    try:
        relatives = [path.relative_to(root).as_posix() for path in tracked(root, PACKAGE, ".jl")]
        try:
            version, verdicts = parser(root, relatives)
            mode = f"re-parsed every file with julia {version}"
        except JuliaUnavailable as absent:
            if arguments.write or arguments.require_julia:
                print(absent, file=sys.stderr)
                return 2
            if not output.is_file():
                print(f"{output} is absent; run --write on a host with Julia.", file=sys.stderr)
                return 1
            record: Inventory = json.loads(output.read_text(encoding="utf-8"))
            verdicts, changed = carried_verdicts(root, record)
            if changed:
                print(
                    f"{len(changed)} Julia file(s) changed since the record ({changed[0]}, ...); "
                    "run --write on a host with Julia.",
                    file=sys.stderr,
                )
                return 1
            version = record["julia"]["version"]
            mode = "parse verdicts carried by sha256; julia absent"
        if arguments.check and not arguments.write and output.is_file():
            recorded = json.loads(output.read_text(encoding="utf-8"))["julia"]["version"]
            if recorded != version:
                mode += f"; the record names julia {recorded}, so only the verdicts are compared"
                version = recorded
        inventory = build(root, verdicts, version)
    except ValueError as refused:
        print(refused, file=sys.stderr)
        return 2
    text = render(inventory)
    if arguments.write:
        output.write_text(text, encoding="utf-8")
    if arguments.check:
        print(f"julia lane inventory check: {mode}")
        if not output.is_file():
            print(f"{output} is absent; run --write.", file=sys.stderr)
            return 1
        if output.read_text(encoding="utf-8") != text:
            print(f"{output} has drifted; run --write.", file=sys.stderr)
            return 1
    if arguments.summary:
        for name, value in inventory["counts"].items():
            print(f"{name}: {value}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
