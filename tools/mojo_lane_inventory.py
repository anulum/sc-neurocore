# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Mojo lane inventory

"""Classify every Mojo source of the acceleration package and map it to the builder and its Python loaders.

The inventory answers, by rule, which Mojo files hold an implementation, which still use the removed ``fn`` keyword,
which are machine-translated stubs that implement nothing, which the backend builder builds, and which library names
the package's Python modules mention without the builder attributing them to a Mojo build. It is written to
``docs/_generated/mojo_lane_inventory.json`` and checked for drift.

Classes of a Mojo source (comment lines and triple-quoted blocks are not code):

``comment_only``
    The file has no code line.
``translated_stub``
    The file declares at least one single-line function, and every code line that is not a header (a decorator, a
    ``def`` or ``fn`` line, an import) is a string assignment (``var name = '…'``), ``return 0`` or ``pass``. A
    string may hold escaped quotes, and an assignment whose value is a parenthesised run of string lines
    (``var name = (`` … ``)``) counts as one string assignment. Such a file holds the text of a Python original and
    returns zero; it implements nothing. A file with a multi-line signature is never classed as a stub, and a
    parenthesised value that holds anything but strings is code, so real code is never called a stub.
``implementation_fn``
    Any other file that declares a function with ``fn``, which Mojo 1.0.0 and later refuse ("'fn' has been removed").
``implementation_def``
    Any other file.

The builder's view comes from ``tools/build_accel_backends.py`` itself (its target discovery and its per-loader
output paths); this module does not repeat that logic. Independently, every ``lib<name>.so`` string literal in a
package module is listed, and marked when the builder attributes that module's library to a Mojo build. The package's
own declarations (``AUTHORITATIVE_MOJO_ENTRYPOINTS`` and ``NON_AUTHORITATIVE_MOJO_MIRROR_GLOBS`` in
``accel/mojo/__init__.py``) are read and reported beside each source.

The inventory also classes every neuron model module (``neurons/models``) by how it chooses an accelerated backend
for ``backend="auto"``, by AST. A model's own module is read together with every ``accel`` dispatch module it imports
(one hop), since many models delegate to one. The accelerated backends are the package's own ``ACCELERATORS``
(``accel/backend_order.py``). The code that decides ``auto`` is each branch whose condition compares with ``"auto"``
(its condition and body) and each boolean expression that does (``prefer = backend == "rust" or (backend == "auto"
and ...)``), with the module-level functions they call and the module-level values they read. Classes, in this order
of precedence:

``selector``
    The model or a dispatch module it imports calls ``select_backend_order``.
``fixed_order``
    No selector, and the ``auto`` code names two or more accelerated backends as string constants or uses the static
    order (``ACCELERATORS``, ``FASTEST_FIRST_BACKENDS`` or ``with_floor``); without any ``auto`` branch, the module
    uses the static order. An order fixed in code.
``single_backend``
    No selector and no fixed order, and the ``auto`` code names exactly one accelerated backend (a Rust-only
    ``auto``, for one).
``direct_backend``
    None of the above, but the model imports an accelerated backend's own package (``sc_neurocore.accel.<backend>``)
    or opens a shared library itself (``ctypes.CDLL``), without any selector call.
``no_accelerated_backend``
    None of the above, and the module does not import the Rust engine either. A module that only imports the Rust
    engine (``sc_neurocore_engine``) is ``single_backend``.

Every record also carries the features behind its class, so a model that both chooses Rust alone and opens
libraries directly shows both facts.

Only files Git tracks are read, so the inventory is a property of the repository's content, not of a working
directory: an untracked file or an ignored directory (an installed Mojo environment, for one) changes nothing. The
builder's targets are kept only when their source is tracked. The root must be the top of a Git work tree, and a
tracked file missing from it is refused.

Usage::

    python tools/mojo_lane_inventory.py --write
    python tools/mojo_lane_inventory.py --check
    python tools/mojo_lane_inventory.py --summary
"""

from __future__ import annotations

import argparse
import ast
import fnmatch
import json
import re
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import TypedDict

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from tools.benchmark_evidence_gate import tracked_files
from tools.build_accel_backends import _loader_output_paths, discover_targets

DEFAULT_OUTPUT = "docs/_generated/mojo_lane_inventory.json"
PACKAGE = "src/sc_neurocore"
ACCEL = "src/sc_neurocore/accel"
MOJO_ROOT = "src/sc_neurocore/accel/mojo"
CLASSES = ("comment_only", "translated_stub", "implementation_fn", "implementation_def")
MODELS = "src/sc_neurocore/neurons/models"
BACKEND_ORDER = "src/sc_neurocore/accel/backend_order.py"
MODEL_CLASSES = (
    "selector",
    "fixed_order",
    "direct_backend",
    "single_backend",
    "no_accelerated_backend",
)
#: Accel modules that are infrastructure, not one model's dispatch module.
_NOT_DISPATCH = frozenset({"__init__", "backend", "backend_order", "backend_selection"})
_STATIC_ORDER_NAMES = frozenset({"ACCELERATORS", "FASTEST_FIRST_BACKENDS", "with_floor"})

#: One Mojo source: its path, class, declaration, whether the builder builds it, and the modules that load it.
Source = TypedDict(
    "Source",
    {
        "path": str,
        "class": str,
        "declared": str | None,
        "built_by_builder": bool,
        "loaders": list[str],
    },
)


class LibraryLiteral(TypedDict):
    """One ``lib<name>.so`` literal, its module, and the Mojo source the builder attributes it to, if any."""

    python: str
    library: str
    builder_mojo_source: str | None


class Features(TypedDict):
    """What one module's AST shows about backend choice."""

    calls_selector: bool
    backend_names: list[str]
    static_order: bool
    backend_packages: list[str]
    opens_library: bool
    rust_engine: bool
    dispatch_modules: list[str]
    auto_branch: bool
    auto_backend_names: list[str]
    auto_static_order: bool


#: One neuron model module: its path, class, the dispatch modules read with it, and the combined features.
Model = TypedDict(
    "Model",
    {"path": str, "class": str, "dispatch_modules": list[str], "features": Features},
)


class Inventory(TypedDict):
    """The whole inventory as it is committed."""

    generator: str
    counts: dict[str, int]
    sources: list[Source]
    library_literals: list[LibraryLiteral]
    models: list[Model]


_HEADER = re.compile(r"^\s*(@[\w.]+|fn\s|def\s|from\s|import\s)")
_FUNCTION = re.compile(r"^\s*(fn|def)\s+\w+\s*(\[[^\]]*\])?\s*\(.*\)\s*(->\s*[^:]+)?:\s*(#.*)?$")
_FN = re.compile(r"^\s*fn\s+\w+")
_STRING = r"""('(?:[^'\\]|\\.)*'|"(?:[^"\\]|\\.)*")"""
_STUB_LINE = re.compile(rf"^\s*(var\s+\w+\s*=\s*{_STRING}|return\s+0|pass)\s*(#.*)?$")
_OPEN_ASSIGNMENT = re.compile(r"^\s*var\s+\w+\s*=\s*\(\s*(#.*)?$")
_STRING_LINE = re.compile(rf"^\s*{_STRING}\s*(#.*)?$")
_CLOSE = re.compile(r"^\s*\)\s*(#.*)?$")
_LIBRARY = re.compile(r"^lib[\w.-]+\.so$")


def tracked(root: Path, directory: str, suffix: str) -> list[Path]:
    """Return the files Git tracks below a directory of the work tree, with one suffix, sorted.

    Parameters
    ----------
    root : Path
        The repository root; it must be the top of a Git work tree.
    directory : str
        The repository-relative directory.
    suffix : str
        The file suffix wanted, such as ``".py"``.

    Returns
    -------
    list of Path
        The absolute paths, sorted.

    Raises
    ------
    ValueError
        When ``root`` is not the top of a Git work tree, or a tracked file is missing from it.
    """
    listed = tracked_files(root, directory)
    if listed is None:
        raise ValueError(
            f"{root} is not the top of a Git work tree; only tracked files are inventoried."
        )
    paths = sorted(root / name for name in listed if name.endswith(suffix))
    missing = [path.relative_to(root).as_posix() for path in paths if not path.is_file()]
    if missing:
        raise ValueError(f"tracked files are missing from the work tree: {', '.join(missing)}")
    return paths


def code_lines(text: str) -> list[str]:
    """Return the code lines of a Mojo source: no blank, comment-only or triple-quoted lines.

    Parameters
    ----------
    text : str
        The whole Mojo source.

    Returns
    -------
    list of str
        The code lines, in order.
    """
    lines: list[str] = []
    quoted = False
    for line in text.splitlines():
        stripped = line.strip()
        if quoted:
            if stripped.count('"""') % 2 == 1:
                quoted = False
            continue
        if stripped.startswith('"""'):
            if stripped.count('"""') % 2 == 1:
                quoted = True
            continue
        if stripped and not stripped.startswith("#"):
            lines.append(line)
    return lines


def _joined(lines: list[str]) -> list[str]:
    """Return code lines with each parenthesised run of string lines folded into one string assignment.

    Parameters
    ----------
    lines : list of str
        Code lines, as :func:`code_lines` returns them.

    Returns
    -------
    list of str
        The same lines, except that ``var name = (``, one or more string lines and ``)`` become one line
        ``var name = ''``. A run that holds anything but strings, or never closes, is left as it is.
    """
    joined: list[str] = []
    index = 0
    while index < len(lines):
        line = lines[index]
        if _OPEN_ASSIGNMENT.match(line):
            end = index + 1
            while end < len(lines) and _STRING_LINE.match(lines[end]):
                end += 1
            if end > index + 1 and end < len(lines) and _CLOSE.match(lines[end]):
                joined.append(line.split("=", 1)[0] + "= ''")
                index = end + 1
                continue
        joined.append(line)
        index += 1
    return joined


def classify(text: str) -> str:
    """Return the class of one Mojo source by the module's stated rules.

    Parameters
    ----------
    text : str
        The whole Mojo source.

    Returns
    -------
    str
        One of ``comment_only``, ``translated_stub``, ``implementation_fn`` or ``implementation_def``.
    """
    lines = code_lines(text)
    if not lines:
        return "comment_only"
    functions = [line for line in lines if _FUNCTION.match(line)]
    body = [line for line in _joined(lines) if not _HEADER.match(line)]
    if functions and all(_STUB_LINE.match(line) for line in body):
        return "translated_stub"
    if any(_FN.match(line) for line in lines):
        return "implementation_fn"
    return "implementation_def"


def string_tuples(path: Path) -> dict[str, tuple[str, ...]]:
    """Return the module-level tuples of string constants a Python module assigns, by name.

    Parameters
    ----------
    path : Path
        The module.

    Returns
    -------
    dict
        Each assigned name and the string constants of its tuple, in order.
    """
    found: dict[str, tuple[str, ...]] = {}
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)) and isinstance(node.value, ast.Tuple):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name):
                    found[target.id] = tuple(
                        item.value
                        for item in node.value.elts
                        if isinstance(item, ast.Constant) and isinstance(item.value, str)
                    )
    return found


def declarations(root: Path) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return the Mojo package's declared authoritative entry points and mirror globs.

    Parameters
    ----------
    root : Path
        The repository root.

    Returns
    -------
    tuple of (tuple of str, tuple of str)
        The authoritative entry points and the non-authoritative mirror globs, relative to the Mojo package; empty
        when the package declares none.
    """
    init = root / MOJO_ROOT / "__init__.py"
    found = string_tuples(init) if init.is_file() else {}
    return (
        found.get("AUTHORITATIVE_MOJO_ENTRYPOINTS", ()),
        found.get("NON_AUTHORITATIVE_MOJO_MIRROR_GLOBS", ()),
    )


def library_literals(root: Path) -> list[tuple[str, str]]:
    """Return every ``lib<name>.so`` string literal in the package's tracked Python modules.

    Parameters
    ----------
    root : Path
        The repository root, the top of a Git work tree.

    Returns
    -------
    list of (str, str)
        The module path relative to ``root`` and the library name, one pair per module and name, sorted.
    """
    pairs: set[tuple[str, str]] = set()
    for module in tracked(root, PACKAGE, ".py"):
        tree = ast.parse(module.read_text(encoding="utf-8"), filename=str(module))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Constant)
                and isinstance(node.value, str)
                and _LIBRARY.match(node.value)
            ):
                pairs.add((module.relative_to(root).as_posix(), node.value))
    return sorted(pairs)


def _absolute(module: str | None, level: int, package: str) -> str:
    """Return an import's absolute module name, resolving a relative import against the importing package."""
    if level == 0:
        return module or ""
    parts = package.split(".")
    base = parts[: len(parts) - (level - 1)]
    return ".".join([*base, module] if module else base)


def _tests_auto(test: ast.expr) -> bool:
    """Say whether a condition compares something with the string ``"auto"``."""
    return any(
        isinstance(node, ast.Compare)
        and any(
            isinstance(side, ast.Constant) and side.value == "auto"
            for side in (node.left, *node.comparators)
        )
        for node in ast.walk(test)
    )


def _auto_nodes(tree: ast.Module) -> list[ast.AST]:
    """Return the code that decides an ``"auto"`` backend: auto branches and boolean expressions, and what they reach.

    One step reaches the bodies of module-level functions the branches call by name and the values of module-level
    names they read, so an order kept in a constant or a helper is seen.
    """
    functions = {item.name: item for item in tree.body if isinstance(item, ast.FunctionDef)}
    values: dict[str, ast.expr] = {}
    for statement in tree.body:
        if isinstance(statement, ast.Assign):
            for target in statement.targets:
                if isinstance(target, ast.Name):
                    values[target.id] = statement.value
        elif (
            isinstance(statement, ast.AnnAssign)
            and isinstance(statement.target, ast.Name)
            and statement.value is not None
        ):
            values[statement.target.id] = statement.value
    branches: list[ast.AST] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.IfExp) and _tests_auto(node.test):
            branches.extend((node.test, node.body))
        elif isinstance(node, ast.If) and _tests_auto(node.test):
            branches.extend((node.test, *node.body))
        elif isinstance(node, ast.BoolOp) and _tests_auto(node):
            branches.append(node)
    reached: list[ast.AST] = list(branches)
    for branch in branches:
        for node in ast.walk(branch):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id in functions
            ):
                reached.append(functions[node.func.id])
    for item in list(reached):
        for node in ast.walk(item):
            if isinstance(node, ast.Name) and node.id in values:
                reached.append(values[node.id])
    return reached


def _choice(nodes: list[ast.AST], accelerators: tuple[str, ...]) -> tuple[bool, set[str], bool]:
    """Return whether some nodes call the selector, which accelerated backends they name, and whether they use the static order."""
    selector = static = False
    names: set[str] = set()
    for item in nodes:
        for node in ast.walk(item):
            if isinstance(node, ast.Call):
                called = node.func
                name = (
                    called.id
                    if isinstance(called, ast.Name)
                    else called.attr
                    if isinstance(called, ast.Attribute)
                    else ""
                )
                selector = selector or name == "select_backend_order"
            elif (
                isinstance(node, ast.Constant)
                and isinstance(node.value, str)
                and node.value in accelerators
            ):
                names.add(node.value)
            elif isinstance(node, ast.Name) and node.id in _STATIC_ORDER_NAMES:
                static = True
    return selector, names, static


def module_features(path: Path, root: Path, accelerators: tuple[str, ...]) -> Features:
    """Return what one module's AST shows about how it chooses an accelerated backend.

    Parameters
    ----------
    path : Path
        A tracked Python module below ``src``.
    root : Path
        The repository root.
    accelerators : tuple of str
        The package's accelerated backend names.

    Returns
    -------
    Features
        Selector calls, named backends, use of the static order, backend packages imported, shared libraries opened
        (a ``CDLL`` call or the guarded ``load_mojo_library``),
        the Rust engine imported, and the ``accel`` dispatch modules imported (their names only).
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    package = ".".join(path.relative_to(root / "src").with_suffix("").parts[:-1])
    selector, names, static = _choice([tree], accelerators)
    auto = _auto_nodes(tree)
    _, auto_names, auto_static = _choice(auto, accelerators)
    opens = engine = False
    packages: set[str] = set()
    dispatch: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            opens = opens or node.func.id == "load_mojo_library"
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            opens = opens or node.func.attr in ("CDLL", "load_mojo_library")
            engine = engine or (
                node.func.attr == "import_module"
                and bool(node.args)
                and isinstance(node.args[0], ast.Constant)
                and node.args[0].value == "sc_neurocore_engine"
            )
        elif isinstance(node, ast.Import):
            modules = [alias.name for alias in node.names]
            engine = engine or any(m.split(".")[0] == "sc_neurocore_engine" for m in modules)
            for module in modules:
                _accel_import(module, accelerators, packages, dispatch)
        elif isinstance(node, ast.ImportFrom):
            base = _absolute(node.module, node.level, package)
            engine = engine or base.split(".")[0] == "sc_neurocore_engine"
            for module in (base, *(f"{base}.{alias.name}" for alias in node.names)):
                _accel_import(module, accelerators, packages, dispatch)
    return {
        "calls_selector": selector,
        "backend_names": sorted(names),
        "static_order": static,
        "backend_packages": sorted(packages),
        "opens_library": opens,
        "rust_engine": engine,
        "dispatch_modules": sorted(dispatch),
        "auto_branch": bool(auto),
        "auto_backend_names": sorted(auto_names),
        "auto_static_order": auto_static,
    }


def _accel_import(
    module: str, accelerators: tuple[str, ...], packages: set[str], dispatch: set[str]
) -> None:
    """Record an imported ``sc_neurocore.accel`` module as a backend package or a dispatch module."""
    parts = module.split(".")
    if parts[:2] == ["sc_neurocore", "accel"] and len(parts) >= 3:
        if parts[2] in accelerators:
            packages.add(".".join(parts[:3]))
        elif parts[2] not in _NOT_DISPATCH:
            dispatch.add(parts[2])


def classify_model(features: Features) -> str:
    """Return a model module's class by the module's stated rules and their precedence.

    Parameters
    ----------
    features : Features
        The model's own features combined with those of the dispatch modules it imports.

    Returns
    -------
    str
        One of :data:`MODEL_CLASSES`.
    """
    if features["calls_selector"]:
        return "selector"
    if features["auto_branch"]:
        if len(features["auto_backend_names"]) >= 2 or features["auto_static_order"]:
            return "fixed_order"
        if features["auto_backend_names"]:
            return "single_backend"
    elif features["static_order"]:
        return "fixed_order"
    if features["backend_packages"] or features["opens_library"]:
        return "direct_backend"
    if features["rust_engine"]:
        return "single_backend"
    return "no_accelerated_backend"


def models(root: Path) -> list[Model]:
    """Return every tracked neuron model module with its class, reading its ``accel`` dispatch modules one hop.

    Parameters
    ----------
    root : Path
        The repository root, the top of a Git work tree.

    Returns
    -------
    list of Model
        One record per model module, sorted by path.
    """
    accel_modules = {
        path.stem: path for path in tracked(root, ACCEL, ".py") if path.parent == root / ACCEL
    }
    order = root / BACKEND_ORDER
    accelerators = (
        string_tuples(order).get("ACCELERATORS", ()) if order in accel_modules.values() else ()
    )
    records: list[Model] = []
    for path in tracked(root, MODELS, ".py"):
        own = module_features(path, root, accelerators)
        hops = [name for name in own["dispatch_modules"] if name in accel_modules]
        combined = [
            own,
            *(module_features(accel_modules[name], root, accelerators) for name in hops),
        ]
        merged: Features = {
            "calls_selector": any(item["calls_selector"] for item in combined),
            "backend_names": sorted({name for item in combined for name in item["backend_names"]}),
            "static_order": any(item["static_order"] for item in combined),
            "backend_packages": sorted(
                {name for item in combined for name in item["backend_packages"]}
            ),
            "opens_library": any(item["opens_library"] for item in combined),
            "rust_engine": any(item["rust_engine"] for item in combined),
            "dispatch_modules": own["dispatch_modules"],
            "auto_branch": own["auto_branch"],
            "auto_backend_names": own["auto_backend_names"],
            "auto_static_order": own["auto_static_order"],
        }
        records.append(
            {
                "path": path.relative_to(root).as_posix(),
                "class": classify_model(merged),
                "dispatch_modules": [
                    accel_modules[name].relative_to(root).as_posix() for name in hops
                ],
                "features": merged,
            }
        )
    return records


def build(root: Path) -> Inventory:
    """Return the whole inventory of the repository at ``root``.

    Parameters
    ----------
    root : Path
        The repository root.

    Returns
    -------
    Inventory
        The counts, the sources, the library literals and the model modules.
    """
    accel = root / ACCEL
    mojo = root / MOJO_ROOT
    mojo_sources = tracked(root, MOJO_ROOT, ".mojo")
    tracked_sources = set(mojo_sources)
    targets = {
        target.output: target.source
        for target in discover_targets("mojo", accel_root=accel)
        if target.source in tracked_sources
    }
    attributed: dict[str, list[str]] = {}
    literals: list[LibraryLiteral] = []
    for module, library in library_literals(root):
        outputs = _loader_output_paths("mojo", [root / module], accel)
        built = sorted(output for output in outputs if output.name == library and output in targets)
        source = targets[built[0]].relative_to(root).as_posix() if built else None
        if source is not None:
            attributed.setdefault(source, []).append(module)
        literals.append({"python": module, "library": library, "builder_mojo_source": source})
    authoritative, mirrors = declarations(root)
    sources: list[Source] = []
    for path in mojo_sources:
        relative = path.relative_to(mojo).as_posix()
        declared = (
            "authoritative"
            if relative in authoritative
            else "mirror"
            if any(fnmatch.fnmatch(relative, pattern) for pattern in mirrors)
            else None
        )
        full = path.relative_to(root).as_posix()
        sources.append(
            {
                "path": full,
                "class": classify(path.read_text(encoding="utf-8")),
                "declared": declared,
                "built_by_builder": path in targets.values(),
                "loaders": sorted(set(attributed.get(full, []))),
            }
        )
    counts: dict[str, int] = {"sources": len(sources)}
    for name in CLASSES:
        counts[name] = sum(1 for source in sources if source["class"] == name)
    counts["built_by_builder"] = sum(1 for source in sources if source["built_by_builder"])
    counts["implementation_fn_built_by_builder"] = sum(
        1
        for source in sources
        if source["class"] == "implementation_fn" and source["built_by_builder"]
    )
    counts["translated_stub_declared_mirror"] = sum(
        1
        for source in sources
        if source["class"] == "translated_stub" and source["declared"] == "mirror"
    )
    counts["library_literals"] = len(literals)
    counts["library_literals_attributed_to_mojo"] = sum(
        1 for item in literals if item["builder_mojo_source"]
    )
    model_records = models(root)
    counts["models"] = len(model_records)
    for name in MODEL_CLASSES:
        counts[f"models_{name}"] = sum(1 for record in model_records if record["class"] == name)
    return {
        "generator": "tools/mojo_lane_inventory.py",
        "counts": counts,
        "sources": sources,
        "library_literals": literals,
        "models": model_records,
    }


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


def main(argv: Sequence[str] | None = None) -> int:
    """Write, check or summarise the inventory.

    Parameters
    ----------
    argv : sequence of str, optional
        Command-line arguments; ``sys.argv[1:]`` when omitted.

    Returns
    -------
    int
        ``0`` on success, ``1`` when ``--check`` finds the file absent or stale, ``2`` when the root is not the top
        of a Git work tree or a tracked file is missing from it.
    """
    parser = argparse.ArgumentParser(
        description="Classify Mojo sources and map them to the builder and loaders."
    )
    parser.add_argument("--write", action="store_true", help="write the inventory")
    parser.add_argument("--check", action="store_true", help="fail when the file has drifted")
    parser.add_argument("--summary", action="store_true", help="print the counts")
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--output", type=Path, default=None)
    arguments = parser.parse_args(argv)
    root: Path = arguments.root
    output: Path = arguments.output if arguments.output is not None else root / DEFAULT_OUTPUT
    try:
        inventory = build(root)
    except ValueError as refused:
        print(refused, file=sys.stderr)
        return 2
    text = render(inventory)
    if arguments.write:
        output.write_text(text, encoding="utf-8")
    if arguments.check:
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
