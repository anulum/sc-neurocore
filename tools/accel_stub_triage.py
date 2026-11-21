# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — accelerated-lane stub triage

"""Class the Python original of every accelerated-lane stub as a compute kernel or as application code.

The owner's decision for the stubs of the Mojo and Julia lanes: a stub that copies a compute kernel is replaced by a
real implementation held by a parity test against today's Python module, and a stub that copies application code is
retired through a deletion list the owner approves first. This record is that triage. It reads the two lane
inventories, which name the module each stub copies, and classes each named module from its own source by rule; it
is written to ``docs/_generated/accel_stub_triage.json`` and checked for drift.

Markers, read from the original's syntax tree (nothing is imported or executed):

``app``
    It imports a web, interface, command-line, network, process, storage or plotting library (:data:`APP_IMPORTS`),
    or imports ``argparse`` and has an ``if __name__ == "__main__"`` block.
``numeric``
    Inside a function it calls into ``numpy``, ``scipy``, ``torch``, ``jax`` or ``numba``, through the imported
    module (``np.exp(x)``) or a name imported from it (``from numpy import exp``; ``exp(x)``); or it calls into ``math``, ``cmath``, ``random`` or ``statistics`` and has at least
    :data:`PURE_OPERATORS` multiplicative operators (``* / ** @``) inside functions; or it has at least
    :data:`OPERATORS` of them.

Classes: ``compute`` has the numeric marker and no app marker; ``application`` has the app marker and no numeric
marker; a module with both or neither is ``unclear``. The thresholds only move a module between compute and unclear,
never into application, so a threshold can delay a port but never delete a kernel. An unclear module is decided only
by a recorded reading (:data:`READINGS`, with its reason and reader); a reading of a module the rule decides is
refused. A stub whose lane names several equal candidates is resolved only by a recorded reading too
(:data:`AMBIGUOUS_STUBS`), to one of its candidates.

The record lists every original with its class, markers and stubs (lane, path, rule, verbatim line share, loaders,
the tracked tests and benchmarks that name the stub, and its declaration); the deletion list (every stub of an
application original); and the compute order: originals whose stubs have a loader or are named by a benchmark first,
then by the marker that made them compute (array-library calls, then pure numeric calls, then operators alone), then
those with stubs in more lanes, then by path. For comparison the record also gives the classes that array-library
calls alone would give. A port starts from today's Python module, never the transcript.
Julia files that parse but that nothing loads or declares are read with the Julia tool's own header rule and listed
in the lane ``julia_parseable_unloaded``. A stub of either lane whose module no rule names is listed apart in
``unmapped``, never dropped.

Usage::

    python tools/accel_stub_triage.py --write
    python tools/accel_stub_triage.py --check
    python tools/accel_stub_triage.py --summary
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, TypedDict

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from tools import julia_lane_inventory
from tools.mojo_lane_inventory import tracked

DEFAULT_OUTPUT = "docs/_generated/accel_stub_triage.json"
MOJO_RECORD = "docs/_generated/mojo_lane_inventory.json"
JULIA_RECORD = "docs/_generated/julia_lane_inventory.json"
CLASSES = ("compute", "application", "unclear")
LANES = ("mojo", "julia", "julia_parseable_unloaded")

#: Imports that mark an original as application code.
APP_IMPORTS = frozenset(
    {
        "fastapi",
        "starlette",
        "flask",
        "uvicorn",
        "tkinter",
        "PyQt5",
        "PyQt6",
        "PySide2",
        "PySide6",
        "click",
        "typer",
        "http",
        "urllib",
        "requests",
        "socket",
        "subprocess",
        "sqlite3",
        "matplotlib",
        "plotly",
    }
)
#: Array libraries whose calls inside a function mark numeric work.
NUMERIC_LIBRARIES = frozenset({"numpy", "scipy", "torch", "jax", "numba"})
#: Pure-Python numeric modules whose calls, with enough arithmetic, mark numeric work.
PURE_NUMERIC_MODULES = frozenset({"math", "cmath", "random", "statistics"})
#: Multiplicative operators inside functions that, with a pure numeric call, mark numeric work.
PURE_OPERATORS = 3
#: Multiplicative operators inside functions that alone mark numeric work.
OPERATORS = 10

#: A recorded reading: the class a seat gave a module, why, and which seat read it.
Reading = TypedDict("Reading", {"class": str, "reason": str, "reader": str})


class Resolution(TypedDict):
    """A recorded resolution of an ambiguous stub: the candidate its port starts from, why, and which seat read it."""

    module: str
    reason: str
    reader: str


#: Readings of modules the rule leaves unclear, by repository path.
READINGS: dict[str, Reading] = {
    "src/sc_neurocore/adapters/holonomic/l10_fire.py": {
        "class": "compute",
        "reason": "JAX array numerics through the package's _jax_compat shim (jnp, rng), a relative import the rule does not see",
        "reader": "aadb",
    },
    "src/sc_neurocore/adapters/holonomic/l11_noos.py": {
        "class": "compute",
        "reason": "JAX array numerics through the package's _jax_compat shim (jnp, rng), a relative import the rule does not see",
        "reader": "aadb",
    },
    "src/sc_neurocore/adapters/holonomic/l12_gaian.py": {
        "class": "compute",
        "reason": "JAX array numerics through the package's _jax_compat shim (jnp, rng), a relative import the rule does not see",
        "reader": "aadb",
    },
    "src/sc_neurocore/adapters/holonomic/l14_trans.py": {
        "class": "compute",
        "reason": "JAX array numerics through the package's _jax_compat shim (jnp, rng), a relative import the rule does not see",
        "reader": "aadb",
    },
    "src/sc_neurocore/adapters/holonomic/l15_cons.py": {
        "class": "compute",
        "reason": "JAX array numerics through the package's _jax_compat shim (jnp, rng), a relative import the rule does not see",
        "reader": "aadb",
    },
    "src/sc_neurocore/adapters/holonomic/l16_meta.py": {
        "class": "compute",
        "reason": "JAX array numerics through the package's _jax_compat shim (jnp, rng), a relative import the rule does not see",
        "reader": "aadb",
    },
    "src/sc_neurocore/adapters/holonomic/l1_quantum.py": {
        "class": "compute",
        "reason": "JAX array numerics through the package's _jax_compat shim (jnp, rng), a relative import the rule does not see",
        "reader": "aadb",
    },
    "src/sc_neurocore/adapters/holonomic/l3_gen.py": {
        "class": "compute",
        "reason": "JAX array numerics through the package's _jax_compat shim (jnp, rng), a relative import the rule does not see",
        "reader": "aadb",
    },
    "src/sc_neurocore/adapters/holonomic/l5_org.py": {
        "class": "compute",
        "reason": "JAX array numerics through the package's _jax_compat shim (jnp, rng), a relative import the rule does not see",
        "reader": "aadb",
    },
    "src/sc_neurocore/adapters/holonomic/l8_cosm.py": {
        "class": "compute",
        "reason": "JAX array numerics through the package's _jax_compat shim (jnp, rng), a relative import the rule does not see",
        "reader": "aadb",
    },
    "src/sc_neurocore/edge/bitstream.py": {
        "class": "compute",
        "reason": "packed 32-bit bitstream arithmetic (popcount, SC and/or/xor/mux) in bitwise operators the rule does not count",
        "reader": "aadb",
    },
    "src/sc_neurocore/edge/lfsr.py": {
        "class": "compute",
        "reason": "an LFSR-16 encoder, bit-compatible with the engine, in bitwise operators",
        "reader": "aadb",
    },
    "src/sc_neurocore/edge/neuron.py": {
        "class": "compute",
        "reason": "LIF and Izhikevich neuron updates in the SC domain",
        "reader": "aadb",
    },
    "src/sc_neurocore/edge/sc_network.py": {
        "class": "compute",
        "reason": "feed-forward inference over packed binary connection masks",
        "reader": "aadb",
    },
    "src/sc_neurocore/neurons/dendritic.py": {
        "class": "compute",
        "reason": "a stochastic neuron update comparing XOR-combined dendritic inputs",
        "reader": "aadb",
    },
    "src/sc_neurocore/neurons/fixed_point_lif.py": {
        "class": "compute",
        "reason": "bit-true fixed-point LIF neuron, LFSR and bitstream models of the Verilog core",
        "reader": "aadb",
    },
    "src/sc_neurocore/neurons/models/spinnaker2.py": {
        "class": "compute",
        "reason": "a fixed-point LIF update (multiply and shift) of the SpiNNaker2 neuron",
        "reader": "aadb",
    },
    "src/sc_neurocore/robotics/cpg.py": {
        "class": "compute",
        "reason": "a central pattern generator built from coupled neuron updates",
        "reader": "aadb",
    },
    "src/sc_neurocore/spike_codec/entropy.py": {
        "class": "compute",
        "reason": "canonical Huffman encoding and decoding of integer streams",
        "reader": "aadb",
    },
    "src/sc_neurocore/synapses/r_stdp.py": {
        "class": "compute",
        "reason": "a reward-modulated STDP weight update",
        "reader": "aadb",
    },
}

_TWIN = (
    "The two copies hold the same code apart from the order of their imports, and the package's adapter discovery "
    "names the adapters.holonomic copy, so the port starts from it."
)
#: Resolutions of stubs whose lane names several equal candidates, by stub path.
AMBIGUOUS_STUBS: dict[str, Resolution] = {
    "src/sc_neurocore/accel/mojo/kernels/dna_storage.mojo": {
        "module": "src/sc_neurocore/adapters/holonomic/dna_storage.py",
        "reason": _TWIN,
        "reader": "aadb",
    },
    "src/sc_neurocore/accel/mojo/kernels/grn.mojo": {
        "module": "src/sc_neurocore/adapters/holonomic/grn.py",
        "reason": _TWIN,
        "reader": "aadb",
    },
}


class Markers(TypedDict):
    """What the syntax tree of one original shows.

    ``basis`` names the first numeric marker that holds: ``a`` array-library calls, ``b`` pure numeric calls with
    enough operators, ``c`` operators alone, or ``None``.
    """

    app: list[str]
    numeric: list[str]
    operators: int
    basis: str | None


class Variant(TypedDict):
    """The classes the rule would give with the numeric marker ``a`` alone, for comparison."""

    compute: int
    application: int
    unclear: int
    application_modules: list[str]


class Stub(TypedDict):
    """One stub that copies an original, in one lane."""

    lane: str
    path: str
    rule: str
    verbatim_line_share: str | None
    loaders: list[str]
    named_by: list[str]
    declared: str | None


#: One Python module that stubs copy: its class, how it was decided, its markers and its stubs.
Original = TypedDict(
    "Original",
    {
        "module": str,
        "class": str,
        "decided_by": str,
        "reason": str | None,
        "markers": Markers,
        "stubs": list[Stub],
        "used": bool,
    },
)


class Deletion(TypedDict):
    """One stub of an application original, with what its removal must also update."""

    module: str
    lane: str
    path: str
    loaders: list[str]
    named_by: list[str]
    declared: str | None


class Inventory(TypedDict):
    """The whole triage as it is committed."""

    generator: str
    counts: dict[str, int]
    variant_a_only: Variant
    originals: list[Original]
    deletion_list: list[Deletion]
    compute_order: list[str]
    unmapped: list[str]


def _main_guard(test: ast.expr) -> bool:
    """Return whether a condition is ``__name__ == "__main__"``."""
    return (
        isinstance(test, ast.Compare)
        and isinstance(test.left, ast.Name)
        and test.left.id == "__name__"
        and any(
            isinstance(item, ast.Constant) and item.value == "__main__" for item in test.comparators
        )
    )


def markers(text: str) -> Markers:
    """Return the app and numeric markers of one Python module's source.

    Parameters
    ----------
    text : str
        The whole module source.

    Returns
    -------
    Markers
        The app markers, the numeric markers, and the multiplicative operators inside functions.
    """
    tree = ast.parse(text)
    imported: set[str] = set()
    aliases: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split(".")[0]
                imported.add(top)
                aliases[alias.asname or top] = top
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            top = node.module.split(".")[0]
            imported.add(top)
            for alias in node.names:
                aliases[alias.asname or alias.name] = top
    app = sorted(imported & APP_IMPORTS)
    if "argparse" in imported and any(
        isinstance(node, ast.If) and _main_guard(node.test) for node in tree.body
    ):
        app.append("argparse with a main block")
    inside: dict[int, ast.AST] = {}
    for function in ast.walk(tree):
        if isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for node in ast.walk(function):
                inside[id(node)] = node
    libraries: set[str] = set()
    pure = False
    operators = 0
    for node in inside.values():
        if isinstance(node, ast.Call) and isinstance(node.func, (ast.Attribute, ast.Name)):
            base: ast.expr = node.func
            while isinstance(base, ast.Attribute):
                base = base.value
            if isinstance(base, ast.Name):
                library = aliases.get(base.id)
                if library in NUMERIC_LIBRARIES:
                    libraries.add(library)
                elif library in PURE_NUMERIC_MODULES:
                    pure = True
        elif isinstance(node, ast.BinOp) and isinstance(
            node.op, (ast.Mult, ast.Div, ast.Pow, ast.MatMult)
        ):
            operators += 1
    numeric = [f"{library} calls" for library in sorted(libraries)]
    if pure and operators >= PURE_OPERATORS:
        numeric.append("pure numeric calls with multiplicative operators")
    if operators >= OPERATORS:
        numeric.append("multiplicative operators")
    basis = (
        "a"
        if libraries
        else "b"
        if pure and operators >= PURE_OPERATORS
        else "c"
        if operators >= OPERATORS
        else None
    )
    return {"app": app, "numeric": numeric, "operators": operators, "basis": basis}


def classify_a_only(found: Markers) -> str:
    """Return the class the rule would give if only array-library calls marked numeric work.

    Parameters
    ----------
    found : Markers
        The markers of one original.

    Returns
    -------
    str
        ``compute``, ``application`` or ``unclear``.
    """
    library = found["basis"] == "a"
    if library and not found["app"]:
        return "compute"
    if found["app"] and not library:
        return "application"
    return "unclear"


def classify(found: Markers) -> str:
    """Return the class the markers give by the stated rule.

    Parameters
    ----------
    found : Markers
        The markers of one original.

    Returns
    -------
    str
        ``compute``, ``application`` or ``unclear``.
    """
    if found["numeric"] and not found["app"]:
        return "compute"
    if found["app"] and not found["numeric"]:
        return "application"
    return "unclear"


_PATH_TOKEN = re.compile(r"[\w./-]+\.(?:mojo|jl)\b")


def _path_tokens(texts: dict[str, str]) -> dict[str, set[str]]:
    """Return, for each file, the ``.mojo`` and ``.jl`` path tokens its text holds, read in one pass."""
    return {name: set(_PATH_TOKEN.findall(text)) for name, text in texts.items()}


def _mentions(root: Path) -> dict[str, set[str]]:
    """Return the stub path tokens of every tracked Python file under ``tests`` and ``benchmarks``, by path."""
    texts: dict[str, str] = {}
    for directory in ("tests", "benchmarks"):
        for path in tracked(root, directory, ".py"):
            texts[path.relative_to(root).as_posix()] = path.read_text(encoding="utf-8")
    return _path_tokens(texts)


def _named_by(stub: str, lane_root: str, tokens: dict[str, set[str]]) -> list[str]:
    """Return the tracked test and benchmark files that name a stub by its path below its lane root.

    A file names the stub when one of its path tokens is that path or ends with ``/`` and that path, so
    ``lif.jl`` is not found inside ``adaptive_lif.jl``.
    """
    short = stub.removeprefix(lane_root + "/")
    return sorted(
        name
        for name, held in tokens.items()
        if any(token == short or token.endswith("/" + short) for token in held)
    )


def lane_stubs(
    root: Path, mojo: dict[str, Any], julia: dict[str, Any]
) -> tuple[dict[str, list[Stub]], list[str]]:
    """Return the stubs of both lanes grouped by the original they copy, and the Julia files left unmapped.

    Parameters
    ----------
    root : Path
        The repository root.
    mojo, julia : dict
        The two committed lane inventories.

    Returns
    -------
    tuple
        Each original's stubs, and the stubs of either lane whose module no rule names (for Julia, also the
        parseable, unloaded, undeclared files whose header names none), listed rather than dropped.

    Raises
    ------
    ValueError
        When an ambiguous stub has no recorded resolution, or its resolution is not one of its candidates.
    """
    texts = _mentions(root)
    grouped: dict[str, list[Stub]] = {}
    unmapped: list[str] = []
    for source in mojo["sources"]:
        if source["class"] != "translated_stub":
            continue
        module = source["transcribed_module"]
        if source["transcription_rule"] == "ambiguous":
            resolution = AMBIGUOUS_STUBS.get(source["path"])
            if resolution is None or resolution["module"] not in source["transcription_candidates"]:
                raise ValueError(
                    f"{source['path']} is ambiguous and has no valid recorded resolution"
                )
            module = resolution["module"]
        if module is None:
            unmapped.append(source["path"])
            continue
        grouped.setdefault(module, []).append(
            {
                "lane": "mojo",
                "path": source["path"],
                "rule": source["transcription_rule"],
                "verbatim_line_share": source["verbatim_line_share"],
                "loaders": list(source["loaders"]),
                "named_by": _named_by(source["path"], "src/sc_neurocore/accel/mojo", texts),
                "declared": source["declared"],
            }
        )
    modules = julia_lane_inventory.package_modules(root)
    stripped: dict[str, set[str]] = {}

    def read(module: str) -> set[str]:
        """Return a package module's stripped lines, read once."""
        if module not in stripped:
            text = (root / module).read_text(encoding="utf-8")
            stripped[module] = {line.strip() for line in text.splitlines()}
        return stripped[module]

    for source in julia["sources"]:
        declared = (
            "authoritative" if source["authoritative"] else "mirror" if source["mirror"] else None
        )
        if not source["parses"]:
            lane, copied = "julia", source
        elif not (source["loaded"] or source["authoritative"] or source["mirror"]):
            lane = "julia_parseable_unloaded"
            found = julia_lane_inventory.transcription(
                (root / source["path"]).read_text(encoding="utf-8"), modules, read
            )
            copied = {**source, **found}
        else:
            continue
        module = copied["transcribed_module"]
        if module is None:
            unmapped.append(source["path"])
            continue
        grouped.setdefault(module, []).append(
            {
                "lane": lane,
                "path": source["path"],
                "rule": copied["transcription_rule"],
                "verbatim_line_share": copied["verbatim_line_share"],
                "loaders": list(source["loaders"]),
                "named_by": _named_by(source["path"], julia_lane_inventory.JULIA_ROOT, texts),
                "declared": declared,
            }
        )
    return grouped, sorted(unmapped)


def build(root: Path) -> Inventory:
    """Return the whole triage of the repository at ``root``.

    Parameters
    ----------
    root : Path
        The repository root, with both lane inventories committed.

    Returns
    -------
    Inventory
        The counts, every original, the deletion list, the compute order, and the stubs of either lane that no
        rule maps to a module.

    Raises
    ------
    ValueError
        When a reading names a module the rule already decides, or a module no stub copies.
    """
    mojo = json.loads((root / MOJO_RECORD).read_text(encoding="utf-8"))
    julia = json.loads((root / JULIA_RECORD).read_text(encoding="utf-8"))
    grouped, unmapped = lane_stubs(root, mojo, julia)
    for module in READINGS:
        if module not in grouped:
            raise ValueError(f"the reading of {module} names a module no stub copies")
    originals: list[Original] = []
    for module in sorted(grouped):
        found = markers((root / module).read_text(encoding="utf-8"))
        ruled = classify(found)
        reading = READINGS.get(module)
        if reading is not None and ruled != "unclear":
            raise ValueError(
                f"the reading of {module} overrides the rule, which classes it {ruled}"
            )
        stubs = sorted(grouped[module], key=lambda stub: (LANES.index(stub["lane"]), stub["path"]))
        originals.append(
            {
                "module": module,
                "class": ruled if reading is None else reading["class"],
                "decided_by": "rule" if reading is None else "reading",
                "reason": None
                if reading is None
                else f"{reading['reason']} (read by {reading['reader']})",
                "markers": found,
                "stubs": stubs,
                "used": any(
                    stub["loaders"]
                    or any(name.startswith("benchmarks/") for name in stub["named_by"])
                    for stub in stubs
                ),
            }
        )
    deletion: list[Deletion] = [
        {
            "module": original["module"],
            "lane": stub["lane"],
            "path": stub["path"],
            "loaders": stub["loaders"],
            "named_by": stub["named_by"],
            "declared": stub["declared"],
        }
        for original in originals
        if original["class"] == "application"
        for stub in original["stubs"]
    ]
    compute = sorted(
        (original for original in originals if original["class"] == "compute"),
        key=lambda original: (
            not original["used"],
            "abc".index(original["markers"]["basis"] or "c"),
            -len({stub["lane"] for stub in original["stubs"]}),
            original["module"],
        ),
    )
    alone = [classify_a_only(original["markers"]) for original in originals]
    variant: Variant = {
        "compute": alone.count("compute"),
        "application": alone.count("application"),
        "unclear": alone.count("unclear"),
        "application_modules": [
            original["module"]
            for original, ruled in zip(originals, alone, strict=True)
            if ruled == "application"
        ],
    }
    counts: dict[str, int] = {"originals": len(originals)}
    for name in CLASSES:
        counts[name] = sum(1 for original in originals if original["class"] == name)
        counts[f"{name}_stubs"] = sum(
            len(original["stubs"]) for original in originals if original["class"] == name
        )
    counts["decided_by_reading"] = sum(
        1 for original in originals if original["decided_by"] == "reading"
    )
    counts["ambiguous_stubs_resolved"] = len(AMBIGUOUS_STUBS)
    for lane in LANES:
        counts[f"stubs_{lane}"] = sum(
            1 for original in originals for stub in original["stubs"] if stub["lane"] == lane
        )
    counts["unmapped_mojo"] = sum(
        1 for path in unmapped if path.startswith("src/sc_neurocore/accel/mojo/")
    )
    counts["unmapped_julia"] = len(unmapped) - counts["unmapped_mojo"]
    return {
        "generator": "tools/accel_stub_triage.py",
        "counts": counts,
        "variant_a_only": variant,
        "originals": originals,
        "deletion_list": deletion,
        "compute_order": [original["module"] for original in compute],
        "unmapped": unmapped,
    }


def render(inventory: Inventory) -> str:
    """Return the triage as its committed text.

    Parameters
    ----------
    inventory : Inventory
        The triage from :func:`build`.

    Returns
    -------
    str
        Two-space indented JSON with one trailing newline.
    """
    return json.dumps(inventory, indent=2, ensure_ascii=True) + "\n"


def main(argv: Sequence[str] | None = None, builder: Callable[[Path], Inventory] = build) -> int:
    """Write, check or summarise the triage.

    Parameters
    ----------
    argv : sequence of str, optional
        Command-line arguments; ``sys.argv[1:]`` when omitted.
    builder : callable, optional
        Builds the triage of a root; tests replace it.

    Returns
    -------
    int
        ``0`` on success, ``1`` when ``--check`` finds the file absent or stale, ``2`` when the tree is refused.
    """
    options = argparse.ArgumentParser(description="Class each accelerated-lane stub's original.")
    options.add_argument("--write", action="store_true", help="write the triage")
    options.add_argument("--check", action="store_true", help="fail when the file has drifted")
    options.add_argument("--summary", action="store_true", help="print the counts")
    options.add_argument("--root", type=Path, default=REPO_ROOT)
    options.add_argument("--output", type=Path, default=None)
    arguments = options.parse_args(argv)
    root: Path = arguments.root
    output: Path = arguments.output if arguments.output is not None else root / DEFAULT_OUTPUT
    try:
        inventory = builder(root)
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
