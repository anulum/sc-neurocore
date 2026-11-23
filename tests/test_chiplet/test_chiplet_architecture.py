# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Chiplet responsibility-boundary contracts

"""Architecture and historical-compatibility tests for the chiplet split."""

from __future__ import annotations

import ast
import pickle
from pathlib import Path

from sc_neurocore.chiplet import ChipletDie, ChipletTopology, InterposerTech
from sc_neurocore.chiplet import chiplet_gen
from sc_neurocore.chiplet import link_protocols, partition, power, routing, rtl, thermal, topology


PACKAGE_ROOT = Path(__file__).parents[2] / "src" / "sc_neurocore" / "chiplet"
REPO_ROOT = Path(__file__).parents[2]


def test_facade_defines_no_new_public_behaviour() -> None:
    """The ``chiplet_gen`` facade only re-exports: no definitions and its 36 names."""
    facade = PACKAGE_ROOT / "chiplet_gen.py"
    facade_lines = facade.read_text(encoding="utf-8").splitlines()
    tree = ast.parse("\n".join(facade_lines))
    definitions = [
        node.name
        for node in tree.body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    assert definitions == []
    assert len(chiplet_gen.__all__) == 36


# The private helpers the RTL emission is divided into, with the top-level definitions each owns.
PRIVATE_OWNED_DEFINITIONS: dict[str, frozenset[str]] = {
    "_sv": frozenset({"_require_sv_identifier", "sv_signed"}),
    "_rtl_die": frozenset({"_arbiter", "_route_inits", "emit_die_wrapper"}),
    "_rtl_links": frozenset({"_cycles", "bridge_timing", "emit_bridge", "emit_routing_table"}),
    "_rtl_top": frozenset({"die_clock", "die_reset", "emit_constraints", "emit_top"}),
}

# Concerns a chiplet test file may cover: one responsibility module each, plus the
# package-wide edge cases of the facade and the public API.
TEST_CONCERNS = (
    "topology",
    "routing",
    "thermal",
    "rtl",
    "link_protocols",
    "power",
    "partition",
    "gen_edge_cases",
    "public_api",
    "architecture",
)


def test_responsibility_modules_partition_the_legacy_surface() -> None:
    """Each public chiplet symbol has exactly one owning module, and each private helper its own names."""
    modules = (topology, routing, thermal, rtl, link_protocols, power, partition)
    owners: dict[str, str] = {}
    for module in modules:
        for symbol in module.__all__:
            assert symbol not in owners, (
                f"{symbol} owned by both {owners[symbol]} and {module.__name__}"
            )
            owners[symbol] = module.__name__
    assert set(owners) == set(chiplet_gen.__all__)
    kinds = (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
    for name, declared in PRIVATE_OWNED_DEFINITIONS.items():
        source = (PACKAGE_ROOT / f"{name}.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        assert {node.name for node in tree.body if isinstance(node, kinds)} == declared, name
        if name != "_sv":
            assert all(len(line) <= 100 for line in source.splitlines()), name


def test_each_chiplet_test_file_covers_one_concern() -> None:
    """Every ``test_chiplet_*`` file names the module or package-wide concern it tests."""
    for path in Path(__file__).parent.glob("test_chiplet_*.py"):
        stem = path.stem.removeprefix("test_chiplet_")
        assert any(
            stem == concern or stem.startswith(f"{concern}_") for concern in TEST_CONCERNS
        ), path.name


def test_focused_import_graph_is_acyclic() -> None:
    """The chiplet modules import exactly the expected dependencies, without a cycle."""
    expected_dependencies = {
        "_sv": set(),
        "topology": set(),
        "routing": {"topology"},
        "thermal": {"topology"},
        "rtl": {"_rtl_die", "_rtl_links", "_rtl_top", "routing", "topology"},
        "_rtl_die": {"_sv", "routing", "topology"},
        "_rtl_links": {"_sv", "routing", "topology"},
        "_rtl_top": {"_sv", "topology"},
        "link_protocols": {"_sv", "topology"},
        "power": {"_sv"},
        "partition": {"routing"},
    }
    module_names = set(expected_dependencies)
    dependencies: dict[str, set[str]] = {}
    for name in module_names:
        tree = ast.parse((PACKAGE_ROOT / f"{name}.py").read_text(encoding="utf-8"))
        dependencies[name] = {
            node.module.rsplit(".", maxsplit=1)[-1]
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
            and node.module is not None
            and node.module.startswith("sc_neurocore.chiplet.")
            and node.module.rsplit(".", maxsplit=1)[-1] in module_names
        }
    assert dependencies == expected_dependencies

    resolved: set[str] = set()
    remaining = dict(dependencies)
    while remaining:
        ready = {name for name, deps in remaining.items() if deps <= resolved}
        assert ready, f"cyclic chiplet dependency graph: {remaining}"
        resolved.update(ready)
        for name in ready:
            del remaining[name]


def test_pickle_qualified_names_survive_the_move() -> None:
    """Values made before the split pickle back to equal values of the same type."""
    values = [
        ChipletDie(3),
        ChipletTopology.ring(2),
        InterposerTech.UCIE,
    ]
    for value in values:
        restored = pickle.loads(pickle.dumps(value))
        assert restored == value
        assert type(restored) is type(value)


def test_false_polyglot_mirrors_and_registry_entries_are_absent() -> None:
    """No chiplet generator mirror exists, and the Rust safety crate declares none."""
    absent = [
        "src/sc_neurocore/accel/rust/safety/chiplet_gen.rs",
        "src/sc_neurocore/accel/go/services/chiplet_gen/chiplet_gen.go",
        "src/sc_neurocore/accel/julia/chiplet/chiplet_gen.jl",
        "src/sc_neurocore/accel/mojo/kernels/chiplet_gen.mojo",
    ]
    assert all(not (REPO_ROOT / relative).exists() for relative in absent)
    registry = REPO_ROOT / "src/sc_neurocore/accel/rust/safety/lib.rs"
    assert "pub mod chiplet_gen;" not in registry.read_text(encoding="utf-8")
