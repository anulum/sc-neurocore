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
    assert len(facade_lines) <= 150


def test_responsibility_modules_partition_the_legacy_surface() -> None:
    modules = (topology, routing, thermal, rtl, link_protocols, power, partition)
    owners: dict[str, str] = {}
    for module in modules:
        for symbol in module.__all__:
            assert symbol not in owners, (
                f"{symbol} owned by both {owners[symbol]} and {module.__name__}"
            )
            owners[symbol] = module.__name__
        module_path = PACKAGE_ROOT / f"{module.__name__.rsplit('.', maxsplit=1)[-1]}.py"
        assert len(module_path.read_text(encoding="utf-8").splitlines()) <= 500
    assert set(owners) == set(chiplet_gen.__all__)
    assert len((PACKAGE_ROOT / "_sv.py").read_text(encoding="utf-8").splitlines()) <= 100
    assert all(
        len(path.read_text(encoding="utf-8").splitlines()) <= 500
        for path in Path(__file__).parent.glob("test_chiplet_*.py")
    )


def test_focused_import_graph_is_acyclic() -> None:
    expected_dependencies = {
        "_sv": set(),
        "topology": set(),
        "routing": {"topology"},
        "thermal": {"topology"},
        "rtl": {"_sv", "routing", "topology"},
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
    absent = [
        "src/sc_neurocore/accel/rust/safety/chiplet_gen.rs",
        "src/sc_neurocore/accel/go/services/chiplet_gen/chiplet_gen.go",
        "src/sc_neurocore/accel/julia/chiplet/chiplet_gen.jl",
        "src/sc_neurocore/accel/mojo/kernels/chiplet_gen.mojo",
    ]
    assert all(not (REPO_ROOT / relative).exists() for relative in absent)
    for registry in (
        REPO_ROOT / "src/sc_neurocore/accel/rust/safety/lib.rs",
        REPO_ROOT / "src/sc_neurocore/accel/rust/safety/mod.rs",
    ):
        assert "pub mod chiplet_gen;" not in registry.read_text(encoding="utf-8")
