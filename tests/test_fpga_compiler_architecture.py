# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — FPGA compiler responsibility-boundary tests

"""Keep the network compiler facade acyclic and organised by responsibility."""

from __future__ import annotations

import ast
from pathlib import Path

_PACKAGE = Path(__file__).parents[1] / "src/sc_neurocore/nir_bridge"
_RESPONSIBILITY_ENTRYPOINTS = {
    "fpga_aer_interconnect.py": "build_aer_interconnect",
    "fpga_connection_routing.py": "validate_connection_routing",
    "fpga_direct_interconnect.py": "build_direct_interconnect",
    "fpga_folded_interconnect.py": "build_folded_interconnect",
    "fpga_neuron_rtl.py": "build_neuron_module",
    "fpga_scnir_hierarchy.py": "build_scnir_hierarchy_modules",
    "fpga_weight_rom.py": "build_weight_rom",
}


def _tree(filename: str) -> ast.Module:
    """Parse one tracked compiler module from the repository source tree."""
    return ast.parse((_PACKAGE / filename).read_text(encoding="utf-8"))


def test_facade_owns_only_resource_policy_and_pipeline_composition() -> None:
    tree = _tree("fpga_compiler.py")
    definitions = {
        node.name
        for node in tree.body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
    }

    assert definitions == {"_check_synthesis_resource_bounds", "compile_network_to_fpga"}


def test_responsibility_modules_are_acyclic_and_expose_their_entrypoint() -> None:
    for filename, entrypoint in _RESPONSIBILITY_ENTRYPOINTS.items():
        tree = _tree(filename)
        definitions = {
            node.name
            for node in tree.body
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
        }
        facade_imports = {
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
            and node.module is not None
            and node.module.endswith("fpga_compiler")
        }

        assert entrypoint in definitions, filename
        assert facade_imports == set(), filename
