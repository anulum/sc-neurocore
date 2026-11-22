# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — DNA export, laboratory planning and text view division contract

"""Each DNA export, planning and view helper is defined in the module that owns its responsibility.

``bridges/dna_io.py`` held interchange exports, synthesis-order and assembly planning, and text
views. The planning helpers now live in ``dna_lab`` and the text views in ``dna_views``. The
façade ``dna_mapper`` still exposes every name as the same object, under its historical
``__module__``, so pickles and imports through the façade are unchanged.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path
from typing import Any

import sc_neurocore.bridges.dna_io as dna_io
import sc_neurocore.bridges.dna_lab as dna_lab
import sc_neurocore.bridges.dna_mapper as dna_mapper
import sc_neurocore.bridges.dna_views as dna_views

OWNERS = {
    dna_io: ("export_fasta", "export_genbank", "export_json", "export_nupack_input"),
    dna_lab: ("PlateLayout", "estimate_cost", "generate_protocol"),
    dna_views: ("visualize_circuit", "visualize_kinetics"),
}


def _defining_file(value: Any) -> str:
    """Return the file that defines a function or a class, by its code, not its ``__module__``.

    The façade rewrites ``__module__`` for compatibility, so a class is located by one of its
    own methods' code objects.

    Parameters
    ----------
    value : Any
        A function or a class.

    Returns
    -------
    str
        The defining file's name.
    """
    if inspect.isclass(value):
        value = next(item for item in vars(value).values() if inspect.isfunction(item))
    return Path(value.__code__.co_filename).name


def test_each_helper_is_defined_by_its_owner_and_exposed_by_the_facade() -> None:
    """The façade's object is the owner's object, and its source is the owner's file."""
    for owner, names in OWNERS.items():
        owner_file = Path(inspect.getfile(owner)).name
        for name in names:
            value = getattr(owner, name)
            assert getattr(dna_mapper, name) is value, name
            assert _defining_file(value) == owner_file, name
            assert value.__module__ == "sc_neurocore.bridges.dna_mapper", name


def test_no_module_carries_another_responsibility() -> None:
    """Each module's top-level definitions are exactly its own helpers."""
    for owner, names in OWNERS.items():
        tree = ast.parse(Path(inspect.getfile(owner)).read_text(encoding="utf-8"))
        defined = {
            node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))
        }
        assert defined == set(names), owner.__name__
