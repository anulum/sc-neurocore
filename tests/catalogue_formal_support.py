# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_catalogue_formal.py

from __future__ import annotations

"""Inventory and optional SymbiYosys smoke for dual-axis perfect models.

These tests drive the real emitted ``.sby`` / formal RTL under
``hdl/formal/catalogue/`` and the emitter tool — not a re-implemented harness."""

import shutil


import subprocess


import sys


from pathlib import Path


import pytest


ROOT = Path(__file__).resolve().parents[1]


CATALOGUE = ROOT / "hdl" / "formal" / "catalogue"


EMITTER = ROOT / "tools" / "emit_catalogue_formal.py"


def _perfect_class_names() -> set[str]:
    if sys.version_info >= (3, 11):
        import tomllib
    else:
        import tomli as tomllib

    from sc_neurocore.neurons.descriptor_tiers import is_perfect
    from sc_neurocore.neurons.model_descriptor import parse_model_descriptor

    names: set[str] = set()
    desc_dir = ROOT / "src" / "sc_neurocore" / "neurons" / "model_descriptors"
    for path in desc_dir.glob("*.toml"):
        desc = parse_model_descriptor(tomllib.loads(path.read_text(encoding="utf-8")))
        if is_perfect(desc):
            names.add(desc.class_name)
    return names


__all__ = [
    "shutil",
    "subprocess",
    "sys",
    "Path",
    "pytest",
    "ROOT",
    "CATALOGUE",
    "EMITTER",
    "_perfect_class_names",
]
