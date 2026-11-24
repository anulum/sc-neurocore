# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_nir_import.py

"""A one-node graph helper for the dict-form NIR importer tests."""

from __future__ import annotations

from sc_neurocore.compiler.intelligence.nir_import import (
    NIRGraph,
    import_nir_graph,
)


def _one(node_type: object = None, **params: object) -> NIRGraph:
    """Import a graph of one node ``n0`` with these parameters and, unless None, this ``type``."""
    spec = dict(params)
    if node_type is not None:
        spec["type"] = node_type
    return import_nir_graph({"nodes": {"n0": spec}, "edges": []})


__all__ = ["import_nir_graph", "_one"]
