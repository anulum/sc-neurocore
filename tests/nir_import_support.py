# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_nir_import.py

from __future__ import annotations

"""Tests for the lightweight dict-form NIR importer.

Covers the broadened node-type support (the six NIR point-neuron types plus the
Izhikevich extension), the shared-template reconciliation with
``nir_bridge.neuron_templates``, alias/fallback resolution, multi-compartment
state equations, and threshold/reset/parameter resolution.
"""
import pytest
from sc_neurocore.compiler.intelligence.nir_import import (
    NEURON_TEMPLATES,
    import_nir_graph,
)


def _one(node_type=None, **params):
    spec = dict(params)
    if node_type is not None:
        spec["type"] = node_type
    return import_nir_graph({"nodes": {"n0": spec}, "edges": []})


__all__ = ["pytest", "NEURON_TEMPLATES", "import_nir_graph", "_one"]
