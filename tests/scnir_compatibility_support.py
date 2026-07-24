# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_scnir_compatibility.py

from __future__ import annotations

"""Contract tests for the executable SC-NIR compatibility matrix."""

import json

import hashlib

from pathlib import Path

from typing import cast

import pytest

pytest.importorskip("nir")

from sc_neurocore.ir import (
    SCNIRCompatibilityRow,
    scnir_compatibility_matrix,
    scnir_compatibility_matrix_dicts,
    validate_scnir_compatibility_matrix,
)

from sc_neurocore.cli import main

from sc_neurocore.nir_bridge.node_map import NODE_MAP

REPO_ROOT = Path(__file__).resolve().parents[1]

class _Foo:  # parser primitive stand-in; __name__ drives the matrix comparison
    pass

def _row(
    primitive: str,
    *,
    support_level: str = "boundary",
    stream_metadata: tuple[str, ...] = ("signal_kind=spike",),
    audit_evidence: tuple[str, ...] = ("tests/test_scnir_compatibility.py",),
) -> SCNIRCompatibilityRow:
    return SCNIRCompatibilityRow(
        nir_primitive=primitive,
        support_level=support_level,  # type: ignore[arg-type]
        parser_node="node",
        neuron_graph_lowering="lowering",
        scnir_stream_metadata=stream_metadata,
        source_metadata=(),
        hdl_support="none",
        audit_evidence=audit_evidence,
        limitation="",
    )

def _patch_matrix(
    monkeypatch: pytest.MonkeyPatch,
    rows: tuple[SCNIRCompatibilityRow, ...],
) -> None:
    monkeypatch.setattr("sc_neurocore.nir_bridge.node_map.NODE_MAP", {_Foo: None})
    monkeypatch.setattr("sc_neurocore.ir.scnir_compatibility._MATRIX", rows)


__all__ = ['json', 'hashlib', 'Path', 'cast', 'pytest', 'SCNIRCompatibilityRow', 'scnir_compatibility_matrix', 'scnir_compatibility_matrix_dicts', 'validate_scnir_compatibility_matrix', 'main', 'NODE_MAP', 'REPO_ROOT', '_Foo', '_row', '_patch_matrix']
