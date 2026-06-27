# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Protobuf schema surface tests

"""Regression tests for the data-only Protobuf schema surface."""

from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PROTO_DIR = REPO_ROOT / "src" / "sc_neurocore" / "proto"


def test_proto_tree_is_data_only_schema_surface() -> None:
    """The Protobuf surface ships schemas, not a Python import package."""
    schema_names = {path.name for path in PROTO_DIR.glob("*.proto")}
    docs = (REPO_ROOT / "docs" / "api" / "proto.md").read_text(encoding="utf-8")

    assert schema_names == {"core.proto", "telemetry.proto"}
    assert not (PROTO_DIR / "__init__.py").exists()
    assert "data-only schema directory" in docs
    assert "does not ship a `sc_neurocore.proto` Python package" in docs


def test_proto_schemas_keep_canonical_generation_metadata() -> None:
    """The schema files must keep proto3 and canonical downstream package metadata."""
    core = (PROTO_DIR / "core.proto").read_text(encoding="utf-8")
    telemetry = (PROTO_DIR / "telemetry.proto").read_text(encoding="utf-8")

    assert core.startswith('syntax = "proto3";')
    assert "package vision2030.core;" in core
    assert 'option go_package = "github.com/anulum/sc-neurocore/vision2030/proto/core";' in core
    assert "message Tensor" in core
    assert "message BitstreamMetadata" in core

    assert telemetry.startswith('syntax = "proto3";')
    assert "package vision2030.telemetry;" in telemetry
    assert 'import "core.proto";' in telemetry
    assert (
        'option go_package = "github.com/anulum/sc-neurocore/vision2030/proto/telemetry";'
        in telemetry
    )
    assert "message HILFrame" in telemetry
