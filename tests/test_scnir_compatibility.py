# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for SC-NIR compatibility matrix

"""Contract tests for the executable SC-NIR compatibility matrix."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path

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


def test_scnir_compatibility_matrix_covers_parser_primitives() -> None:
    validate_scnir_compatibility_matrix()

    primitives = {row.nir_primitive for row in scnir_compatibility_matrix()}
    assert {primitive.__name__ for primitive in NODE_MAP}.issubset(primitives)
    assert "NIRGraph" in primitives


def test_scnir_compatibility_matrix_evidence_paths_exist() -> None:
    validate_scnir_compatibility_matrix(evidence_root=REPO_ROOT)


def test_scnir_compatibility_matrix_rejects_missing_evidence_root(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="missing audit evidence paths"):
        validate_scnir_compatibility_matrix(evidence_root=tmp_path)


def test_scnir_compatibility_audit_report_summarises_evidence() -> None:
    from sc_neurocore.ir import build_scnir_compatibility_audit

    report = build_scnir_compatibility_audit(evidence_root=REPO_ROOT)

    matrix = json.loads(json.dumps(scnir_compatibility_matrix_dicts(), sort_keys=True))
    evidence_paths = sorted(
        {path for row in scnir_compatibility_matrix() for path in row.audit_evidence}
    )
    matrix_digest = hashlib.sha256(
        (json.dumps(matrix, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
    ).hexdigest()
    first_evidence = REPO_ROOT / evidence_paths[0]
    first_digest = hashlib.sha256(first_evidence.read_bytes()).hexdigest()

    assert report["schema_version"] == "sc-neurocore.scnir.compatibility-audit.v0.2"
    assert report["status"] == "valid"
    assert report["evidence_root"] == str(REPO_ROOT.resolve())
    assert report["primitive_count"] == len(matrix)
    assert report["support_level_counts"]["metadata_and_hdl"] >= 1
    assert report["closure_status"] == "closed_for_local_handoff"
    assert report["closure_blocker_count"] == 0
    assert report["parser_only_primitives"] == []
    assert report["metadata_only_primitives"] == []
    assert report["boundary_primitives"] == ["Input", "Output"]
    assert report["closed_handoff_primitives"] == sorted(
        row.nir_primitive
        for row in scnir_compatibility_matrix()
        if row.support_level == "metadata_and_hdl"
    )
    assert report["requires_external_hardware_evidence"] is True
    assert report["external_hardware_evidence_status"] == "not_claimed"
    assert report["audit_evidence_file_count"] == len(evidence_paths)
    assert report["audit_evidence_paths"] == evidence_paths
    assert report["matrix_sha256"] == matrix_digest
    assert report["audit_evidence_files"][0] == {
        "path": evidence_paths[0],
        "sha256": first_digest,
        "size_bytes": first_evidence.stat().st_size,
    }
    assert report["matrix"] == matrix


def test_scnir_compatibility_cli_validates_evidence_root(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        "sys.argv",
        ["sc-neurocore", "scnir", "compatibility", str(REPO_ROOT)],
    )

    assert main() == 0
    assert "SC-NIR compatibility matrix valid" in capsys.readouterr().out


def test_scnir_compatibility_cli_writes_matrix_report(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    output = tmp_path / "scnir_compatibility.json"
    monkeypatch.setattr(
        "sys.argv",
        ["sc-neurocore", "scnir", "compatibility", str(REPO_ROOT), "--output", str(output)],
    )

    assert main() == 0
    assert f"report written: {output}" in capsys.readouterr().out
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert isinstance(payload, list)
    assert payload == json.loads(json.dumps(scnir_compatibility_matrix_dicts(), sort_keys=True))
    assert payload[0]["nir_primitive"] == "Input"


def test_scnir_closure_audit_cli_writes_versioned_report(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    output = tmp_path / "scnir_closure_audit.json"
    monkeypatch.setattr(
        "sys.argv",
        ["sc-neurocore", "scnir", "closure-audit", str(REPO_ROOT), "--output", str(output)],
    )

    assert main() == 0
    assert f"report written: {output}" in capsys.readouterr().out
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "sc-neurocore.scnir.compatibility-audit.v0.2"
    assert payload["status"] == "valid"
    assert payload["primitive_count"] == len(scnir_compatibility_matrix())
    assert payload["closure_status"] == "closed_for_local_handoff"
    assert payload["closure_blocker_count"] == 0
    assert payload["parser_only_primitives"] == []
    assert payload["metadata_only_primitives"] == []
    assert payload["boundary_primitives"] == ["Input", "Output"]
    assert payload["requires_external_hardware_evidence"] is True
    assert payload["external_hardware_evidence_status"] == "not_claimed"
    assert payload["audit_evidence_file_count"] >= 1
    assert payload["matrix_sha256"]
    assert payload["audit_evidence_files"][0]["sha256"]
    assert payload["matrix"][0]["nir_primitive"] == "Input"


def test_scnir_compatibility_cli_rejects_missing_evidence_root(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        "sys.argv",
        ["sc-neurocore", "scnir", "compatibility", str(tmp_path)],
    )

    assert main() == 1
    assert "SC-NIR compatibility matrix invalid" in capsys.readouterr().out


def test_scnir_compatibility_matrix_is_deterministic_json() -> None:
    left = scnir_compatibility_matrix_dicts()
    right = scnir_compatibility_matrix_dicts()

    assert left == right
    assert json.loads(json.dumps(left, sort_keys=True))[0]["nir_primitive"] == "Input"


def test_scnir_compatibility_matrix_marks_closed_hdl_population_rows() -> None:
    rows = {row.nir_primitive: row for row in scnir_compatibility_matrix()}

    lif = rows["LIF"]
    assert lif.support_level == "metadata_and_hdl"
    assert "signal_kind=spike" in lif.scnir_stream_metadata
    assert "encoding=unipolar" in lif.scnir_stream_metadata
    assert "lfsr16" in lif.source_metadata
    assert "sobol16" in lif.source_metadata

    li = rows["LI"]
    assert li.support_level == "metadata_and_hdl"
    assert "signal_kind=analogue_state" in li.scnir_stream_metadata
    assert "direct analogue-state MAC" in li.hdl_support


def test_scnir_compatibility_matrix_does_not_overclaim_parser_only_rows() -> None:
    rows = {row.nir_primitive: row for row in scnir_compatibility_matrix()}

    conv1d = rows["Conv1d"]
    assert conv1d.support_level == "metadata_and_hdl"
    assert "convolution_lowered_weight" in conv1d.scnir_stream_metadata
    assert "dense Toeplitz" in conv1d.hdl_support

    conv2d = rows["Conv2d"]
    assert conv2d.support_level == "metadata_and_hdl"
    assert "convolution_lowered_weight" in conv2d.scnir_stream_metadata
    assert "dense 2D convolution" in conv2d.hdl_support

    for primitive in ("SumPool2d", "AvgPool2d"):
        row = rows[primitive]
        assert row.support_level == "metadata_and_hdl"
        assert "pool2d_lowered_weight" in row.scnir_stream_metadata
        assert "dense pooling" in row.hdl_support

    scale = rows["Scale"]
    assert scale.support_level == "metadata_and_hdl"
    assert "folded_weight_scale" in scale.scnir_stream_metadata
    assert "folded fixed-point gain" in scale.hdl_support

    flatten = rows["Flatten"]
    assert flatten.support_level == "metadata_and_hdl"
    assert "shape_preserving_flatten" in flatten.scnir_stream_metadata
    assert "fixed-point weight indexing" in flatten.hdl_support

    threshold = rows["Threshold"]
    assert threshold.support_level == "metadata_and_hdl"
    assert "threshold_transform" in threshold.scnir_stream_metadata
    assert "fixed-point comparator" in threshold.hdl_support

    delay = rows["Delay"]
    assert delay.support_level == "metadata_and_hdl"
    assert "delay_steps>=0 or vector[int>=0]" in delay.scnir_stream_metadata
    assert "per-source delay taps" in delay.hdl_support

    integrator = rows["I"]
    assert integrator.support_level == "metadata_and_hdl"
    assert "signal_kind=analogue_state" in integrator.scnir_stream_metadata
    assert "integrator state-update module" in integrator.hdl_support

    nested = rows["NIRGraph"]
    assert nested.support_level == "metadata_and_hdl"
    assert "inline_single_port_subgraph" in nested.scnir_stream_metadata
    assert "inline_exact_multiport_subgraph" in nested.scnir_stream_metadata
    assert "hierarchy_instance_metadata" in nested.scnir_stream_metadata
    assert "manifest_hierarchy_counts" in nested.source_metadata
    assert "manifest_external_input_layout" in nested.source_metadata
    assert "hierarchy_boundary_hdl_modules" in nested.source_metadata
    assert "top_module_hierarchy_contract_instances" in nested.source_metadata
    assert "scalar_hierarchy_weight_outputs" in nested.source_metadata
    assert "packed_vector_matrix_hierarchy_weight_outputs" in nested.source_metadata
    assert "namespaced inline fixed-point terms" in nested.hdl_support
    assert "stable external input-bus lanes" in nested.hdl_support
    assert "standalone hierarchy boundary module artefacts" in nested.hdl_support
    assert "top-level contract instances" in nested.hdl_support
    assert "packed hierarchy weight outputs" in nested.hdl_support
    assert "tests/test_cli.py" in nested.audit_evidence
    assert "tests/test_scnir_handoff_audit.py" in nested.audit_evidence
    assert "multi-output" in nested.limitation
    assert "Ambiguous" in nested.limitation
    assert (
        "Ambiguous multi-port nested NIRGraph boundary mappings still fail closed"
        in nested.limitation
    )


def test_scnir_compatibility_matrix_records_weight_and_recurrent_delay_semantics() -> None:
    rows = {row.nir_primitive: row for row in scnir_compatibility_matrix()}

    affine = rows["Affine"]
    assert affine.support_level == "metadata_and_hdl"
    assert "signal_kind=weight" in affine.scnir_stream_metadata
    assert "encoding=bipolar" in affine.scnir_stream_metadata

    linear = rows["Linear"]
    assert "delay_steps=0_or_1" in linear.scnir_stream_metadata
    assert "recurrent unit-delay" in linear.limitation


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


def _patch_matrix(monkeypatch, rows: tuple[SCNIRCompatibilityRow, ...]) -> None:
    monkeypatch.setattr("sc_neurocore.nir_bridge.node_map.NODE_MAP", {_Foo: None})
    monkeypatch.setattr("sc_neurocore.ir.scnir_compatibility._MATRIX", rows)


def test_validate_matrix_flags_missing_parser_primitive(monkeypatch) -> None:
    _patch_matrix(monkeypatch, ())
    with pytest.raises(ValueError, match="misses parser primitives"):
        validate_scnir_compatibility_matrix()


def test_validate_matrix_flags_stale_primitive(monkeypatch) -> None:
    _patch_matrix(monkeypatch, (_row("_Foo"), _row("Ghost")))
    with pytest.raises(ValueError, match="stale primitives"):
        validate_scnir_compatibility_matrix()


def test_validate_matrix_flags_duplicate_row(monkeypatch) -> None:
    _patch_matrix(monkeypatch, (_row("_Foo"), _row("_Foo")))
    with pytest.raises(ValueError, match="duplicate SC-NIR compatibility row"):
        validate_scnir_compatibility_matrix()


def test_validate_matrix_flags_hdl_support_without_metadata(monkeypatch) -> None:
    _patch_matrix(monkeypatch, (_row("_Foo", support_level="metadata_and_hdl", stream_metadata=()),))
    with pytest.raises(ValueError, match="claims HDL support without stream metadata"):
        validate_scnir_compatibility_matrix()


def test_validate_matrix_flags_missing_audit_evidence(monkeypatch) -> None:
    _patch_matrix(monkeypatch, (_row("_Foo", audit_evidence=()),))
    with pytest.raises(ValueError, match="no audit evidence pointer"):
        validate_scnir_compatibility_matrix()
