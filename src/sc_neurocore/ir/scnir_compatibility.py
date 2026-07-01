# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — SC-NIR compatibility matrix

"""Executable compatibility matrix for NIR primitives in the SC-NIR pipeline.

The NIR parser supports more primitives than the current SC-NIR to FPGA
handoff path can lower.  This module makes that distinction explicit and
machine-checkable so documentation, tests, and future closure gates cannot
silently over-claim support.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

SCNIR_COMPATIBILITY_AUDIT_VERSION = "sc-neurocore.scnir.compatibility-audit.v0.2"

SCNIRSupportLevel = Literal[
    "boundary",
    "parser_only",
    "metadata_only",
    "metadata_and_hdl",
]


@dataclass(frozen=True, slots=True)
class SCNIRCompatibilityRow:
    """One compatibility row for a NIR primitive."""

    nir_primitive: str
    support_level: SCNIRSupportLevel
    parser_node: str
    neuron_graph_lowering: str
    scnir_stream_metadata: tuple[str, ...]
    source_metadata: tuple[str, ...]
    hdl_support: str
    audit_evidence: tuple[str, ...]
    limitation: str

    def as_dict(self) -> dict[str, object]:
        """Return a deterministic JSON-ready row."""
        payload = asdict(self)
        payload["scnir_stream_metadata"] = list(self.scnir_stream_metadata)
        payload["source_metadata"] = list(self.source_metadata)
        payload["audit_evidence"] = list(self.audit_evidence)
        return payload


_MATRIX: tuple[SCNIRCompatibilityRow, ...] = (
    SCNIRCompatibilityRow(
        nir_primitive="Input",
        support_level="boundary",
        parser_node="SCInputNode",
        neuron_graph_lowering="input boundary",
        scnir_stream_metadata=(),
        source_metadata=(),
        hdl_support="external input bus",
        audit_evidence=("tests/test_scnir_handoff_audit.py", "tests/test_nir_fpga_pipeline.py"),
        limitation="Boundary node; it does not create an SC-NIR stream by itself.",
    ),
    SCNIRCompatibilityRow(
        nir_primitive="Output",
        support_level="boundary",
        parser_node="SCOutputNode",
        neuron_graph_lowering="output boundary",
        scnir_stream_metadata=(),
        source_metadata=(),
        hdl_support="top-level output bus",
        audit_evidence=("tests/test_scnir_handoff_audit.py", "tests/test_nir_fpga_pipeline.py"),
        limitation="Boundary node; output semantics come from the upstream population stream.",
    ),
    SCNIRCompatibilityRow(
        nir_primitive="LIF",
        support_level="metadata_and_hdl",
        parser_node="SCLIFNode",
        neuron_graph_lowering="lif population",
        scnir_stream_metadata=("signal_kind=spike", "encoding=unipolar", "delay_steps=0"),
        source_metadata=("lfsr16", "sobol16", "seed", "bitstream_length", "precision"),
        hdl_support="canonical ODE module plus direct or AER interconnect",
        audit_evidence=("tests/test_scnir_fpga_integration.py", "tests/test_nir_fpga_pipeline.py"),
        limitation="Shared RTL neuron module requires homogeneous per-neuron parameters.",
    ),
    SCNIRCompatibilityRow(
        nir_primitive="IF",
        support_level="metadata_and_hdl",
        parser_node="SCIFNode",
        neuron_graph_lowering="if population",
        scnir_stream_metadata=("signal_kind=spike", "encoding=unipolar", "delay_steps=0"),
        source_metadata=("lfsr16", "sobol16", "seed", "bitstream_length", "precision"),
        hdl_support="canonical ODE module plus direct or AER interconnect",
        audit_evidence=("tests/test_nir_fpga_pipeline.py",),
        limitation="Shared RTL neuron module requires homogeneous per-neuron parameters.",
    ),
    SCNIRCompatibilityRow(
        nir_primitive="LI",
        support_level="metadata_and_hdl",
        parser_node="SCLINode",
        neuron_graph_lowering="li population",
        scnir_stream_metadata=("signal_kind=analogue_state", "encoding=bipolar", "delay_steps=0"),
        source_metadata=("lfsr16", "sobol16", "seed", "bitstream_length", "precision"),
        hdl_support="canonical ODE module with direct analogue-state MAC routing",
        audit_evidence=("tests/test_scnir_fpga_integration.py", "tests/test_nir_fpga_pipeline.py"),
        limitation="Analogue-state streams stay on direct fixed-point MAC routes in mixed graphs.",
    ),
    SCNIRCompatibilityRow(
        nir_primitive="I",
        support_level="metadata_and_hdl",
        parser_node="SCIntegratorNode",
        neuron_graph_lowering="integrator population",
        scnir_stream_metadata=("signal_kind=analogue_state", "encoding=bipolar", "delay_steps=0"),
        source_metadata=("lfsr16", "sobol16", "seed", "bitstream_length", "precision"),
        hdl_support="canonical integrator state-update module with direct analogue-state MAC routing",
        audit_evidence=("tests/test_scnir_convert.py", "tests/test_scnir_fpga_integration.py"),
        limitation="Shared RTL neuron module requires homogeneous integrator gain r.",
    ),
    SCNIRCompatibilityRow(
        nir_primitive="CubaLIF",
        support_level="metadata_and_hdl",
        parser_node="SCCubaLIFNode",
        neuron_graph_lowering="cuba_lif population",
        scnir_stream_metadata=("signal_kind=spike", "encoding=unipolar", "delay_steps=0"),
        source_metadata=("lfsr16", "sobol16", "seed", "bitstream_length", "precision"),
        hdl_support="canonical two-state ODE module plus direct or AER interconnect",
        audit_evidence=("tests/test_nir_fpga_pipeline.py",),
        limitation="Shared RTL neuron module requires homogeneous tau_syn, tau_mem, r, and w_in.",
    ),
    SCNIRCompatibilityRow(
        nir_primitive="CubaLI",
        support_level="metadata_and_hdl",
        parser_node="SCCubaLINode",
        neuron_graph_lowering="cuba_li population",
        scnir_stream_metadata=("signal_kind=analogue_state", "encoding=bipolar", "delay_steps=0"),
        source_metadata=("lfsr16", "sobol16", "seed", "bitstream_length", "precision"),
        hdl_support="canonical two-state ODE module with direct analogue-state MAC routing",
        audit_evidence=("tests/test_scnir_fpga_integration.py", "tests/test_nir_fpga_pipeline.py"),
        limitation="Analogue-state streams stay on direct fixed-point MAC routes in mixed graphs.",
    ),
    SCNIRCompatibilityRow(
        nir_primitive="Affine",
        support_level="metadata_and_hdl",
        parser_node="SCAffineNode",
        neuron_graph_lowering="weighted connection with bias",
        scnir_stream_metadata=("signal_kind=weight", "encoding=bipolar", "delay_steps=0"),
        source_metadata=("lfsr16", "sobol16", "seed", "bitstream_length", "precision"),
        hdl_support="weight ROM plus direct or weighted-event interconnect",
        audit_evidence=("tests/test_scnir_fpga_integration.py", "tests/test_nir_fpga_pipeline.py"),
        limitation="The downstream neuron population owns the connection stream layer.",
    ),
    SCNIRCompatibilityRow(
        nir_primitive="Linear",
        support_level="metadata_and_hdl",
        parser_node="SCLinearNode",
        neuron_graph_lowering="weighted connection without bias",
        scnir_stream_metadata=("signal_kind=weight", "encoding=bipolar", "delay_steps=0_or_1"),
        source_metadata=("lfsr16", "sobol16", "seed", "bitstream_length", "precision"),
        hdl_support="weight ROM plus direct or weighted-event interconnect",
        audit_evidence=("tests/test_scnir_fpga_integration.py", "tests/test_nir_fpga_pipeline.py"),
        limitation="Parser-inserted recurrent unit-delay streams are represented as delay_steps=1.",
    ),
    SCNIRCompatibilityRow(
        nir_primitive="Scale",
        support_level="metadata_and_hdl",
        parser_node="SCScaleNode",
        neuron_graph_lowering="adjacent source-side or post-weight scale folded into connection weights",
        scnir_stream_metadata=("signal_kind=weight", "folded_weight_scale"),
        source_metadata=("lfsr16", "sobol16", "seed", "bitstream_length", "precision"),
        hdl_support="folded fixed-point gain in direct/AER weight terms",
        audit_evidence=("tests/test_scnir_fpga_integration.py",),
        limitation=(
            "Scale is hardware-closed only when adjacent to an Affine/Linear "
            "connection; ambiguous fan-in/fan-out or incompatible scale lengths "
            "fail closed before FPGA lowering."
        ),
    ),
    SCNIRCompatibilityRow(
        nir_primitive="Threshold",
        support_level="metadata_and_hdl",
        parser_node="SCThresholdNode",
        neuron_graph_lowering="source-side or post-weight threshold transform on weighted connections",
        scnir_stream_metadata=("signal_kind=weight", "threshold_transform"),
        source_metadata=("seed", "bitstream_length", "precision"),
        hdl_support="fixed-point comparator before weighted event contribution or destination current",
        audit_evidence=("tests/test_scnir_convert.py", "tests/test_scnir_fpga_integration.py"),
        limitation=(
            "Threshold is hardware-closed only when adjacent to an Affine/Linear "
            "connection with scalar or exact-width threshold vectors; multiple "
            "thresholds on one side or incompatible widths fail closed."
        ),
    ),
    SCNIRCompatibilityRow(
        nir_primitive="Flatten",
        support_level="metadata_and_hdl",
        parser_node="SCFlattenNode",
        neuron_graph_lowering="shape-preserving pass-through into adjacent weighted connections",
        scnir_stream_metadata=("signal_kind=weight", "shape_preserving_flatten"),
        source_metadata=("seed", "bitstream_length", "precision"),
        hdl_support="folded fixed-point weight indexing with exact flattened width checks",
        audit_evidence=("tests/test_scnir_convert.py", "tests/test_scnir_fpga_integration.py"),
        limitation=(
            "Flatten is hardware-closed only when NIR shape metadata proves "
            "the flattened element count exactly matches adjacent weight and "
            "destination widths; unknown or incompatible shapes fail closed."
        ),
    ),
    SCNIRCompatibilityRow(
        nir_primitive="Delay",
        support_level="metadata_and_hdl",
        parser_node="SCDelayNode",
        neuron_graph_lowering="scalar or per-source source-side delay on weighted population connections",
        scnir_stream_metadata=("signal_kind=weight", "delay_steps>=0 or vector[int>=0]"),
        source_metadata=("seed", "bitstream_length", "precision"),
        hdl_support="direct-interconnect source register chains with per-source delay taps",
        audit_evidence=("tests/test_scnir_fpga_integration.py", "tests/test_nir_fpga_pipeline.py"),
        limitation=(
            "Delay is hardware-closed for scalar and exact source-width vector delays "
            "feeding Affine/Linear population connections; invalid vector widths fail closed."
        ),
    ),
    SCNIRCompatibilityRow(
        nir_primitive="SumPool2d",
        support_level="metadata_and_hdl",
        parser_node="SCSumPool2dNode",
        neuron_graph_lowering="shape-known pooling lowered to a dense weighted connection",
        scnir_stream_metadata=("signal_kind=weight", "pool2d_lowered_weight"),
        source_metadata=("lfsr16", "sobol16", "seed", "bitstream_length", "precision"),
        hdl_support="dense pooling fixed-point MAC terms through the weight-stream path",
        audit_evidence=("tests/test_scnir_convert.py", "tests/test_scnir_fpga_integration.py"),
        limitation=(
            "SumPool2d is hardware-closed only when NIR provides exact CHW "
            "input/output shape metadata and positive kernel/stride geometry."
        ),
    ),
    SCNIRCompatibilityRow(
        nir_primitive="AvgPool2d",
        support_level="metadata_and_hdl",
        parser_node="SCAvgPool2dNode",
        neuron_graph_lowering="shape-known average pooling lowered to scaled dense weights",
        scnir_stream_metadata=("signal_kind=weight", "pool2d_lowered_weight"),
        source_metadata=("lfsr16", "sobol16", "seed", "bitstream_length", "precision"),
        hdl_support="dense pooling fixed-point MAC terms scaled by kernel area",
        audit_evidence=("tests/test_scnir_convert.py", "tests/test_scnir_fpga_integration.py"),
        limitation=(
            "AvgPool2d is hardware-closed only when NIR provides exact CHW "
            "input/output shape metadata and positive kernel/stride geometry."
        ),
    ),
    SCNIRCompatibilityRow(
        nir_primitive="Conv1d",
        support_level="metadata_and_hdl",
        parser_node="SCConv1dNode",
        neuron_graph_lowering="shape-known convolution lowered to a dense weighted connection",
        scnir_stream_metadata=("signal_kind=weight", "convolution_lowered_weight"),
        source_metadata=("lfsr16", "sobol16", "seed", "bitstream_length", "precision"),
        hdl_support="dense Toeplitz-style fixed-point MAC terms through the weight-stream path",
        audit_evidence=("tests/test_scnir_convert.py", "tests/test_scnir_fpga_integration.py"),
        limitation=(
            "Conv1d is hardware-closed only when NIR provides finite integer "
            "input_shape, numeric padding, positive stride/dilation/groups, "
            "and a destination width matching the flattened convolution output."
        ),
    ),
    SCNIRCompatibilityRow(
        nir_primitive="Conv2d",
        support_level="metadata_and_hdl",
        parser_node="SCConv2dNode",
        neuron_graph_lowering="shape-known 2D convolution lowered to a dense weighted connection",
        scnir_stream_metadata=("signal_kind=weight", "convolution_lowered_weight"),
        source_metadata=("lfsr16", "sobol16", "seed", "bitstream_length", "precision"),
        hdl_support="dense 2D convolution fixed-point MAC terms through the weight-stream path",
        audit_evidence=("tests/test_scnir_convert.py", "tests/test_scnir_fpga_integration.py"),
        limitation=(
            "Conv2d is hardware-closed only when NIR provides finite integer "
            "input_shape, numeric padding, positive stride/dilation/groups, "
            "and a destination width matching the flattened convolution output."
        ),
    ),
    SCNIRCompatibilityRow(
        nir_primitive="NIRGraph",
        support_level="metadata_and_hdl",
        parser_node="SCSubgraphNode or SCMultiPortSubgraphNode",
        neuron_graph_lowering=(
            "single-port and exactly boundary-mapped multi-port subgraphs are "
            "inlined with namespaced nodes, including exact multi-output mappings"
        ),
        scnir_stream_metadata=(
            "inline_single_port_subgraph",
            "inline_exact_multiport_subgraph",
            "namespaced_stream_ids",
            "hierarchy_instance_metadata",
        ),
        source_metadata=(
            "lfsr16",
            "sobol16",
            "seed",
            "bitstream_length",
            "precision",
            "manifest_hierarchy_counts",
            "manifest_external_input_layout",
            "hierarchy_boundary_hdl_modules",
            "top_module_hierarchy_contract_instances",
            "scalar_hierarchy_weight_outputs",
            "packed_vector_matrix_hierarchy_weight_outputs",
        ),
        hdl_support=(
            "namespaced inline fixed-point terms with stable external input-bus "
            "lanes for single-port and exact multi-port boundary mappings, plus "
            "standalone hierarchy boundary module artefacts, top-level contract "
            "instances, and packed hierarchy weight outputs consumed by the top-level MAC"
        ),
        audit_evidence=(
            "tests/test_scnir_convert.py",
            "tests/test_scnir_fpga_integration.py",
            "tests/test_cli.py",
            "tests/test_scnir_handoff_audit.py",
        ),
        limitation=(
            "Single-port and exact one-edge-per-port nested graphs, including exact "
            "multi-output boundaries, are inlined into the parent hardware graph. "
            "Ambiguous multi-port nested NIRGraph boundary mappings still fail closed."
        ),
    ),
)


def scnir_compatibility_matrix() -> tuple[SCNIRCompatibilityRow, ...]:
    """Return the deterministic SC-NIR compatibility matrix."""
    return _MATRIX


def scnir_compatibility_matrix_dicts() -> tuple[dict[str, object], ...]:
    """Return the matrix as deterministic JSON-ready dictionaries."""
    return tuple(row.as_dict() for row in _MATRIX)


def build_scnir_compatibility_audit(evidence_root: str | Path) -> dict[str, object]:
    """Build a versioned closure-audit report for the SC-NIR compatibility matrix.

    The report is intentionally derived from the executable matrix after
    validation, so release automation consumes the same data that enforces
    parser coverage and evidence-path freshness.
    """
    root = Path(evidence_root).resolve()
    validate_scnir_compatibility_matrix(evidence_root=root)

    support_level_counts: dict[str, int] = {}
    for row in _MATRIX:
        support_level_counts[row.support_level] = support_level_counts.get(row.support_level, 0) + 1
    parser_only_primitives = sorted(
        row.nir_primitive for row in _MATRIX if row.support_level == "parser_only"
    )
    metadata_only_primitives = sorted(
        row.nir_primitive for row in _MATRIX if row.support_level == "metadata_only"
    )
    boundary_primitives = sorted(
        row.nir_primitive for row in _MATRIX if row.support_level == "boundary"
    )
    closed_handoff_primitives = sorted(
        row.nir_primitive for row in _MATRIX if row.support_level == "metadata_and_hdl"
    )
    closure_blocker_count = len(parser_only_primitives) + len(metadata_only_primitives)
    evidence_paths = sorted({path for row in _MATRIX for path in row.audit_evidence})
    evidence_files = []
    for path in evidence_paths:
        evidence_file = root / path
        evidence_bytes = evidence_file.read_bytes()
        evidence_files.append(
            {
                "path": path,
                "sha256": hashlib.sha256(evidence_bytes).hexdigest(),
                "size_bytes": len(evidence_bytes),
            }
        )
    matrix = list(scnir_compatibility_matrix_dicts())
    matrix_bytes = (json.dumps(matrix, sort_keys=True, separators=(",", ":")) + "\n").encode(
        "utf-8"
    )

    return {
        "schema_version": SCNIR_COMPATIBILITY_AUDIT_VERSION,
        "status": "valid",
        "evidence_root": str(root),
        "primitive_count": len(_MATRIX),
        "support_level_counts": dict(sorted(support_level_counts.items())),
        "closure_status": ("closed_for_local_handoff" if closure_blocker_count == 0 else "blocked"),
        "closure_blocker_count": closure_blocker_count,
        "parser_only_primitives": parser_only_primitives,
        "metadata_only_primitives": metadata_only_primitives,
        "boundary_primitives": boundary_primitives,
        "closed_handoff_primitives": closed_handoff_primitives,
        "requires_external_hardware_evidence": True,
        "external_hardware_evidence_status": "not_claimed",
        "external_hardware_evidence_note": (
            "Local SC-NIR/HDL handoff closure does not claim Vivado, PYNQ, "
            "or physical hardware evidence."
        ),
        "audit_evidence_file_count": len(evidence_paths),
        "audit_evidence_paths": evidence_paths,
        "audit_evidence_files": evidence_files,
        "matrix_sha256": hashlib.sha256(matrix_bytes).hexdigest(),
        "matrix": matrix,
    }


def validate_scnir_compatibility_matrix(evidence_root: str | Path | None = None) -> None:
    """Fail if the matrix drifts from parser-declared support or stale evidence paths.

    Parameters
    ----------
    evidence_root:
        Optional repository root used to verify that every ``audit_evidence``
        path in the matrix resolves to an existing file.
    """
    from sc_neurocore.nir_bridge.node_map import NODE_MAP

    matrix_primitives = {row.nir_primitive for row in _MATRIX}
    parser_primitives = {primitive.__name__ for primitive in NODE_MAP}
    missing = sorted(parser_primitives - matrix_primitives)
    stale = sorted(matrix_primitives - parser_primitives - {"NIRGraph"})
    if missing:
        raise ValueError(f"SC-NIR compatibility matrix misses parser primitives: {missing}")
    if stale:
        raise ValueError(f"SC-NIR compatibility matrix contains stale primitives: {stale}")

    seen: set[str] = set()
    for row in _MATRIX:
        if row.nir_primitive in seen:
            raise ValueError(f"duplicate SC-NIR compatibility row: {row.nir_primitive}")
        seen.add(row.nir_primitive)
        if row.support_level == "metadata_and_hdl" and not row.scnir_stream_metadata:
            raise ValueError(f"{row.nir_primitive} claims HDL support without stream metadata")
        if not row.audit_evidence:
            raise ValueError(f"{row.nir_primitive} has no audit evidence pointer")
        if evidence_root is not None:
            root = Path(evidence_root)
            missing_evidence = [
                evidence_path
                for evidence_path in row.audit_evidence
                if not (root / evidence_path).is_file()
            ]
            if missing_evidence:
                raise ValueError(
                    f"{row.nir_primitive} has missing audit evidence paths: {missing_evidence}"
                )
