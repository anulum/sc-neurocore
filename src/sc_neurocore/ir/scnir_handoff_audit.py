# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — SC-NIR HDL handoff audit

"""Executable audit checks for ``compile-nir`` SC-NIR HDL handoff artefacts."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence, cast

from .scnir_schema import SCNIRDocument, SCNIRStream, load_scnir, scnir_to_dict

SCNIR_HDL_HANDOFF_MANIFEST_VERSION = "sc-neurocore.scnir.hdl-sources.v0.2"


class SCNIRHDLHandoffAuditError(ValueError):
    """Raised when a compile-nir HDL handoff directory is incomplete or inconsistent."""


@dataclass(frozen=True, slots=True)
class SCNIRHDLHandoffAuditReport:
    """Deterministic summary of a validated SC-NIR HDL handoff directory."""

    directory: str
    module_name: str
    schema_version: str
    manifest_schema_version: str
    bitstream_length: int
    interconnect: str
    q_format: str
    stream_count: int
    source_module_count: int
    hierarchy_instance_count: int
    hierarchy_port_count: int
    hierarchy_instances: dict[str, dict[str, object]]
    external_input_count: int
    external_inputs: tuple[dict[str, int | str], ...]
    total_neurons: int
    total_synapses: int
    signal_kinds: dict[str, int]
    signal_routes: dict[str, str]
    artefacts: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        """Return a stable JSON-ready report."""

        return {
            "status": "valid",
            "directory": self.directory,
            "module_name": self.module_name,
            "schema_version": self.schema_version,
            "manifest_schema_version": self.manifest_schema_version,
            "bitstream_length": self.bitstream_length,
            "interconnect": self.interconnect,
            "q_format": self.q_format,
            "stream_count": self.stream_count,
            "source_module_count": self.source_module_count,
            "hierarchy_instance_count": self.hierarchy_instance_count,
            "hierarchy_port_count": self.hierarchy_port_count,
            "hierarchy_instances": self.hierarchy_instances,
            "external_input_count": self.external_input_count,
            "external_inputs": [dict(row) for row in self.external_inputs],
            "total_neurons": self.total_neurons,
            "total_synapses": self.total_synapses,
            "signal_kinds": self.signal_kinds,
            "signal_routes": self.signal_routes,
            "artefacts": list(self.artefacts),
        }


def audit_scnir_hdl_handoff(directory: str | Path) -> SCNIRHDLHandoffAuditReport:
    """Validate a ``compile-nir`` HDL output directory and return an audit report.

    The audit is intentionally structural and fail-closed: every SC-NIR stream
    must have exactly one matching source-manifest row and emitted source
    module, aggregate counts must match the typed document, and top-level
    SC-NIR localparams must agree with the JSON handoff metadata.
    """

    root = Path(directory)
    if not root.is_dir():
        raise SCNIRHDLHandoffAuditError(f"handoff directory does not exist: {root}")

    document_path = root / "scnir_document.json"
    manifest_path = root / "scnir_source_manifest.json"
    _require_file(document_path, "SC-NIR document")
    _require_file(manifest_path, "SC-NIR source manifest")

    try:
        document = load_scnir(document_path)
    except Exception as exc:
        raise SCNIRHDLHandoffAuditError(f"invalid scnir_document.json: {exc}") from exc

    manifest = _load_manifest(manifest_path)
    _verify_manifest_header(manifest)

    module_name = _expect_non_empty_string(manifest, "module_name")
    top_module_path = root / f"{module_name}.v"
    _require_file(top_module_path, "top module")
    _require_file(root / "sc_nir_weight_rom.v", "weight ROM")

    streams = tuple(document.streams)
    sources = _expect_mapping_sequence(manifest, "sources")
    stream_count = _expect_int(manifest, "scnir_stream_count")
    if stream_count != len(streams):
        raise SCNIRHDLHandoffAuditError(
            f"scnir_stream_count {stream_count} does not match document stream count {len(streams)}"
        )
    if len(sources) != len(streams):
        raise SCNIRHDLHandoffAuditError(
            f"sources length {len(sources)} does not match document stream count {len(streams)}"
        )

    bitstream_length = _expect_int(manifest, "bitstream_length")
    source_modules = _verify_source_rows(root, document, sources)
    signal_kinds = _signal_kind_counts(document)
    signal_routes = _signal_routes(
        document, interconnect=_expect_non_empty_string(manifest, "interconnect")
    )
    hierarchy_instances = _hierarchy_instances(document)
    hierarchy_port_count = sum(
        len(cast(list[object], instance["ports"])) for instance in hierarchy_instances.values()
    )
    _expect_equal(manifest.get("scnir_signal_kinds"), signal_kinds, "scnir_signal_kinds")
    _expect_equal(manifest.get("scnir_signal_routes"), signal_routes, "scnir_signal_routes")
    _expect_equal(
        manifest.get("scnir_hierarchy_instance_count"),
        len(hierarchy_instances),
        "scnir_hierarchy_instance_count",
    )
    _expect_equal(
        manifest.get("scnir_hierarchy_port_count"),
        hierarchy_port_count,
        "scnir_hierarchy_port_count",
    )
    hierarchy_modules = _verify_hierarchy_modules(root, document)
    top_module = top_module_path.read_text(encoding="utf-8")
    _verify_hierarchy_top_instances(top_module, document)
    external_inputs = _external_inputs(manifest)
    _expect_top_localparam(top_module, "SCNIR_BITSTREAM_LENGTH", bitstream_length)
    _expect_top_localparam(top_module, "SCNIR_STREAM_COUNT", len(streams))
    _expect_top_localparam(top_module, "SCNIR_SOURCE_MODULE_COUNT", len(sources))

    artefacts = tuple(
        sorted(
            {
                "scnir_document.json",
                "scnir_source_manifest.json",
                "sc_nir_weight_rom.v",
                f"{module_name}.v",
                *source_modules,
                *hierarchy_modules,
            }
        )
    )
    return SCNIRHDLHandoffAuditReport(
        directory=str(root),
        module_name=module_name,
        schema_version=document.schema_version,
        manifest_schema_version=cast(str, manifest["schema_version"]),
        bitstream_length=bitstream_length,
        interconnect=cast(str, manifest["interconnect"]),
        q_format=_expect_non_empty_string(manifest, "q_format"),
        stream_count=len(streams),
        source_module_count=len(sources),
        hierarchy_instance_count=len(hierarchy_instances),
        hierarchy_port_count=hierarchy_port_count,
        hierarchy_instances=hierarchy_instances,
        external_input_count=len(external_inputs),
        external_inputs=external_inputs,
        total_neurons=_expect_int(manifest, "total_neurons"),
        total_synapses=_expect_int(manifest, "total_synapses"),
        signal_kinds=signal_kinds,
        signal_routes=signal_routes,
        artefacts=artefacts,
    )


def write_scnir_hdl_handoff_audit(
    directory: str | Path,
    output_path: str | Path,
) -> SCNIRHDLHandoffAuditReport:
    """Validate a handoff directory and write the JSON audit report."""

    report = audit_scnir_hdl_handoff(directory)
    Path(output_path).write_text(
        json.dumps(report.as_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def _load_manifest(path: Path) -> Mapping[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise SCNIRHDLHandoffAuditError(f"invalid scnir_source_manifest.json: {exc}") from exc
    if not isinstance(raw, Mapping):
        raise SCNIRHDLHandoffAuditError("scnir_source_manifest.json must be a JSON object")
    return raw


def _verify_manifest_header(manifest: Mapping[str, Any]) -> None:
    expected_keys = {
        "schema_version",
        "module_name",
        "bitstream_length",
        "source_kind",
        "interconnect",
        "q_format",
        "total_neurons",
        "total_synapses",
        "scnir_stream_count",
        "scnir_signal_kinds",
        "scnir_signal_routes",
        "scnir_external_inputs",
        "scnir_hierarchy_instance_count",
        "scnir_hierarchy_port_count",
        "sources",
    }
    actual_keys = set(manifest)
    if actual_keys != expected_keys:
        unknown = sorted(actual_keys - expected_keys)
        missing = sorted(expected_keys - actual_keys)
        raise SCNIRHDLHandoffAuditError(
            f"source manifest keys mismatch: missing={missing}, unknown={unknown}"
        )
    if manifest["schema_version"] != SCNIR_HDL_HANDOFF_MANIFEST_VERSION:
        raise SCNIRHDLHandoffAuditError(
            f"source manifest schema_version must be {SCNIR_HDL_HANDOFF_MANIFEST_VERSION!r}"
        )
    _expect_positive_int(manifest, "bitstream_length")
    _expect_non_empty_string(manifest, "source_kind")
    _expect_non_empty_string(manifest, "interconnect")
    _expect_non_empty_string(manifest, "q_format")
    _expect_non_negative_int(manifest, "total_neurons")
    _expect_non_negative_int(manifest, "total_synapses")
    _expect_non_negative_int(manifest, "scnir_stream_count")
    _expect_non_negative_int(manifest, "scnir_hierarchy_instance_count")
    _expect_non_negative_int(manifest, "scnir_hierarchy_port_count")


def _external_inputs(manifest: Mapping[str, Any]) -> tuple[dict[str, int | str], ...]:
    rows = _expect_mapping_sequence(manifest, "scnir_external_inputs")
    result: list[dict[str, int | str]] = []
    seen: set[str] = set()
    cursor = 0
    for index, row in enumerate(rows):
        expected = {"source", "offset", "width"}
        actual = set(row)
        if actual != expected:
            raise SCNIRHDLHandoffAuditError(
                f"scnir_external_inputs[{index}] keys mismatch: "
                f"missing={sorted(expected - actual)}, unknown={sorted(actual - expected)}"
            )
        source = _expect_non_empty_string(row, "source")
        if source in seen:
            raise SCNIRHDLHandoffAuditError(f"scnir_external_inputs duplicate source {source!r}")
        seen.add(source)
        offset = _expect_int(row, "offset")
        width = _expect_int(row, "width")
        if offset != cursor:
            raise SCNIRHDLHandoffAuditError(
                f"scnir_external_inputs[{index}].offset {offset} does not match "
                f"contiguous offset {cursor}"
            )
        if width <= 0:
            raise SCNIRHDLHandoffAuditError(
                f"scnir_external_inputs[{index}].width must be positive"
            )
        result.append({"source": source, "offset": offset, "width": width})
        cursor += width
    return tuple(result)


def _verify_source_rows(
    root: Path,
    document: SCNIRDocument,
    sources: Sequence[Mapping[str, Any]],
) -> tuple[str, ...]:
    streams_by_id = {stream.stream_id: stream for stream in document.streams}
    rows_by_id: dict[str, Mapping[str, Any]] = {}
    module_files: list[str] = []
    for index, row in enumerate(sources):
        _verify_source_row_keys(row, index)
        stream_id = _expect_non_empty_string(row, "stream_id")
        if stream_id in rows_by_id:
            raise SCNIRHDLHandoffAuditError(f"duplicate source row for stream_id {stream_id!r}")
        if stream_id not in streams_by_id:
            raise SCNIRHDLHandoffAuditError(
                f"source row references unknown stream_id {stream_id!r}"
            )
        rows_by_id[stream_id] = row
        stream = streams_by_id[stream_id]
        _verify_source_row_matches_stream(row, stream, index)
        module_name = _expect_non_empty_string(row, "module_name")
        source_file = root / f"{module_name}.v"
        _require_file(source_file, f"source module file for {stream_id}")
        module_files.append(source_file.name)

    # Every stream is covered: the caller guarantees len(sources) == len(streams),
    # and the duplicate/unknown guards above make the row→stream mapping a bijection.
    return tuple(module_files)


def _verify_hierarchy_modules(root: Path, document: SCNIRDocument) -> tuple[str, ...]:
    module_files: list[str] = []
    for instance in document.hierarchy:
        module_name = instance.module_name
        module_file = f"{module_name}.v"
        module_path = root / module_file
        _require_file(module_path, f"hierarchy module file {module_file}")
        module_text = module_path.read_text(encoding="utf-8")
        if f"module {module_name}" not in module_text:
            raise SCNIRHDLHandoffAuditError(
                f"hierarchy module file {module_file} does not declare module {module_name!r}"
            )
        for port in instance.ports:
            if port.port_name not in module_text:
                raise SCNIRHDLHandoffAuditError(
                    f"hierarchy module file {module_file} is missing port {port.port_name!r}"
                )
        module_files.append(module_file)
    return tuple(module_files)


def _verify_hierarchy_top_instances(verilog: str, document: SCNIRDocument) -> None:
    for instance in document.hierarchy:
        module_name = instance.module_name
        instance_name = f"{module_name}_hierarchy_inst"
        if f"{module_name} {instance_name}" not in verilog:
            raise SCNIRHDLHandoffAuditError(
                f"top module missing hierarchy instance {instance_name!r} for {module_name!r}"
            )
        for port in instance.ports:
            if f".{port.port_name}(" not in verilog:
                raise SCNIRHDLHandoffAuditError(
                    f"top module hierarchy instance {instance_name!r} missing port "
                    f"{port.port_name!r}"
                )


def _verify_source_row_keys(row: Mapping[str, Any], index: int) -> None:
    expected = {
        "stream_id",
        "layer",
        "module_name",
        "source_kind",
        "seed",
        "bitstream_length",
        "encoding",
        "signal_kind",
        "delay_steps",
        "total_bits",
        "fractional_bits",
        "transforms",
        "online_learning",
        "lfsr_polynomial",
        "tap_mask",
        "sobol_dimension",
    }
    actual = set(row)
    if actual != expected:
        raise SCNIRHDLHandoffAuditError(
            f"sources[{index}] keys mismatch: "
            f"missing={sorted(expected - actual)}, unknown={sorted(actual - expected)}"
        )


def _verify_source_row_matches_stream(
    row: Mapping[str, Any],
    stream: SCNIRStream,
    index: int,
) -> None:
    expected = {
        "layer": stream.layer,
        "bitstream_length": stream.bitstream_length,
        "encoding": stream.encoding,
        "signal_kind": stream.signal_kind,
        "delay_steps": _delay_steps_for_row(stream.delay_steps),
        "total_bits": stream.precision.total_bits,
        "fractional_bits": stream.precision.fractional_bits,
        "source_kind": f"{stream.source.kind}16",
        "transforms": _stream_transform_rows(stream),
        "online_learning": dict(stream.online_learning)
        if stream.online_learning is not None
        else None,
    }
    source = stream.source
    if source.seed is not None:
        expected["seed"] = source.seed & 0xFFFF
    if source.lfsr_polynomial is not None:
        expected["lfsr_polynomial"] = source.lfsr_polynomial
    if source.tap_mask is not None:
        expected["tap_mask"] = source.tap_mask
    if source.sobol_dimension is not None:
        expected["sobol_dimension"] = source.sobol_dimension
    for key, value in expected.items():
        if row[key] != value:
            raise SCNIRHDLHandoffAuditError(
                f"sources[{index}].{key} {row[key]!r} does not match stream "
                f"{stream.stream_id!r} value {value!r}"
            )


def _stream_transform_rows(stream: SCNIRStream) -> list[dict[str, object]]:
    return [
        {
            "kind": transform.kind,
            "position": transform.position,
            "comparison": transform.comparison,
            "values": [float(value) for value in transform.values],
        }
        for transform in stream.transforms
    ]


def _delay_steps_for_row(delay_steps: int | Sequence[int]) -> int | list[int]:
    if isinstance(delay_steps, int):
        return delay_steps
    return [int(value) for value in delay_steps]


def _signal_kind_counts(document: SCNIRDocument) -> dict[str, int]:
    scnir_to_dict(document)
    counts: dict[str, int] = {}
    for stream in document.streams:
        counts[stream.signal_kind] = counts.get(stream.signal_kind, 0) + 1
    return dict(sorted(counts.items()))


def _signal_routes(document: SCNIRDocument, *, interconnect: str) -> dict[str, str]:
    present_kinds = {stream.signal_kind for stream in document.streams}
    routes = {
        "analogue_state": "direct_mac",
        "spike": "weighted_event_aer" if interconnect == "aer" else "direct_wire",
        "weight": "stochastic_source_module",
    }
    return {kind: routes[kind] for kind in routes if kind in present_kinds}


def _hierarchy_instances(document: SCNIRDocument) -> dict[str, dict[str, object]]:
    scnir_to_dict(document)
    rows: dict[str, dict[str, object]] = {}
    for instance in sorted(document.hierarchy, key=lambda item: item.instance_id):
        rows[instance.instance_id] = {
            "module_name": instance.module_name,
            "ports": [
                {
                    "port_name": port.port_name,
                    "direction": port.direction,
                    "stream_id": port.stream_id,
                    "signal_kind": port.signal_kind,
                    "bit_width": port.bit_width,
                }
                for port in instance.ports
            ],
        }
    return rows


def _expect_top_localparam(verilog: str, name: str, value: int) -> None:
    needle = f"localparam integer {name} = {value};"
    if needle not in verilog:
        raise SCNIRHDLHandoffAuditError(f"top module missing {needle!r}")


def _expect_equal(actual: object, expected: object, field: str) -> None:
    if actual != expected:
        raise SCNIRHDLHandoffAuditError(
            f"{field} {actual!r} does not match expected value {expected!r}"
        )


def _require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise SCNIRHDLHandoffAuditError(f"missing {label}: {path}")


def _expect_mapping_sequence(
    manifest: Mapping[str, Any], key: str
) -> tuple[Mapping[str, Any], ...]:
    value = manifest[key]
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise SCNIRHDLHandoffAuditError(f"{key} must be a sequence")
    rows: list[Mapping[str, Any]] = []
    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            raise SCNIRHDLHandoffAuditError(f"{key}[{index}] must be a JSON object")
        rows.append(item)
    return tuple(rows)


def _expect_non_empty_string(mapping: Mapping[str, Any], key: str) -> str:
    value = mapping[key]
    if not isinstance(value, str) or not value:
        raise SCNIRHDLHandoffAuditError(f"{key} must be a non-empty string")
    return value


def _expect_int(mapping: Mapping[str, Any], key: str) -> int:
    value = mapping[key]
    if not isinstance(value, int) or isinstance(value, bool):
        raise SCNIRHDLHandoffAuditError(f"{key} must be an integer")
    return value


def _expect_positive_int(mapping: Mapping[str, Any], key: str) -> int:
    value = _expect_int(mapping, key)
    if value <= 0:
        raise SCNIRHDLHandoffAuditError(f"{key} must be positive")
    return value


def _expect_non_negative_int(mapping: Mapping[str, Any], key: str) -> int:
    value = _expect_int(mapping, key)
    if value < 0:
        raise SCNIRHDLHandoffAuditError(f"{key} must be non-negative")
    return value
