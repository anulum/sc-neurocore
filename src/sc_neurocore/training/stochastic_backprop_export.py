# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Stochastic backpropagation SC-NIR export manifest

"""Export stochastic backpropagation training evidence into SC-NIR metadata."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, cast

from sc_neurocore.ir.scnir_handoff_audit import (
    SCNIR_HDL_HANDOFF_MANIFEST_VERSION,
    SCNIRHDLHandoffAuditReport,
    audit_scnir_hdl_handoff,
)
from sc_neurocore.ir.scnir_schema import SCNIR_SCHEMA_VERSION, validate_scnir_dict

from .sc_estimators import DifferentiableSCConfig

STOCHASTIC_BACKPROP_EXPORT_SCHEMA_VERSION = "sc-neurocore.stochastic-backprop-export.v1"
_VERILOG_HEADER = "\n".join(
    [
        "// SPDX-License-Identifier: AGPL-3.0-or-later",
        "// Commercial license available",
        "// Copyright Concepts 1996-2026 Miroslav Sotek. All rights reserved.",
        "// Copyright Code 2020-2026 Miroslav Sotek. All rights reserved.",
        "// ORCID: 0009-0009-3560-0851",
        "// Contact: www.anulum.li | protoscience@anulum.li",
        "// SC-NeuroCore - Generated stochastic backpropagation SC-NIR handoff HDL",
        "",
    ]
)


def build_stochastic_backprop_export_manifest(
    benchmark_report: Mapping[str, Any],
    sc_config: DifferentiableSCConfig,
    *,
    producer: str = "sc-neurocore.training.stochastic_backprop_export",
    replay_uri: str = "benchmarks/results/stochastic_backprop_benchmark.json",
) -> dict[str, Any]:
    """Build a deterministic SC-NIR handoff manifest for stochastic backpropagation."""

    _validate_benchmark_matches_config(benchmark_report, sc_config)
    scnir_document = _build_scnir_document(sc_config, producer=producer, replay_uri=replay_uri)
    validate_scnir_dict(scnir_document)

    stream_evidence = _mapping(benchmark_report["stream_evidence"], "stream_evidence")
    loss = _mapping(benchmark_report["loss"], "loss")
    objective_terms = _mapping(benchmark_report["objective_terms"], "objective_terms")
    training = _mapping(benchmark_report["training"], "training")

    manifest = {
        "schema_version": STOCHASTIC_BACKPROP_EXPORT_SCHEMA_VERSION,
        "scnir_schema_version": SCNIR_SCHEMA_VERSION,
        "training": {
            "bitstream_length": sc_config.bitstream_length,
            "encoding": sc_config.encoding,
            "generator": sc_config.generator,
            "estimator": sc_config.estimator,
            "input_seed": sc_config.input_seed,
            "weight_seed": sc_config.weight_seed,
            "correlation": sc_config.correlation,
            "steps": training["steps"],
            "learning_rate": training["learning_rate"],
        },
        "evidence": {
            "benchmark_schema_version": benchmark_report["schema_version"],
            "evidence_class": benchmark_report["evidence_class"],
            "hardware_measurement_claimed": benchmark_report["hardware_measurement_claimed"],
            "initial_loss": loss["initial"],
            "final_loss": loss["final"],
            "sampled_product_mae": stream_evidence["sampled_product_mae"],
            "input_max_abs_correlation": stream_evidence["input_max_abs_correlation"],
            "weight_max_abs_correlation": stream_evidence["weight_max_abs_correlation"],
            "objective_total": objective_terms["total"],
        },
        "scnir_document": scnir_document,
    }
    _validate_manifest(manifest)
    return manifest


def write_stochastic_backprop_export_manifest(
    path: str | Path,
    benchmark_report: Mapping[str, Any],
    sc_config: DifferentiableSCConfig,
    *,
    producer: str = "sc-neurocore.training.stochastic_backprop_export",
    replay_uri: str = "benchmarks/results/stochastic_backprop_benchmark.json",
) -> Path:
    """Write a canonical stochastic backpropagation SC-NIR export manifest."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = build_stochastic_backprop_export_manifest(
        benchmark_report,
        sc_config,
        producer=producer,
        replay_uri=replay_uri,
    )
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def write_stochastic_backprop_handoff_bundle(
    export_manifest: Mapping[str, Any],
    output_dir: str | Path,
    *,
    module_name: str = "stochastic_backprop_handoff",
) -> SCNIRHDLHandoffAuditReport:
    """Materialise an auditable SC-NIR HDL handoff bundle from an export manifest."""

    _validate_manifest(export_manifest)
    scnir_document = _mapping(export_manifest["scnir_document"], "scnir_document")
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    (root / "scnir_document.json").write_text(
        json.dumps(scnir_document, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    source_rows = _source_rows_from_scnir(scnir_document)
    bitstream_length = _shared_bitstream_length(scnir_document)
    source_manifest = {
        "schema_version": SCNIR_HDL_HANDOFF_MANIFEST_VERSION,
        "module_name": module_name,
        "bitstream_length": bitstream_length,
        "source_kind": "stochastic_backprop_export",
        "interconnect": "direct",
        "q_format": "Q1.15",
        "total_neurons": 0,
        "total_synapses": 1,
        "scnir_stream_count": len(source_rows),
        "scnir_signal_kinds": _signal_kind_counts(scnir_document),
        "scnir_signal_routes": {
            "spike": "direct_wire",
            "weight": "stochastic_source_module",
        },
        "scnir_external_inputs": [],
        "scnir_hierarchy_instance_count": 0,
        "scnir_hierarchy_port_count": 0,
        "sources": source_rows,
    }
    (root / "scnir_source_manifest.json").write_text(
        json.dumps(source_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (root / "sc_nir_weight_rom.v").write_text(
        f"{_VERILOG_HEADER}module sc_nir_weight_rom; endmodule\n",
        encoding="utf-8",
    )
    _write_top_module(
        root / f"{module_name}.v",
        module_name=module_name,
        bitstream_length=bitstream_length,
        stream_count=len(source_rows),
    )
    for row in source_rows:
        source_module = cast(str, row["module_name"])
        _write_source_module(root / f"{source_module}.v", module_name=source_module)
    return audit_scnir_hdl_handoff(root)


def _validate_benchmark_matches_config(
    benchmark_report: Mapping[str, Any],
    sc_config: DifferentiableSCConfig,
) -> None:
    for key in (
        "schema_version",
        "evidence_class",
        "hardware_measurement_claimed",
        "sc_config",
        "training",
        "loss",
        "objective_terms",
        "stream_evidence",
    ):
        if key not in benchmark_report:
            raise ValueError(f"benchmark_report missing {key}")
    if benchmark_report["hardware_measurement_claimed"] is not False:
        raise ValueError("benchmark_report must not claim hardware measurement")

    report_config = _mapping(benchmark_report["sc_config"], "sc_config")
    expected = {
        "bitstream_length": sc_config.bitstream_length,
        "encoding": sc_config.encoding,
        "generator": sc_config.generator,
        "estimator": sc_config.estimator,
        "input_seed": sc_config.input_seed,
        "weight_seed": sc_config.weight_seed,
        "correlation": sc_config.correlation,
    }
    for key, expected_value in expected.items():
        if report_config.get(key) != expected_value:
            raise ValueError(
                f"benchmark_report sc_config {key} mismatch: "
                f"expected {expected_value!r}, got {report_config.get(key)!r}"
            )


def _source_rows_from_scnir(scnir_document: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    streams = scnir_document["streams"]
    if not isinstance(streams, list):
        raise ValueError("scnir_document.streams must be a list")
    for index, stream_raw in enumerate(streams):
        stream = _mapping(stream_raw, f"streams[{index}]")
        precision = _mapping(stream["precision"], f"streams[{index}].precision")
        source = _mapping(stream["source"], f"streams[{index}].source")
        stream_id = str(stream["stream_id"])
        rows.append(
            {
                "stream_id": stream_id,
                "layer": stream["layer"],
                "module_name": _source_module_name(index, stream_id),
                "source_kind": f"{source['kind']}16",
                "seed": None if source["seed"] is None else int(source["seed"]) & 0xFFFF,
                "bitstream_length": stream["bitstream_length"],
                "encoding": stream["encoding"],
                "signal_kind": stream["signal_kind"],
                "delay_steps": stream["delay_steps"],
                "total_bits": precision["total_bits"],
                "fractional_bits": precision["fractional_bits"],
                "transforms": stream["transforms"],
                "online_learning": stream["online_learning"],
                "lfsr_polynomial": source["lfsr_polynomial"],
                "tap_mask": source["tap_mask"],
                "sobol_dimension": source["sobol_dimension"],
            }
        )
    return rows


def _source_module_name(index: int, stream_id: str) -> str:
    safe = "".join(char if char.isalnum() else "_" for char in stream_id)
    return f"scnir_src_{index:03d}_{safe}"


def _shared_bitstream_length(scnir_document: Mapping[str, Any]) -> int:
    streams = scnir_document["streams"]
    if not isinstance(streams, list) or not streams:
        raise ValueError("scnir_document.streams must contain at least one stream")
    lengths = {int(_mapping(stream, "stream")["bitstream_length"]) for stream in streams}
    if len(lengths) != 1:
        raise ValueError("stochastic backprop handoff requires one shared bitstream_length")
    return lengths.pop()


def _signal_kind_counts(scnir_document: Mapping[str, Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    streams = scnir_document["streams"]
    if not isinstance(streams, list):
        raise ValueError("scnir_document.streams must be a list")
    for stream_raw in streams:
        stream = _mapping(stream_raw, "stream")
        signal_kind = str(stream["signal_kind"])
        counts[signal_kind] = counts.get(signal_kind, 0) + 1
    return dict(sorted(counts.items()))


def _write_top_module(
    path: Path,
    *,
    module_name: str,
    bitstream_length: int,
    stream_count: int,
) -> None:
    path.write_text(
        "\n".join(
            [
                _VERILOG_HEADER.rstrip(),
                f"module {module_name};",
                f"localparam integer SCNIR_BITSTREAM_LENGTH = {bitstream_length};",
                f"localparam integer SCNIR_STREAM_COUNT = {stream_count};",
                f"localparam integer SCNIR_SOURCE_MODULE_COUNT = {stream_count};",
                "endmodule",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _write_source_module(path: Path, *, module_name: str) -> None:
    path.write_text(
        f"{_VERILOG_HEADER}module {module_name}; endmodule\n",
        encoding="utf-8",
    )


def _build_scnir_document(
    sc_config: DifferentiableSCConfig,
    *,
    producer: str,
    replay_uri: str,
) -> dict[str, Any]:
    source_kind = _source_kind(sc_config.generator)
    input_stream = _stream(
        stream_id="stochastic_backprop.input",
        layer="stochastic_backprop_training",
        signal_kind="spike",
        bitstream_length=sc_config.bitstream_length,
        encoding=sc_config.encoding,
        source={
            **_empty_source(),
            "kind": source_kind,
            "seed": sc_config.input_seed,
            **_source_specific(source_kind, sc_config, dimension=1),
        },
        constraints=[
            _constraint(
                peer_stream_id="stochastic_backprop.weight",
                max_abs_correlation=max(0.0, abs(float(sc_config.correlation))),
            )
        ],
    )
    weight_stream = _stream(
        stream_id="stochastic_backprop.weight",
        layer="stochastic_backprop_training",
        signal_kind="weight",
        bitstream_length=sc_config.bitstream_length,
        encoding=sc_config.encoding,
        source={
            **_empty_source(),
            "kind": source_kind,
            "seed": sc_config.weight_seed,
            **_source_specific(source_kind, sc_config, dimension=2),
        },
        constraints=[
            _constraint(
                peer_stream_id="stochastic_backprop.input",
                max_abs_correlation=max(0.0, abs(float(sc_config.correlation))),
            )
        ],
    )
    product_stream = _stream(
        stream_id="stochastic_backprop.product",
        layer="stochastic_backprop_training",
        signal_kind="spike",
        bitstream_length=sc_config.bitstream_length,
        encoding=sc_config.encoding,
        source={
            **_empty_source(),
            "kind": "replay",
            "replay_uri": replay_uri,
        },
        constraints=[
            _constraint(peer_stream_id="stochastic_backprop.input", max_abs_correlation=1.0),
            _constraint(peer_stream_id="stochastic_backprop.weight", max_abs_correlation=1.0),
        ],
    )
    return {
        "schema_version": SCNIR_SCHEMA_VERSION,
        "producer": producer,
        "streams": [input_stream, weight_stream, product_stream],
        "hierarchy": [],
    }


def _stream(
    *,
    stream_id: str,
    layer: str,
    signal_kind: str,
    bitstream_length: int,
    encoding: str,
    source: Mapping[str, Any],
    constraints: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "stream_id": stream_id,
        "layer": layer,
        "bitstream_length": bitstream_length,
        "encoding": encoding,
        "signal_kind": signal_kind,
        "precision": {
            "signed": encoding == "bipolar",
            "total_bits": 16,
            "fractional_bits": 15,
            "accumulator_bits": 32,
            "rounding": "stochastic",
            "overflow": "saturate",
        },
        "source": dict(source),
        "delay_steps": 0,
        "transforms": [],
        "correlation_constraints": constraints,
        "online_learning": None,
    }


def _constraint(*, peer_stream_id: str, max_abs_correlation: float) -> dict[str, Any]:
    return {
        "peer_stream_id": peer_stream_id,
        "policy": "max_correlation",
        "max_abs_correlation": max_abs_correlation,
        "seed_domain": "stochastic_backprop_training",
    }


def _empty_source() -> dict[str, Any]:
    return {
        "seed": None,
        "lfsr_polynomial": None,
        "tap_mask": None,
        "sobol_dimension": None,
        "halton_base": None,
        "replay_uri": None,
        "hardware_id": None,
    }


def _source_kind(generator: str) -> str:
    if generator in {"sobol", "low_discrepancy"}:
        return "sobol"
    if generator == "halton":
        return "halton"
    if generator == "lfsr":
        return "lfsr"
    if generator == "bernoulli":
        return "replay"
    raise ValueError(f"unsupported generator for export: {generator}")


def _source_specific(
    source_kind: str,
    sc_config: DifferentiableSCConfig,
    *,
    dimension: int,
) -> dict[str, Any]:
    if source_kind == "sobol":
        return {"sobol_dimension": dimension}
    if source_kind == "halton":
        return {"halton_base": 2 if dimension == 1 else 3}
    if source_kind == "lfsr":
        return {
            "lfsr_polynomial": sc_config.lfsr_polynomial,
            "tap_mask": 0xB400,
        }
    if source_kind == "replay":
        return {"replay_uri": "deterministic_bernoulli_training_replay"}
    return {}


def _validate_manifest(manifest: Mapping[str, Any]) -> None:
    if manifest["schema_version"] != STOCHASTIC_BACKPROP_EXPORT_SCHEMA_VERSION:
        raise ValueError("manifest schema_version is unsupported")
    if manifest["scnir_schema_version"] != SCNIR_SCHEMA_VERSION:
        raise ValueError("manifest scnir_schema_version is unsupported")
    validate_scnir_dict(_mapping(manifest["scnir_document"], "scnir_document"))


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must be a mapping")
    return cast(Mapping[str, Any], value)
