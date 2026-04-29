# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for NIR silicon mapping reports

"""Tests for deterministic NIR silicon mapping reports."""

from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

from sc_neurocore.nir_bridge import (
    SiliconMappingConfig,
    build_silicon_mapping_report,
    write_silicon_mapping_report,
)
from sc_neurocore.nir_bridge.silicon_mapping import SCHEMA_VERSION


def _network() -> SimpleNamespace:
    return SimpleNamespace(
        nodes={
            "input": {"node_type": "Input", "shape": (4,)},
            "dense": {"node_type": "Linear", "weight": (8, 4)},
            "lif": {"node_type": "LIF", "n_neurons": 8},
            "output": {"node_type": "Output", "shape": (8,)},
        },
        topo_order=["input", "dense", "lif", "output"],
        edges=[("input", "dense"), ("dense", "lif"), ("lif", "output")],
    )


def test_silicon_mapping_report_defaults_to_loihi_spinnaker_and_akida() -> None:
    report = build_silicon_mapping_report(_network())

    assert report["schema_version"] == SCHEMA_VERSION
    assert report["source"] == {
        "node_count": 4,
        "edge_count": 3,
        "topological_order": ["input", "dense", "lif", "output"],
    }
    assert [target["target_id"] for target in report["targets"]] == [
        "loihi2",
        "spinnaker2",
        "akida",
    ]
    for target in report["targets"]:
        assert target["lowering_status"] == "clean"
        assert target["summary"]["native_nodes"] == 4
        assert target["summary"]["estimated_neurons"] == 28
        assert target["summary"]["estimated_synapses"] == 32
        assert target["summary"]["routing_edges"] == 3
        assert target["fallback_requirements"] == []
        assert target["noise_back_annotation_hooks"]


def test_akida_mapping_marks_delay_as_explicit_fallback() -> None:
    report = build_silicon_mapping_report(
        {
            "input": {"node_type": "Input", "shape": (4,)},
            "delay": {"node_type": "Delay"},
            "conv": {"node_type": "Conv2d", "weight": (8, 4, 3, 3)},
        },
        SiliconMappingConfig(targets=("akida",)),
    )

    target = report["targets"][0]
    delay_node = next(node for node in target["nodes"] if node["name"] == "delay")
    conv_node = next(node for node in target["nodes"] if node["name"] == "conv")
    assert target["target_id"] == "akida"
    assert target["lowering_status"] == "fallback_required"
    assert delay_node["lowering"] == "fallback"
    assert conv_node["lowering"] == "native"
    assert target["fallback_requirements"] == [
        {
            "node": "delay",
            "node_type": "Delay",
            "requirement": "pre-lower or host-side execute before silicon mapping",
        }
    ]


def test_silicon_mapping_report_marks_known_fallback_nodes() -> None:
    report = build_silicon_mapping_report(
        {
            "input": {"node_type": "Input", "shape": (1,)},
            "custom": {"node_type": "CustomPythonNode"},
        },
        SiliconMappingConfig(targets=("loihi2",)),
    )

    target = report["targets"][0]
    custom_node = next(node for node in target["nodes"] if node["name"] == "custom")
    assert target["lowering_status"] == "fallback_required"
    assert target["summary"]["fallback_nodes"] == 1
    assert custom_node["lowering"] == "fallback"
    assert target["fallback_requirements"] == [
        {
            "node": "custom",
            "node_type": "CustomPythonNode",
            "requirement": "pre-lower or host-side execute before silicon mapping",
        }
    ]


def test_silicon_mapping_report_marks_unknown_nodes_unsupported() -> None:
    report = build_silicon_mapping_report(
        {"threshold": {"node_type": "Threshold"}},
        SiliconMappingConfig(targets=("spinnaker2",)),
    )

    target = report["targets"][0]
    assert target["lowering_status"] == "unsupported"
    assert target["summary"]["unsupported_nodes"] == 1
    assert target["nodes"][0]["diagnostics"] == ["node type is not listed in the target manifest"]


def test_silicon_mapping_report_records_bitstream_resampling_requirement() -> None:
    report = build_silicon_mapping_report(
        _network(),
        SiliconMappingConfig(targets=("spinnaker2",), bitstream_length=1024),
    )

    target = report["targets"][0]
    assert target["lowering_status"] == "fallback_required"
    assert target["summary"]["native_bitstream_length"] is False
    assert target["fallback_requirements"] == [
        {
            "node": "*",
            "node_type": "SCBitstream",
            "requirement": "resample stochastic bitstreams to a target-supported length",
        }
    ]


def test_silicon_mapping_report_embeds_valid_noise_annotation() -> None:
    report = build_silicon_mapping_report(
        _network(),
        SiliconMappingConfig(
            targets=("loihi2",),
            noise_observations={"loihi2": {"spike_drop_rate": 0.001}},
        ),
    )

    assert report["targets"][0]["noise_annotation"]["observations"] == {"spike_drop_rate": 0.001}


@pytest.mark.parametrize(
    "config_kwargs, message",
    [
        ({"targets": ()}, "targets must not be empty"),
        ({"bitstream_length": 0}, "bitstream_length must be positive"),
        ({"event_rate_hz": math.inf}, "event_rate_hz must be finite and positive"),
        ({"targets": ("unknown",)}, "unknown neuromorphic target"),
    ],
)
def test_silicon_mapping_config_rejects_invalid_inputs(
    config_kwargs: dict[str, object], message: str
) -> None:
    with pytest.raises((KeyError, ValueError), match=message):
        SiliconMappingConfig(**config_kwargs)


def test_write_silicon_mapping_report_writes_sorted_json(tmp_path: Path) -> None:
    path = write_silicon_mapping_report(
        tmp_path,
        _network(),
        SiliconMappingConfig(targets=("loihi2",)),
    )

    assert path == tmp_path / "nir_silicon_mapping_report.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["targets"][0]["target_id"] == "loihi2"
    assert path.read_text(encoding="utf-8").endswith("\n")
