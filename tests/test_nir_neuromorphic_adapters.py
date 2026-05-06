# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for NIR neuromorphic adapter packages

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from sc_neurocore.nir_bridge import (
    SiliconMappingConfig,
    build_neuromorphic_adapter_bundle,
    build_neuromorphic_adapter_package,
    write_neuromorphic_adapter_bundle,
)
from sc_neurocore.nir_bridge.neuromorphic_adapters import ADAPTER_SCHEMA_VERSION


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


def test_loihi2_adapter_manifest_is_honest_sdk_handoff() -> None:
    package = build_neuromorphic_adapter_package(_network(), "loihi2")
    manifest = package.manifest()

    assert manifest["schema_version"] == ADAPTER_SCHEMA_VERSION
    assert manifest["target_id"] == "loihi2"
    assert manifest["vendor_stack"] == "Intel Lava / Loihi 2"
    assert manifest["sdk_dependency"] == "lava-nc"
    assert manifest["lowering_status"] == "clean"
    assert manifest["hardware_status"].startswith("requires Lava installation")
    assert manifest["fallback_requirements"] == []
    assert manifest["noise_back_annotation_hooks"]


def test_spinnaker2_adapter_records_fallbacks_and_vendor_boundary() -> None:
    package = build_neuromorphic_adapter_package(
        {"custom": {"node_type": "CustomPythonNode"}},
        "spinnaker2",
    )
    manifest = package.manifest()

    assert manifest["target_id"] == "spinnaker2"
    assert manifest["vendor_stack"] == "SpiNNaker2 / SpiNNTools"
    assert manifest["lowering_status"] == "fallback_required"
    assert manifest["fallback_requirements"] == [
        {
            "node": "custom",
            "node_type": "CustomPythonNode",
            "requirement": "pre-lower or host-side execute before silicon mapping",
        }
    ]


def test_adapter_bundle_defaults_to_loihi2_and_spinnaker2() -> None:
    bundle = build_neuromorphic_adapter_bundle(_network())

    assert list(bundle) == ["loihi2", "spinnaker2"]
    assert bundle["loihi2"].manifest()["adapter_name"] == "Loihi 2 Lava handoff"
    assert bundle["spinnaker2"].manifest()["adapter_name"] == "SpiNNaker2 SpiNNTools handoff"


def test_adapter_rejects_targets_without_handoff_contract() -> None:
    with pytest.raises(ValueError, match="unsupported adapter target"):
        build_neuromorphic_adapter_package(_network(), "akida")


def test_adapter_preserves_noise_annotations() -> None:
    package = build_neuromorphic_adapter_package(
        _network(),
        "loihi2",
        SiliconMappingConfig(
            noise_observations={"loihi2": {"spike_drop_rate": 0.002}},
        ),
    )

    assert package.target_report["noise_annotation"]["observations"] == {"spike_drop_rate": 0.002}


def test_write_adapter_bundle_writes_manifest_report_and_readme(tmp_path) -> None:
    paths = write_neuromorphic_adapter_bundle(tmp_path, _network(), targets=("loihi2",))

    manifest_path = paths["loihi2:loihi2/adapter_manifest.json"]
    report_path = paths["loihi2:loihi2/nir_silicon_mapping_report.json"]
    readme_path = paths["loihi2:loihi2/README.md"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert manifest["target_id"] == "loihi2"
    assert report["targets"][0]["target_id"] == "loihi2"
    assert "does not claim execution on vendor hardware" in readme_path.read_text(encoding="utf-8")
