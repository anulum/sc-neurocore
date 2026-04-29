# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for self-hosted hub bundle generation

"""Tests for offline-first hub bundle generation."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from sc_neurocore.hub import (
    HubBundleConfig,
    build_benchmark_plan,
    build_hub_manifest,
    build_model_zoo_index,
    write_hub_bundle,
)


def test_model_zoo_index_lists_plugins_networks_and_weights() -> None:
    index = build_model_zoo_index()

    assert index["schema_version"] == "sc-neurocore.model-zoo-index.v1"
    assert [plugin["name"] for plugin in index["plugins"]] == [
        "AdEx",
        "Hodgkin-Huxley",
        "Izhikevich",
        "LIF",
    ]
    assert {entry["name"] for entry in index["network_configs"]} >= {
        "mnist_classifier",
        "shd_speech_classifier",
        "brunel_balanced_network",
    }
    assert [entry["name"] for entry in index["pretrained"]] == [
        "dvs_gesture",
        "mnist",
        "shd",
    ]


def test_hub_manifest_is_offline_first_and_points_to_local_storage() -> None:
    manifest = build_hub_manifest(HubBundleConfig(studio_port=8123, offline=True))

    assert manifest["schema_version"] == "sc-neurocore.self-hosted-hub.v1"
    assert manifest["services"]["studio"]["url"] == "http://127.0.0.1:8123"
    assert manifest["services"]["benchmark_runner"]["profile"] == "benchmark"
    assert manifest["storage"] == {
        "cache": "cache",
        "models": "models",
        "benchmark_results": "benchmarks/results",
    }
    assert manifest["network_policy"]["external_egress_required"] is False
    assert manifest["network_policy"]["offline_environment"] == {
        "SC_NEUROCORE_HUB_OFFLINE": "1",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    }


def test_benchmark_plan_is_opt_in() -> None:
    plan = build_benchmark_plan(HubBundleConfig(benchmarks_dir="bench"))

    assert plan["runner"]["service"] == "benchmark-runner"
    assert plan["runner"]["profile"] == "benchmark"
    assert plan["mounted_paths"]["benchmarks"] == "./bench"
    assert "not started with the Studio service" in plan["limitations"][1]


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"bind_host": ""}, "bind_host must not be empty"),
        ({"studio_port": 0}, "studio_port must be in the range"),
        ({"image": ""}, "image must not be empty"),
        ({"cache_dir": "/tmp/cache"}, "cache_dir must be"),
        ({"models_dir": "../models"}, "models_dir must be"),
    ],
)
def test_hub_bundle_config_rejects_invalid_values(kwargs: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        HubBundleConfig(**kwargs)


def test_write_hub_bundle_creates_compose_manifests_and_directories(tmp_path: Path) -> None:
    paths = write_hub_bundle(
        tmp_path,
        HubBundleConfig(studio_port=9000, cache_dir="local-cache", benchmarks_dir="bench"),
    )

    assert set(paths) == {
        "compose",
        "env_example",
        "manifest",
        "model_zoo_index",
        "benchmark_plan",
        "readme",
    }
    assert (tmp_path / "local-cache").is_dir()
    assert (tmp_path / "models").is_dir()
    assert (tmp_path / "bench" / "results").is_dir()
    compose = paths["compose"].read_text(encoding="utf-8")
    repo_context = os.path.relpath(Path(__file__).resolve().parents[1], tmp_path)
    assert f"context: {repo_context}" in compose
    assert f"      - {repo_context}:/workspace:ro" in compose
    assert "INSTALL_EXTRAS: studio,nir" in compose
    assert '"127.0.0.1:9000:9000"' in compose
    assert "profiles:" in compose
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    assert manifest["artefacts"]["model_zoo_index"] == "model_zoo_index.json"
    index = json.loads(paths["model_zoo_index"].read_text(encoding="utf-8"))
    assert index == manifest["model_zoo"]
    assert paths["readme"].read_text(encoding="utf-8").endswith("\n")
