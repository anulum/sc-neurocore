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
import yaml

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
    assert "--host 0.0.0.0 --port 8123" in manifest["services"]["studio"]["command"]
    assert manifest["services"]["studio"]["healthcheck"] == "http://127.0.0.1:8123/api/health"
    assert manifest["services"]["benchmark_runner"]["profile"] == "benchmark"
    assert manifest["service_contracts"]["studio"]["readiness_endpoint"] == "/api/health"
    assert (
        "hardware or cloud job submission"
        in manifest["service_contracts"]["studio"]["does_not_provide"]
    )
    assert manifest["storage"] == {
        "cache": "cache",
        "models": "models",
        "dependency_mirrors": ["mirrors/wheelhouse", "mirrors/huggingface"],
        "benchmark_results": "benchmarks/results",
    }
    assert manifest["network_policy"]["external_egress_required"] is False
    assert manifest["network_policy"]["ingress_scope"] == "loopback"
    assert manifest["network_policy"]["offline_environment"] == {
        "SC_NEUROCORE_HUB_OFFLINE": "1",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    }
    assert manifest["network_policy"]["air_gapped_contract"] == {
        "requires_local_dependency_mirrors": True,
        "dependency_mirror_dirs": ["mirrors/wheelhouse", "mirrors/huggingface"],
    }
    assert manifest["container_hardening"] == {
        "non_root_runtime_user": True,
        "read_only_root_filesystem": True,
        "no_new_privileges": True,
        "tmpfs_paths": ["/tmp"],
        "restart_policy": "unless-stopped",
    }


def test_hub_manifest_records_non_loopback_ingress_scope() -> None:
    manifest = build_hub_manifest(HubBundleConfig(bind_host="10.10.0.5", offline=False))

    assert manifest["network_policy"]["ingress_scope"] == "private_network"
    assert manifest["network_policy"]["offline_environment"] == {
        "SC_NEUROCORE_HUB_OFFLINE": "0",
        "HF_HUB_OFFLINE": "0",
        "TRANSFORMERS_OFFLINE": "0",
    }
    assert manifest["network_policy"]["air_gapped_contract"] == {
        "requires_local_dependency_mirrors": False,
        "dependency_mirror_dirs": ["mirrors/wheelhouse", "mirrors/huggingface"],
    }


def test_hub_manifest_records_all_interface_ingress_scope() -> None:
    manifest = build_hub_manifest(HubBundleConfig(bind_host="0.0.0.0"))

    assert manifest["network_policy"]["ingress_scope"] == "all_interfaces"


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
        ({"bind_host": "127.0.0.1:8001"}, "bind_host must be"),
        ({"bind_host": "10.0.0.0/24"}, "bind_host must be"),
        ({"compose_name": "nested/docker-compose.yml"}, "compose_name must be a file name"),
        (
            {"offline": True, "dependency_mirror_dirs": ()},
            "offline mode requires at least one dependency_mirror_dirs entry",
        ),
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
    assert (tmp_path / "mirrors" / "wheelhouse").is_dir()
    assert (tmp_path / "mirrors" / "huggingface").is_dir()
    assert (tmp_path / "bench" / "results").is_dir()
    compose = paths["compose"].read_text(encoding="utf-8")
    repo_context = os.path.relpath(Path(__file__).resolve().parents[1], tmp_path)
    assert f"context: {repo_context}" in compose
    assert f"      - {repo_context}:/workspace:ro" in compose
    assert "INSTALL_EXTRAS: studio,nir" in compose
    assert '"127.0.0.1:9000:9000"' in compose
    assert "python -m uvicorn sc_neurocore.studio.app:create_app" in compose
    assert "--host 0.0.0.0 --port 9000" in compose
    assert "profiles:" in compose
    assert "pull_policy: never" in compose
    assert "read_only: true" in compose
    assert "no-new-privileges:true" in compose
    assert "/api/health" in compose
    assert "neurocore-hub:" in compose
    compose_doc = yaml.safe_load(compose)
    assert compose_doc["services"]["studio"]["healthcheck"]["test"][:3] == [
        "CMD",
        "python",
        "-c",
    ]
    assert compose_doc["services"]["studio"]["read_only"] is True
    assert compose_doc["services"]["benchmark-runner"]["profiles"] == ["benchmark"]
    assert compose_doc["networks"]["neurocore-hub"]["driver"] == "bridge"
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    assert manifest["artefacts"]["model_zoo_index"] == "model_zoo_index.json"
    index = json.loads(paths["model_zoo_index"].read_text(encoding="utf-8"))
    assert index == manifest["model_zoo"]
    assert paths["readme"].read_text(encoding="utf-8").endswith("\n")


def test_ingress_scope_classifies_host_categories() -> None:
    from sc_neurocore.hub.bundle import _ingress_scope

    assert _ingress_scope("example.com") == "operator_selected_hostname"
    assert _ingress_scope("127.0.0.2") == "loopback"
    assert _ingress_scope("8.8.8.8") == "public_or_routable"


def test_validate_bind_host_rejects_whitespace() -> None:
    from sc_neurocore.hub.bundle import _validate_bind_host

    with pytest.raises(ValueError, match="must not contain whitespace"):
        _validate_bind_host("bad host")


def test_model_zoo_index_skips_unresolvable_plugin_names(monkeypatch) -> None:
    # A listed plugin name that fails to resolve is skipped rather than crashing
    # the index build.
    class _GhostRegistry:
        @classmethod
        def with_builtins(cls) -> "_GhostRegistry":
            return cls()

        def list_plugins(self) -> list[str]:
            return ["ghost"]

        def get(self, name: str) -> None:
            return None

    monkeypatch.setattr("sc_neurocore.hub.bundle.PluginRegistry", _GhostRegistry)
    index = build_model_zoo_index()
    assert index["plugins"] == []
