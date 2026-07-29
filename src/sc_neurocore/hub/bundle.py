# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Self-hosted hub bundle generator

"""Offline-first Docker Compose bundle generation for local hub deployments."""

from __future__ import annotations

import ipaddress
import json
import os
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from sc_neurocore.model_zoo import (
    auditory_processing,
    brunel_balanced_network,
    central_pattern_generator,
    cortical_column,
    decision_making_circuit,
    dvs_gesture_classifier,
    mnist_classifier,
    shd_speech_classifier,
    visual_cortex_v1,
    working_memory_circuit,
)
from sc_neurocore.model_zoo.model_zoo import PluginRegistry
from sc_neurocore.model_zoo.pretrained import _REGISTRY as _PRETRAINED_REGISTRY

SCHEMA_VERSION = "sc-neurocore.self-hosted-hub.v1"

# Declared container tmpfs path, not a host tempfile.
_CONTAINER_TMPFS_PATH = "/tmp"  # nosec B108
# Explicit operator-selected all-interface bind is allowed and recorded.
_ALL_INTERFACES_IPV4 = "0.0.0.0"  # nosec B104

_NETWORK_BUILDERS = (
    mnist_classifier,
    dvs_gesture_classifier,
    shd_speech_classifier,
    brunel_balanced_network,
    cortical_column,
    central_pattern_generator,
    decision_making_circuit,
    working_memory_circuit,
    auditory_processing,
    visual_cortex_v1,
)


@dataclass(frozen=True)
class HubBundleConfig:
    """Configuration for a local self-hosted hub bundle."""

    bind_host: str = "127.0.0.1"
    studio_port: int = 8001
    image: str = "sc-neurocore-hub:local"
    offline: bool = True
    cache_dir: str = "cache"
    models_dir: str = "models"
    dependency_mirror_dirs: tuple[str, ...] = ("mirrors/wheelhouse", "mirrors/huggingface")
    benchmarks_dir: str = "benchmarks"
    compose_name: str = "docker-compose.yml"

    def __post_init__(self) -> None:
        if not self.bind_host:
            raise ValueError("bind_host must not be empty")
        _validate_bind_host(self.bind_host)
        if not 1 <= self.studio_port <= 65535:
            raise ValueError("studio_port must be in the range 1..65535")
        if not self.image:
            raise ValueError("image must not be empty")
        for label, value in (
            ("cache_dir", self.cache_dir),
            ("models_dir", self.models_dir),
            ("benchmarks_dir", self.benchmarks_dir),
            ("compose_name", self.compose_name),
        ):
            _validate_relative_path(label, value)
        if self.offline and not self.dependency_mirror_dirs:
            raise ValueError("offline mode requires at least one dependency_mirror_dirs entry")
        for value in self.dependency_mirror_dirs:
            _validate_relative_path("dependency_mirror_dirs", value)
        if PurePosixPath(self.compose_name).name != self.compose_name:
            raise ValueError("compose_name must be a file name, not a nested path")


def build_model_zoo_index() -> dict[str, Any]:
    """Build a deterministic model-zoo index for hub manifests."""

    registry = PluginRegistry.with_builtins()
    plugin_entries = []
    for name in registry.list_plugins():
        plugin = registry.get(name)
        if plugin is None:
            continue
        meta = plugin.meta()
        plugin_entries.append(
            {
                "name": meta.name,
                "version": meta.version,
                "description": meta.description,
                "state_variables": list(meta.state_variables),
                "parameters": dict(sorted(meta.parameters.items())),
            }
        )

    network_entries = [
        {
            "name": builder.__name__,
            "kind": "network_config",
            "import_path": f"{builder.__module__}.{builder.__name__}",
        }
        for builder in _NETWORK_BUILDERS
    ]

    pretrained_entries = [
        {
            "name": name,
            "kind": "pretrained_weights",
            "weight_file": weight_file,
        }
        for name, (_, weight_file) in sorted(_PRETRAINED_REGISTRY.items())
    ]

    return {
        "schema_version": "sc-neurocore.model-zoo-index.v1",
        "plugins": plugin_entries,
        "network_configs": network_entries,
        "pretrained": pretrained_entries,
    }


def build_benchmark_plan(config: HubBundleConfig | None = None) -> dict[str, Any]:
    """Build the benchmark-runner plan included in the hub bundle."""

    cfg = config or HubBundleConfig()
    return {
        "schema_version": "sc-neurocore.hub-benchmark-plan.v1",
        "offline": cfg.offline,
        "runner": {
            "service": "benchmark-runner",
            "profile": "benchmark",
            "command": "python benchmarks/benchmark_suite.py --markdown",
            "results_dir": "/workspace/benchmarks/results",
        },
        "mounted_paths": {
            "benchmarks": f"./{cfg.benchmarks_dir}",
            "cache": f"./{cfg.cache_dir}",
        },
        "limitations": [
            "benchmark output depends on the host CPU and container runtime",
            "the benchmark profile is opt-in and is not started with the Studio service",
        ],
    }


def build_hub_manifest(config: HubBundleConfig | None = None) -> dict[str, Any]:
    """Build a deterministic manifest for a self-hosted hub bundle."""

    cfg = config or HubBundleConfig()
    return {
        "schema_version": SCHEMA_VERSION,
        "offline_default": cfg.offline,
        "services": {
            "studio": {
                "kind": "fastapi_studio",
                "url": f"http://{cfg.bind_host}:{cfg.studio_port}",
                "compose_service": "studio",
                "command": _studio_container_command(cfg),
                "healthcheck": f"http://127.0.0.1:{cfg.studio_port}/api/health",
                "writable_paths": [
                    "/var/lib/sc-neurocore/cache",
                    _CONTAINER_TMPFS_PATH,
                ],
            },
            "benchmark_runner": {
                "kind": "opt_in_benchmark_runner",
                "compose_service": "benchmark-runner",
                "profile": "benchmark",
                "command": "python benchmarks/benchmark_suite.py --markdown",
                "writable_paths": [
                    "/workspace/benchmarks/results",
                    "/var/lib/sc-neurocore/cache",
                    _CONTAINER_TMPFS_PATH,
                ],
            },
        },
        "service_contracts": {
            "studio": {
                "readiness_endpoint": "/api/health",
                "serves": [
                    "visual SNN design studio API",
                    "local model catalogue",
                    "local simulation and code-generation endpoints",
                ],
                "does_not_provide": [
                    "remote model hosting",
                    "automatic container-image publishing",
                    "hardware or cloud job submission",
                ],
            },
            "benchmark_runner": {
                "activation": "docker compose --profile benchmark run --rm benchmark-runner",
                "serves": ["local benchmark-suite execution"],
                "does_not_provide": ["continuous benchmark daemon"],
            },
        },
        "storage": {
            "cache": cfg.cache_dir,
            "models": cfg.models_dir,
            "dependency_mirrors": list(cfg.dependency_mirror_dirs),
            "benchmark_results": f"{cfg.benchmarks_dir}/results",
        },
        "artefacts": {
            "compose": cfg.compose_name,
            "env_example": ".env.example",
            "model_zoo_index": "model_zoo_index.json",
            "benchmark_plan": "benchmark_plan.json",
            "manifest": "hub_manifest.json",
        },
        "model_zoo": build_model_zoo_index(),
        "benchmark_plan": build_benchmark_plan(cfg),
        "network_policy": {
            "bind_host": cfg.bind_host,
            "ingress_scope": _ingress_scope(cfg.bind_host),
            "external_egress_required": False,
            "offline_environment": _offline_environment(cfg),
            "air_gapped_contract": {
                "requires_local_dependency_mirrors": cfg.offline,
                "dependency_mirror_dirs": list(cfg.dependency_mirror_dirs),
            },
        },
        "container_hardening": {
            "non_root_runtime_user": True,
            "read_only_root_filesystem": True,
            "no_new_privileges": True,
            "tmpfs_paths": [
                _CONTAINER_TMPFS_PATH,
            ],
            "restart_policy": "unless-stopped",
        },
        "limitations": [
            "bundle generation does not build or publish a container image",
            "Studio availability depends on installing the package with the studio extra",
            "benchmark results are generated only when the benchmark profile is run",
        ],
    }


def write_hub_bundle(
    output_dir: str | Path, config: HubBundleConfig | None = None
) -> dict[str, Path]:
    """Write a local Docker Compose hub bundle and return generated paths."""

    cfg = config or HubBundleConfig()
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    for rel in (
        cfg.cache_dir,
        cfg.models_dir,
        cfg.benchmarks_dir,
        f"{cfg.benchmarks_dir}/results",
        *cfg.dependency_mirror_dirs,
    ):
        (root / rel).mkdir(parents=True, exist_ok=True)

    paths = {
        "compose": root / cfg.compose_name,
        "env_example": root / ".env.example",
        "manifest": root / "hub_manifest.json",
        "model_zoo_index": root / "model_zoo_index.json",
        "benchmark_plan": root / "benchmark_plan.json",
        "readme": root / "README.md",
    }
    manifest = build_hub_manifest(cfg)
    paths["compose"].write_text(_compose_yaml(cfg, _relative_repo_context(root)), encoding="utf-8")
    paths["env_example"].write_text(_env_example(cfg), encoding="utf-8")
    paths["manifest"].write_text(_json(manifest), encoding="utf-8")
    paths["model_zoo_index"].write_text(_json(manifest["model_zoo"]), encoding="utf-8")
    paths["benchmark_plan"].write_text(_json(manifest["benchmark_plan"]), encoding="utf-8")
    paths["readme"].write_text(_readme(cfg), encoding="utf-8")
    return paths


def _compose_yaml(config: HubBundleConfig, repo_context: str) -> str:
    offline = _offline_environment(config)
    return f"""# SPDX-License-Identifier: AGPL-3.0-or-later
name: sc-neurocore-hub

x-hub-build: &hub-build
  context: {repo_context}
  dockerfile: deploy/Dockerfile
  args:
    INSTALL_EXTRAS: studio,nir

x-hub-env: &hub-env
  SC_NEUROCORE_HUB_OFFLINE: "{offline["SC_NEUROCORE_HUB_OFFLINE"]}"
  HF_HUB_OFFLINE: "{offline["HF_HUB_OFFLINE"]}"
  TRANSFORMERS_OFFLINE: "{offline["TRANSFORMERS_OFFLINE"]}"
  SC_NEUROCORE_CACHE_DIR: /var/lib/sc-neurocore/cache
  SC_NEUROCORE_MODELS_DIR: /var/lib/sc-neurocore/models

services:
  studio:
    build: *hub-build
    image: {config.image}
    pull_policy: never
    command: {_studio_container_command(config)}
    environment: *hub-env
    ports:
      - "{config.bind_host}:{config.studio_port}:{config.studio_port}"
    volumes:
      - ./{config.cache_dir}:/var/lib/sc-neurocore/cache:rw
      - ./{config.models_dir}:/var/lib/sc-neurocore/models:ro
    read_only: true
    tmpfs:
      - /tmp
    security_opt:
      - no-new-privileges:true
    healthcheck:
      test:
        - CMD
        - python
        - -c
        - "import urllib.request; urllib.request.urlopen('http://127.0.0.1:{config.studio_port}/api/health', timeout=2).read()"
      interval: 30s
      timeout: 5s
      retries: 5
      start_period: 20s
    restart: unless-stopped
    networks:
      - neurocore-hub

  benchmark-runner:
    profiles:
      - benchmark
    build: *hub-build
    image: {config.image}
    pull_policy: never
    command: python benchmarks/benchmark_suite.py --markdown
    environment: *hub-env
    working_dir: /workspace
    volumes:
      - {repo_context}:/workspace:ro
      - ./{config.benchmarks_dir}/results:/workspace/benchmarks/results:rw
      - ./{config.cache_dir}:/var/lib/sc-neurocore/cache:rw
    read_only: true
    tmpfs:
      - /tmp
    security_opt:
      - no-new-privileges:true
    networks:
      - neurocore-hub

networks:
  neurocore-hub:
    driver: bridge
"""


def _env_example(config: HubBundleConfig) -> str:
    offline = _offline_environment(config)
    return "\n".join(
        [
            "# SPDX-License-Identifier: AGPL-3.0-or-later",
            f"SC_NEUROCORE_HUB_BIND={config.bind_host}",
            f"SC_NEUROCORE_HUB_PORT={config.studio_port}",
            f"SC_NEUROCORE_HUB_IMAGE={config.image}",
            f"SC_NEUROCORE_HUB_OFFLINE={offline['SC_NEUROCORE_HUB_OFFLINE']}",
            f"HF_HUB_OFFLINE={offline['HF_HUB_OFFLINE']}",
            f"TRANSFORMERS_OFFLINE={offline['TRANSFORMERS_OFFLINE']}",
            "SC_NEUROCORE_CACHE_DIR=./cache",
            "SC_NEUROCORE_MODELS_DIR=./models",
            "",
        ]
    )


def _readme(config: HubBundleConfig) -> str:
    return f"""# SC-NeuroCore Self-Hosted Hub

This bundle starts the local Studio service and provides an opt-in benchmark
runner profile. It is offline-first: cache and model directories are mounted
locally, and no telemetry egress is required by the generated configuration.

## Start Studio

```bash
docker compose up studio
```

Studio binds to `http://{config.bind_host}:{config.studio_port}`.

## Run Benchmarks

```bash
docker compose --profile benchmark run --rm benchmark-runner
```

Benchmark output is written under `{config.benchmarks_dir}/results/`.

## Local Storage

- `{config.cache_dir}/` stores reusable local artefacts.
- `{config.models_dir}/` stores user-provided model files.
- `model_zoo_index.json` lists built-in model-zoo entries included in this
  source checkout.
"""


def _offline_environment(config: HubBundleConfig) -> dict[str, str]:
    value = "1" if config.offline else "0"
    return {
        "SC_NEUROCORE_HUB_OFFLINE": value,
        "HF_HUB_OFFLINE": value,
        "TRANSFORMERS_OFFLINE": value,
    }


def _studio_container_command(config: HubBundleConfig) -> str:
    return (
        "python -m uvicorn sc_neurocore.studio.app:create_app "
        f"--factory --host 0.0.0.0 --port {config.studio_port}"
    )


def _ingress_scope(bind_host: str) -> str:
    if bind_host in {"localhost", "127.0.0.1", "::1"}:
        return "loopback"
    try:
        address = ipaddress.ip_address(bind_host)
    except ValueError:
        return "operator_selected_hostname"
    if address.is_loopback:
        return "loopback"
    if address.is_unspecified:
        return "all_interfaces"
    if address.is_private:
        return "private_network"
    return "public_or_routable"


def _json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def _relative_repo_context(output_dir: Path) -> str:
    return os.path.relpath(_repo_root(), output_dir.resolve())


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _validate_relative_path(label: str, value: str) -> None:
    path = PurePosixPath(value)
    if not value or path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{label} must be a non-empty relative path without '..'")


def _validate_bind_host(value: str) -> None:
    if any(char.isspace() for char in value):
        raise ValueError("bind_host must not contain whitespace")
    if "/" in value:
        raise ValueError("bind_host must be a host name or IP address, not a CIDR")
    if value in {"localhost", _ALL_INTERFACES_IPV4, "::", "::1"}:
        return
    try:
        ipaddress.ip_address(value)
    except ValueError:
        if not all(part and part.replace("-", "").isalnum() for part in value.split(".")):
            raise ValueError("bind_host must be a valid host name or IP address") from None
