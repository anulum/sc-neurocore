<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- (C) Concepts 1996-2026 Miroslav Sotek. All rights reserved. -->
<!-- (C) Code 2020-2026 Miroslav Sotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SC-NeuroCore documentation -->

# Self-Hosted Hub

**Module:** `sc_neurocore.hub`
**Primary API:** `write_hub_bundle(...)`
**CLI:** `sc-neurocore hub-init`

The self-hosted hub generator writes an offline-first Docker Compose bundle
for local SC-NeuroCore Studio and opt-in benchmark execution. It is a bundle
generator, not a hosted service and not a remote registry.

## Generated Artefacts

```python
from sc_neurocore.hub import HubBundleConfig, write_hub_bundle

paths = write_hub_bundle(
    "build/hub",
    HubBundleConfig(
        bind_host="127.0.0.1",
        studio_port=8001,
        image="sc-neurocore-hub:local",
        offline=True,
    ),
)
```

The output directory contains:

| Path | Purpose |
|------|---------|
| `docker-compose.yml` | Studio and benchmark-runner services |
| `.env.example` | Offline/cache/model environment defaults |
| `hub_manifest.json` | Deterministic service, storage, network, and hardening contract |
| `model_zoo_index.json` | Built-in plugin, network-config, and pretrained-weight index |
| `benchmark_plan.json` | Opt-in benchmark-runner contract |
| `README.md` | Local operator instructions |
| `cache/` | Writable local cache mount |
| `models/` | Read-only user model mount |
| `benchmarks/results/` | Writable benchmark output mount |

## Security And Operations Contract

The generated Studio service defaults to `127.0.0.1`, so it is not exposed
outside the host unless the operator explicitly changes `--bind-host`.
Offline mode sets `SC_NEUROCORE_HUB_OFFLINE=1`, `HF_HUB_OFFLINE=1`, and
`TRANSFORMERS_OFFLINE=1`; `--online` clears those generated flags.

Compose hardening in the generated bundle:

| Control | Setting |
|---------|---------|
| Runtime user | Non-root, inherited from `deploy/Dockerfile` |
| Root filesystem | `read_only: true` |
| Writable mounts | Cache, model/result directories as needed, plus `/tmp` tmpfs |
| Privilege escalation | `no-new-privileges:true` |
| Image pulling | `pull_policy: never` |
| Readiness | Studio `/api/health` healthcheck |
| Benchmarks | Opt-in `benchmark` profile |

## CLI

```bash
sc-neurocore hub-init --output build/hub --port 8001
docker compose -f build/hub/docker-compose.yml up studio
```

Optional flags:

| Flag | Default | Meaning |
|------|---------|---------|
| `--bind-host` | `127.0.0.1` | Host address used for Studio port publishing |
| `--port` | `8001` | Studio service port |
| `--hub-image` | `sc-neurocore-hub:local` | Generated Compose image tag |
| `--online` | unset | Generate online-mode environment flags |

## Boundaries

The hub bundle does not build or publish container images by itself, does not
submit hardware or cloud jobs, and does not claim benchmark results until the
operator runs the opt-in benchmark profile. Real availability still depends on
the container build and the package being installed with the Studio runtime
dependencies.

## API

::: sc_neurocore.hub
    options:
      show_root_heading: true
      show_source: true
