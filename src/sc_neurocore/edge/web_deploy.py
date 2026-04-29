# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Browser deployment scaffold

"""Deterministic browser deployment artefacts for SC-NeuroCore models."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SUPPORTED_WEB_MODEL_SUFFIXES = frozenset({".nir", ".pt", ".pth", ".json"})


@dataclass(frozen=True)
class WebDeploymentConfig:
    """Configuration for generating browser deployment artefacts."""

    dt: float = 1.0
    bitstream_length: int = 256
    enable_webgpu: bool = True
    enable_wasm_threads: bool = False

    def __post_init__(self) -> None:
        if self.dt <= 0:
            raise ValueError("dt must be positive")
        if self.bitstream_length <= 0:
            raise ValueError("bitstream_length must be positive")


@dataclass(frozen=True)
class WebDeploymentManifest:
    """Manifest consumed by the generated browser runtime."""

    schema_version: str
    target: str
    model_name: str
    model_format: str
    bitstream_length: int
    dt: float
    artefacts: dict[str, str]
    capabilities: dict[str, bool]
    runtime_contract: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable manifest dictionary."""

        return {
            "schema_version": self.schema_version,
            "target": self.target,
            "model_name": self.model_name,
            "model_format": self.model_format,
            "bitstream_length": self.bitstream_length,
            "dt": self.dt,
            "artefacts": dict(self.artefacts),
            "capabilities": dict(self.capabilities),
            "runtime_contract": dict(self.runtime_contract),
        }


def build_web_deployment(
    model_path: str | Path,
    output_dir: str | Path,
    config: WebDeploymentConfig | None = None,
) -> WebDeploymentManifest:
    """Generate a static browser deployment scaffold for a model artefact."""

    cfg = config or WebDeploymentConfig()
    model = Path(model_path)
    if not model.is_file():
        raise FileNotFoundError(f"model file not found: {model}")

    suffix = model.suffix.lower()
    if suffix not in SUPPORTED_WEB_MODEL_SUFFIXES:
        supported = ", ".join(sorted(SUPPORTED_WEB_MODEL_SUFFIXES))
        raise ValueError(f"unsupported web model format '{suffix}'. Supported: {supported}")

    output = Path(output_dir)
    runtime_dir = output / "runtime"
    model_dir = output / "model"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    copied_model = model_dir / model.name
    shutil.copy2(model, copied_model)

    artefacts = {
        "manifest": "manifest.json",
        "html": "index.html",
        "runtime_js": "runtime/sc_neurocore_web.js",
        "webgpu_shader": "runtime/sc_neurocore_webgpu.wgsl",
        "model": f"model/{model.name}",
    }
    manifest = WebDeploymentManifest(
        schema_version="1.0",
        target="web",
        model_name=model.name,
        model_format=suffix.lstrip("."),
        bitstream_length=cfg.bitstream_length,
        dt=cfg.dt,
        artefacts=artefacts,
        capabilities={
            "webgpu": cfg.enable_webgpu,
            "wasm_threads": cfg.enable_wasm_threads,
            "offline_static": True,
        },
        runtime_contract={
            "entrypoint": "runtime/sc_neurocore_web.js",
            "model_loader": "fetch",
            "sc_encoding": "unipolar bit probability",
            "numeric_contract": "probabilities are clamped to [0, 1]",
            "wasm_module": None,
        },
    )

    (output / "manifest.json").write_text(
        json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output / "index.html").write_text(_render_index_html(), encoding="utf-8")
    (runtime_dir / "sc_neurocore_web.js").write_text(_render_runtime_js(), encoding="utf-8")
    (runtime_dir / "sc_neurocore_webgpu.wgsl").write_text(_render_webgpu_shader(), encoding="utf-8")
    return manifest


def _render_index_html() -> str:
    return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>SC-NeuroCore Web Deployment</title>
  <style>
    :root {
      color-scheme: light dark;
      font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #101418;
      color: #eef2f3;
    }
    body {
      margin: 0;
      min-height: 100vh;
      display: grid;
      place-items: center;
    }
    main {
      width: min(840px, calc(100vw - 32px));
      display: grid;
      gap: 16px;
    }
    h1 {
      margin: 0;
      font-size: 2rem;
      letter-spacing: 0;
    }
    pre {
      overflow: auto;
      padding: 16px;
      border: 1px solid #31404a;
      border-radius: 8px;
      background: #151c22;
    }
    .status {
      padding: 12px 14px;
      border: 1px solid #31404a;
      border-radius: 8px;
      background: #182128;
    }
  </style>
</head>
<body>
  <main>
    <h1>SC-NeuroCore Web Deployment</h1>
    <div class="status" id="status">Loading manifest...</div>
    <pre id="manifest">{}</pre>
  </main>
  <script type="module" src="./runtime/sc_neurocore_web.js"></script>
</body>
</html>
"""


def _render_runtime_js() -> str:
    return """async function detectWebGpu() {
  if (!("gpu" in navigator)) {
    return { available: false, reason: "navigator.gpu is unavailable" };
  }
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) {
    return { available: false, reason: "no WebGPU adapter was returned" };
  }
  return { available: true, reason: "WebGPU adapter available" };
}

async function loadManifest() {
  const response = await fetch("./manifest.json", { cache: "no-store" });
  if (!response.ok) {
    throw new Error(`manifest fetch failed: ${response.status}`);
  }
  return await response.json();
}

function renderStatus(manifest, webgpu) {
  const status = document.getElementById("status");
  const gpuText = manifest.capabilities.webgpu
    ? `WebGPU requested: ${webgpu.reason}`
    : "WebGPU not requested by manifest";
  status.textContent = `${manifest.model_name} ready. ${gpuText}.`;
}

async function main() {
  const manifest = await loadManifest();
  const webgpu = manifest.capabilities.webgpu
    ? await detectWebGpu()
    : { available: false, reason: "disabled" };
  document.getElementById("manifest").textContent = JSON.stringify(
    { manifest, webgpu },
    null,
    2,
  );
  renderStatus(manifest, webgpu);
}

main().catch((error) => {
  document.getElementById("status").textContent = `Deployment load failed: ${error.message}`;
});
"""


def _render_webgpu_shader() -> str:
    return """@group(0) @binding(0) var<storage, read> input_probabilities: array<f32>;
@group(0) @binding(1) var<storage, read_write> output_probabilities: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let index = id.x;
  if (index >= arrayLength(&output_probabilities)) {
    return;
  }
  output_probabilities[index] = clamp(input_probabilities[index], 0.0, 1.0);
}
"""
