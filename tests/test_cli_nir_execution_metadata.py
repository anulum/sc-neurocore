# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Real CLI consumers of recorded NIR execution settings

"""Run real compilation, mapping and HTTP inference with an exported NIR clock."""

from __future__ import annotations

import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from typing import Any
import urllib.error
import urllib.request

import nir
import numpy as np

from sc_neurocore.nir_bridge import from_nir, to_nir

_ROOT = Path(__file__).resolve().parents[1]


def _saved_model(tmp_path: Path) -> Path:
    graph = nir.NIRGraph(
        nodes={
            "input": nir.Input(np.array([2])),
            "lif": nir.LIF(
                tau=np.array([2.0, 3.0]),
                r=np.ones(2),
                v_leak=np.zeros(2),
                v_threshold=np.array([0.7, 0.8]),
            ),
            "output": nir.Output(np.array([2])),
        },
        edges=[("input", "lif"), ("lif", "output")],
    )
    path = tmp_path / "clocked.nir"
    to_nir(from_nir(graph, dt=0.125), path)
    return path


def _env() -> dict[str, str]:
    if os.environ.get("SC_NEUROCORE_INSTALLED_INTEROP") == "1":
        return {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
    return dict(os.environ, PYTHONPATH=str(_ROOT / "src"))


def _cli(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "sc_neurocore.cli", *arguments],
        env=_env(),
        capture_output=True,
        text=True,
        timeout=40,
    )


def test_compile_nir_inherits_recorded_clock_and_refuses_an_explicit_conflict(
    tmp_path: Path,
) -> None:
    """Emit identical real RTL with the inherited clock and its matching explicit option."""
    path = _saved_model(tmp_path)
    outputs = []
    for label, options in [("inherited", []), ("explicit", ["--dt", "0.125"])]:
        directory = tmp_path / label
        result = _cli(
            "compile-nir", str(path), "--module-name", "clocked", "-o", str(directory), *options
        )
        assert result.returncode == 0, result.stdout + result.stderr
        outputs.append({p.name: p.read_bytes() for p in directory.glob("*.v")})
        manifest = json.loads((directory / "scnir_source_manifest.json").read_text())
        assert manifest["total_neurons"] == 2
    assert outputs[0] and outputs[0] == outputs[1]
    rejected = tmp_path / "conflict"
    result = _cli("compile-nir", str(path), "--dt", "1", "-o", str(rejected))
    assert (
        result.returncode != 0
        and "conflict with recorded execution" in result.stdout + result.stderr
    )
    assert not rejected.exists()


def test_map_nir_inherits_clock_at_the_actual_file_boundary(tmp_path: Path) -> None:
    """Generate an actual capability report without inventing a timestep override."""
    path = _saved_model(tmp_path)
    output = tmp_path / "mapping"
    result = _cli("map-nir", str(path), "--hardware-targets", "loihi2", "-o", str(output))
    assert result.returncode == 0, result.stdout + result.stderr
    assert (output / "nir_silicon_mapping_report.json").is_file()
    result = _cli("map-nir", str(path), "--dt", "1", "-o", str(tmp_path / "refused"))
    assert (
        result.returncode != 0
        and "conflict with recorded execution" in result.stdout + result.stderr
    )


def _post(port: int, endpoint: str, value: dict[str, Any]) -> dict[str, Any]:
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/{endpoint}",
        data=json.dumps(value).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=3) as response:
        result: dict[str, Any] = json.load(response)
    return result


def test_serve_replays_the_saved_clock_over_actual_http_requests(tmp_path: Path) -> None:
    """Run the CLI process and compare nontrivial HTTP spike traces with the file's reference."""
    path = _saved_model(tmp_path)
    reference = from_nir(path)
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        port = reservation.getsockname()[1]
    process = subprocess.Popen(
        [sys.executable, "-m", "sc_neurocore.cli", "serve", str(path), "--port", str(port)],
        env=_env(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            if process.poll() is not None:
                stdout, stderr = process.communicate()
                raise AssertionError(stdout + stderr)
            try:
                with urllib.request.urlopen(
                    f"http://127.0.0.1:{port}/health", timeout=0.2
                ) as response:
                    assert json.load(response) == {"status": "ok"}
                break
            except urllib.error.URLError:
                time.sleep(0.02)
        else:
            raise AssertionError("Actual CLI inference server did not become ready")
        spikes = 0
        for step, current in enumerate(np.random.default_rng(2).uniform(0.5, 5, (32, 2)), 1):
            result = _post(port, "step", {"inputs": {"input": current.tolist()}})
            expected = reference.step({"input": current})["output"]
            np.testing.assert_array_equal(result["outputs"]["output"], expected)
            assert result["timestep"] == step
            spikes += int(expected.sum())
        assert 0 < spikes < 64
        assert _post(port, "reset", {})["timestep"] == 0
        reference.reset()
        np.testing.assert_array_equal(
            _post(port, "step", {"inputs": {"input": [3.0, 4.0]}})["outputs"]["output"],
            reference.step({"input": np.array([3.0, 4.0])})["output"],
        )
    finally:
        process.terminate()
        process.communicate(timeout=5)
