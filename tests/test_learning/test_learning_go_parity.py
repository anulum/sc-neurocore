# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Go CGO autonomous-learning parity tests

"""Parity checks for the Go autonomous-learning CGO bridge."""

from __future__ import annotations

import math
import os
import subprocess
from pathlib import Path

import pytest

try:
    from sc_neurocore._native.learning_bridge import (
        RULE_STDP,
        RustPlasticityRule,
        is_available,
    )

    FFI_AVAILABLE = is_available()
except ImportError:  # pragma: no cover - exercised only without the optional native bridge.
    FFI_AVAILABLE = False

REPO_ROOT = Path(__file__).resolve().parents[2]
GO_MODULE_DIR = REPO_ROOT / "src" / "sc_neurocore" / "accel" / "go"
GO_BRIDGE_SOURCE = GO_MODULE_DIR / "autonomous_learning" / "learning_bridge.go"
NATIVE_DIR = REPO_ROOT / "src" / "sc_neurocore" / "_native"
AUTONOMOUS_LEARNING_DOC = REPO_ROOT / "docs" / "api" / "autonomous_learning.md"
GO_IMPORT_PATH = "github.com/anulum/sc-neurocore/accel/autonomous_learning"
STALE_GO_IMPORT_PATH = "sc_neurocore/accel/go/autonomous_learning"
GO_SCRIPT = f"""
package main

import (
    "fmt"
    "log"
    "{GO_IMPORT_PATH}"
)

func main() {{
    rule := autonomous_learning.NewPlasticityRule(autonomous_learning.RuleStdp, 0.5, 0.1, 0.05)
    if rule == nil {{
        fmt.Println("NO_FFI")
        return
    }}
    defer rule.Destroy()

    if err := rule.Step(true, false, 0.0); err != nil {{
        log.Fatal(err)
    }}
    if err := rule.Step(false, true, 0.0); err != nil {{
        log.Fatal(err)
    }}
    fmt.Printf("%.9f\\n", rule.Weight())
}}
"""


def _go_env() -> dict[str, str]:
    """Return the environment needed for local CGO learning bridge execution."""
    env = dict(os.environ)
    env["CGO_ENABLED"] = "1"
    existing_runtime = env.get("LD_LIBRARY_PATH")
    existing_linker = env.get("CGO_LDFLAGS")
    configured = env.get("SC_NEUROCORE_LIB_PATH")
    native_path = str(Path(configured).resolve().parent) if configured else str(NATIVE_DIR)
    env["CGO_LDFLAGS"] = (
        f"-L{native_path}" if not existing_linker else f"-L{native_path} {existing_linker}"
    )
    env["LD_LIBRARY_PATH"] = (
        native_path if not existing_runtime else f"{native_path}:{existing_runtime}"
    )
    return env


def _run_go_script(tmp_path: Path) -> subprocess.CompletedProcess[str]:
    """Run the generated Go script from the autonomous-learning module root."""
    go_file = tmp_path / "main.go"
    go_file.write_text(GO_SCRIPT, encoding="utf-8")
    try:
        return subprocess.run(
            ["go", "run", str(go_file)],
            cwd=GO_MODULE_DIR,
            env=_go_env(),
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        pytest.skip("Go toolchain is not installed")


def _go_weight(result: subprocess.CompletedProcess[str]) -> float:
    """Parse a Go bridge result and fail on broken CGO setup."""
    assert result.returncode == 0, (
        "Go CGO autonomous-learning script failed.\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
    lines = result.stdout.strip().split()
    assert lines, "Go CGO autonomous-learning script produced no stdout"
    assert lines[-1] != "NO_FFI", "Go CGO autonomous-learning bridge returned a nil FFI handle"
    try:
        return float(lines[-1])
    except ValueError as exc:
        raise AssertionError(
            f"Go CGO autonomous-learning output is not a float: {lines[-1]!r}"
        ) from exc


def test_go_script_uses_module_import_path() -> None:
    """Lock the parity script to the Go module import path."""
    assert GO_IMPORT_PATH in GO_SCRIPT
    assert STALE_GO_IMPORT_PATH not in GO_SCRIPT


def test_go_docs_use_module_import_path() -> None:
    """Lock public Go setup docs to the module import path."""
    docs = AUTONOMOUS_LEARNING_DOC.read_text(encoding="utf-8")
    assert GO_IMPORT_PATH in docs
    assert STALE_GO_IMPORT_PATH not in docs


def test_go_bridge_passes_default_timestep_to_rust_ffi() -> None:
    """Lock the Go CGO signatures to the Rust learning C-FFI timestep ABI."""
    source = GO_BRIDGE_SOURCE.read_text(encoding="utf-8")
    assert "const DefaultDt float32 = 0.001" in source
    assert (
        "void step_rule(void* ptr, bool pre_spike, bool post_spike, float reward, float dt);"
        in source
    )
    assert (
        "C.step_rule(r.ptr, C.bool(preSpike), C.bool(postSpike), C.float(reward), C.float(dt))"
        in source
    )
    assert (
        "void step_learner(void* ptr, bool fired, bool pre_spike, float global_reward, float dt);"
        in source
    )
    assert (
        "void step_rule_layer(void* layer_ptr, const bool* pre_spikes, const bool* post_spikes, const float* rewards, float dt);"
        in source
    )


def test_go_bridge_guards_native_pointer_and_slice_domains() -> None:
    """Lock zero-length, non-finite, and closed-handle checks ahead of CGO."""
    source = GO_BRIDGE_SOURCE.read_text(encoding="utf-8")
    assert "if count <= 0" in source
    assert "len(preSpikes) != l.count" in source
    assert "learning reward must be finite" in source
    assert "learning timestep must be finite and positive" in source
    assert "if l == nil || l.ptr == nil" in source


def test_go_env_prepends_native_library_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """Check that local dynamic linking sees the checked-in native library first."""
    monkeypatch.delenv("SC_NEUROCORE_LIB_PATH", raising=False)
    monkeypatch.delenv("CGO_LDFLAGS", raising=False)
    monkeypatch.setenv("LD_LIBRARY_PATH", "/existing")
    env = _go_env()
    assert env["CGO_ENABLED"] == "1"
    assert env["CGO_LDFLAGS"] == f"-L{NATIVE_DIR}"
    assert env["LD_LIBRARY_PATH"] == f"{NATIVE_DIR}:/existing"


def test_go_env_sets_native_library_path_when_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    """Check local dynamic linking setup when no path was already configured."""
    monkeypatch.delenv("SC_NEUROCORE_LIB_PATH", raising=False)
    monkeypatch.delenv("CGO_LDFLAGS", raising=False)
    monkeypatch.delenv("LD_LIBRARY_PATH", raising=False)
    env = _go_env()
    assert env["CGO_LDFLAGS"] == f"-L{NATIVE_DIR}"
    assert env["LD_LIBRARY_PATH"] == str(NATIVE_DIR)


def test_go_env_uses_configured_fresh_library(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep parity execution on the same explicitly built Rust artifact."""
    monkeypatch.setenv("SC_NEUROCORE_LIB_PATH", "/tmp/fresh/libautonomous_learning.so")
    monkeypatch.setenv("CGO_LDFLAGS", "-Wl,--as-needed")
    monkeypatch.delenv("LD_LIBRARY_PATH", raising=False)
    env = _go_env()
    assert env["CGO_LDFLAGS"] == "-L/tmp/fresh -Wl,--as-needed"
    assert env["LD_LIBRARY_PATH"] == "/tmp/fresh"


@pytest.mark.parametrize(
    ("result", "message"),
    [
        (
            subprocess.CompletedProcess(
                args=["go"], returncode=1, stdout="", stderr="compile failed"
            ),
            "failed",
        ),
        (subprocess.CompletedProcess(args=["go"], returncode=0, stdout="", stderr=""), "no stdout"),
        (
            subprocess.CompletedProcess(args=["go"], returncode=0, stdout="NO_FFI\n", stderr=""),
            "nil FFI",
        ),
        (
            subprocess.CompletedProcess(
                args=["go"], returncode=0, stdout="not-a-float\n", stderr=""
            ),
            "not a float",
        ),
    ],
)
def test_go_weight_rejects_broken_results(
    result: subprocess.CompletedProcess[str],
    message: str,
) -> None:
    """Check that setup failures are reported as failures, not hidden skips."""
    with pytest.raises(AssertionError, match=message):
        _go_weight(result)


def test_run_go_script_reports_missing_toolchain(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Check the optional-toolchain boundary when Go is unavailable."""

    def missing_go(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
        raise FileNotFoundError("go")

    monkeypatch.setattr(subprocess, "run", missing_go)
    with pytest.raises(pytest.skip.Exception, match="Go toolchain is not installed"):
        _run_go_script(tmp_path)


@pytest.mark.skipif(not FFI_AVAILABLE, reason="Rust FFI not available")
def test_go_python_learning_parity(tmp_path: Path) -> None:
    """Compare the Go CGO bridge against the Python ctypes bridge."""
    rule = RustPlasticityRule(rule_type=RULE_STDP, weight=0.5, param_a=0.1, param_b=0.05)
    rule.step(True, False)
    rule.step(False, True)

    go_weight = _go_weight(_run_go_script(tmp_path))

    assert math.isclose(rule.weight, go_weight, rel_tol=1e-5), (
        f"Parity mismatch: Python={rule.weight}, Go={go_weight}"
    )
