# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia public runtime configuration refusal

"""Exercise operator configuration admission through real ConvertedSNN calls."""

import importlib.metadata
from pathlib import Path

import pytest

from sc_neurocore.conversion import ConvertedSNN
from tests.julia_runtimes import require_julia_runtime


@pytest.mark.parametrize(
    "condition,message",
    [
        ("disabled", "SC_NEUROCORE_IF_JULIA_ENABLED=1"),
        ("invalid-opt-in", "opt-in must be 0 or 1"),
        ("missing-executable", "absolute Julia executable"),
        ("non-executable", "absolute Julia executable"),
        ("missing-project", "absolute locked Julia project"),
        ("threads", "one thread"),
        ("signals", "signal handling"),
        ("conda", "offline packages"),
        ("online", "offline packages"),
        ("version", "versions must match exactly"),
        ("manifest-version", "versions must match exactly"),
        ("malformed-project", "locked project or JuliaCall dependency unavailable"),
        ("malformed-manifest", "locked project or JuliaCall dependency unavailable"),
        ("empty-manifest-version", "locked project or JuliaCall dependency unavailable"),
    ],
)
def test_public_julia_refuses_unsafe_configuration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, condition: str, message: str
) -> None:
    """Refuse invalid operator settings before initializing or resolving Julia packages."""
    executables = [require_julia_runtime("1.11")]
    assert executables, "installed Julia executable required"
    version = importlib.metadata.version("juliacall")
    project = tmp_path / "project"
    project.mkdir()
    declaration = project / "Project.toml"
    manifest = project / "Manifest.toml"
    declaration.write_text(f'[compat]\nPythonCall = "={version}"\n')
    manifest.write_text(f'[[deps.PythonCall]]\nversion = "{version}"\n')
    settings = {
        "SC_NEUROCORE_IF_JULIA_ENABLED": "1",
        "PYTHON_JULIACALL_EXE": str(executables[-1]),
        "PYTHON_JULIACALL_PROJECT": str(project),
        "PYTHON_JULIACALL_THREADS": "1",
        "PYTHON_JULIACALL_HANDLE_SIGNALS": "yes",
        "JULIA_CONDAPKG_BACKEND": "Null",
        "JULIA_PKG_OFFLINE": "true",
    }
    if condition == "disabled":
        settings["SC_NEUROCORE_IF_JULIA_ENABLED"] = "0"
    elif condition == "invalid-opt-in":
        settings["SC_NEUROCORE_IF_JULIA_ENABLED"] = "yes"
    elif condition == "missing-executable":
        settings["PYTHON_JULIACALL_EXE"] = ""
    elif condition == "non-executable":
        executable = tmp_path / "not-executable"
        executable.write_text("not a Julia executable")
        executable.chmod(0o600)
        settings["PYTHON_JULIACALL_EXE"] = str(executable)
    elif condition == "missing-project":
        manifest.unlink()
    elif condition == "threads":
        settings["PYTHON_JULIACALL_THREADS"] = "2"
    elif condition == "signals":
        settings["PYTHON_JULIACALL_HANDLE_SIGNALS"] = "no"
    elif condition == "conda":
        settings["JULIA_CONDAPKG_BACKEND"] = "Current"
    elif condition == "online":
        settings["JULIA_PKG_OFFLINE"] = "false"
    elif condition == "version":
        declaration.write_text('[compat]\nPythonCall = "=0.0.0"\n')
    elif condition == "manifest-version":
        manifest.write_text('[[deps.PythonCall]]\nversion = "0.0.0"\n')
    elif condition == "malformed-project":
        declaration.write_text("[malformed")
    elif condition == "malformed-manifest":
        manifest.write_text("[deps]\nPythonCall = 42\n")
    elif condition == "empty-manifest-version":
        manifest.write_text("[deps]\nPythonCall = []\n")
    for key, value in settings.items():
        monkeypatch.setenv(key, value)
    monkeypatch.delenv("SC_NEUROCORE_IF_RUST_LIB", raising=False)
    monkeypatch.delenv("SC_NEUROCORE_IF_GO_LIB", raising=False)
    model = ConvertedSNN([[[1.0]]], [None], [1.0], T=16)
    with pytest.raises(RuntimeError, match=message):
        model.run([1.0], backend="auto" if condition == "invalid-opt-in" else "julia")
    assert model.run([1.0], backend="numpy").tolist() == [16.0]
