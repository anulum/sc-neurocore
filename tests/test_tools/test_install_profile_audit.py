# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_tool() -> Any:
    tool_path = _repo_root() / "tools" / "install_profile_audit.py"
    spec = importlib.util.spec_from_file_location("install_profile_audit", tool_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_install_profile_audit_reports_trimmed_base_boundary() -> None:
    tool = _load_tool()

    report = tool.build_install_profile_audit(_repo_root())

    assert report["schema_version"] == tool.INSTALL_PROFILE_AUDIT_SCHEMA_VERSION
    assert report["project"]["name"] == "sc-neurocore"
    assert report["base_dependencies"] == [
        "numpy>=1.24",
        "scipy>=1.10",
        "defusedxml>=0.7.1",
        "tomli>=2.0; python_version < '3.11'",
    ]
    assert report["heavy_dependencies_in_base"] == []
    assert report["polyglot_research_sources_in_wheel"] == []
    assert "full" in report["optional_extras"]
    assert "license" in report["optional_extras"]
    assert report["offline_hardware_profile"] == {
        "profile": "hdl-offline",
        "pip_install": 'pip install "sc-neurocore[hdl]"',
        "docker_build_arg": "INSTALL_EXTRAS=hdl",
        "vivado_required_for_baseline": False,
        "static_primitive_pattern": "hdl/primitives/*.v",
        "expected_static_primitives": [
            "hdl/primitives/sc_bitstream_encoder.v",
            "hdl/primitives/sc_bitstream_synapse.v",
            "hdl/primitives/sc_dense_layer_core.v",
            "hdl/primitives/sc_dotproduct_to_current.v",
            "hdl/primitives/sc_firing_rate_bank.v",
            "hdl/primitives/sc_lif_neuron.v",
        ],
        "missing_static_primitives": [],
        "conda_recipe_aligned": True,
        "docker_wheel_build_covers_static_primitives": True,
        "hub_dependency_mirrors": ["mirrors/wheelhouse", "mirrors/huggingface"],
        "hub_air_gapped_contract": {
            "requires_local_dependency_mirrors": True,
            "dependency_mirror_dirs": ["mirrors/wheelhouse", "mirrors/huggingface"],
        },
        "hub_offline_mirror_gate": True,
    }
    assert report["passed"] is True


def test_conda_recipe_tracks_base_install_contract() -> None:
    tool = _load_tool()

    recipe = tool._read_conda_recipe(_repo_root())

    assert recipe["version"] == "3.15.2"
    assert recipe["run_dependencies"] == [
        "python >=3.10",
        "numpy >=1.24",
        "scipy >=1.10",
        "defusedxml >=0.7.1",
        "tomli >=2.0  # [py<311]",
    ]
    assert "sc_neurocore.hdl.resources" in recipe["test_imports"]
    assert any("list_baseline_primitive_rtl" in command for command in recipe["test_commands"])


def test_install_profile_audit_cli_writes_json(tmp_path: Path) -> None:
    output = tmp_path / "install_profile_audit.json"
    result = subprocess.run(
        [
            sys.executable,
            str(_repo_root() / "tools" / "install_profile_audit.py"),
            "--repo",
            str(_repo_root()),
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["passed"] is True
    assert payload["install_measurement"] == {"measured": False}


def test_install_measurement_uses_base_install_and_records_diagnostics(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    tool = _load_tool()
    commands: list[list[str]] = []

    class FakeTemporaryDirectory:
        def __init__(self, prefix: str) -> None:
            assert prefix == "scn-install-profile-"
            self.path = tmp_path

        def __enter__(self) -> str:
            return str(self.path)

        def __exit__(self, *args: object) -> None:
            return None

    class FakeEnvBuilder:
        def __init__(self, *, with_pip: bool) -> None:
            assert with_pip is True

        def create(self, venv_dir: Path) -> None:
            (venv_dir / "bin").mkdir(parents=True)

    def fake_run(command: list[str], **kwargs: Any) -> Any:
        assert kwargs["capture_output"] is True
        assert kwargs["text"] is True
        commands.append(command)
        if command[2:4] == ["pip", "install"]:
            return SimpleNamespace(
                returncode=0,
                stdout="\n".join(f"install stdout {index}" for index in range(20)),
                stderr="\n".join(f"install stderr {index}" for index in range(20)),
            )
        if command[2:4] == ["pip", "list"]:
            return SimpleNamespace(
                returncode=0,
                stdout=json.dumps(
                    [
                        {"name": "numpy", "version": "2.0.0"},
                        {"name": "sc-neurocore", "version": "3.15.2"},
                    ]
                ),
                stderr="",
            )
        return SimpleNamespace(
            returncode=1,
            stdout="",
            stderr="\n".join(f"smoke stderr {index}" for index in range(20)),
        )

    monkeypatch.setattr(tool.tempfile, "TemporaryDirectory", FakeTemporaryDirectory)
    monkeypatch.setattr(tool.venv, "EnvBuilder", FakeEnvBuilder)
    monkeypatch.setattr(tool.subprocess, "run", fake_run)

    measurement = tool._measure_local_no_deps_install(_repo_root())

    install_command = commands[0]
    assert "--no-deps" not in install_command
    assert "--no-build-isolation" not in install_command
    assert measurement["command"] == "python -m pip install <repo>"
    assert measurement["install_stdout_tail"] == [
        f"install stdout {index}" for index in range(8, 20)
    ]
    assert measurement["install_stderr_tail"] == [
        f"install stderr {index}" for index in range(8, 20)
    ]
    assert measurement["smoke_stderr_tail"] == [f"smoke stderr {index}" for index in range(8, 20)]
    assert measurement["installed_package_count"] == 2
    assert measurement["heavy_optional_packages_installed"] == []
    assert measurement["passed"] is False
