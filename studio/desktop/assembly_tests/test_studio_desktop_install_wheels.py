# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Complete real desktop wheel assembly command contracts

"""Exercise planning, normal full replacement and movement of the admitted staging runtime."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from tools.studio_desktop_assembly_process import run_assembly_command
from tools.studio_desktop_install_wheels import assemble_runtime
from tools.studio_desktop_runtime_inventory import (
    RuntimeInventory,
    inspect_runtime,
    verify_installed_wheels,
)
from tools.studio_desktop_wheel_catalog import WheelCatalog


def test_actual_complete_plan_preserves_installed_records(
    assembly_runtime: Path,
    assembly_catalog_path: Path,
    assembly_catalog: WheelCatalog,
    assembly_uv: Path,
    assembly_seconds: float,
    tmp_path: Path,
) -> None:
    """Planning checks real inputs and runs the normal uninstall dry run without mutation."""
    before = inspect_runtime(
        assembly_runtime, assembly_catalog, assembly_seconds, tmp_path / "before"
    )
    receipts = tmp_path / "plan"
    result = assemble_runtime(
        assembly_runtime,
        assembly_catalog_path,
        assembly_uv,
        receipts,
        assembly_seconds,
        apply=False,
    )
    assert result["state"] == "planned"
    assert result["packages"] == before.packages
    assert result["bundle_sealed"] is False and result["desktop_acceptance"] is False
    shared = result["shared_wheel_payloads"]
    assert isinstance(shared, dict) and len(shared) == 4
    assert all(len(item["archives"]) == 3 for item in shared.values())
    assert not (receipts / "uninstall").exists()
    assert (receipts / "selected-catalog.json").read_bytes() == assembly_catalog_path.read_bytes()
    assert (receipts / "uninstall-dry-run/receipt.json").is_file()
    after = inspect_runtime(
        assembly_runtime, assembly_catalog, assembly_seconds, tmp_path / "after"
    )
    assert after.record_sha256 == before.record_sha256
    assert after.packages == before.packages


def test_control_environment_is_refused_before_receipts_or_uninstall(
    assembly_catalog_path: Path,
    assembly_uv: Path,
    assembly_seconds: float,
    tmp_path: Path,
) -> None:
    """Even an explicit apply cannot replace its own controlling interpreter environment."""
    receipts = tmp_path / "forbidden-control"
    with pytest.raises(ValueError, match="must be external"):
        assemble_runtime(
            Path(sys.prefix),
            assembly_catalog_path,
            assembly_uv,
            receipts,
            assembly_seconds,
            apply=True,
        )
    assert not receipts.exists()


def test_managed_installation_layout_is_refused_before_receipts(
    assembly_catalog_path: Path,
    assembly_uv: Path,
    assembly_seconds: float,
    tmp_path: Path,
) -> None:
    """An owner-marked installed-version layout cannot be treated as disposable build staging."""
    root = tmp_path / "installed"
    runtime = root / "versions/selected-version/runtime"
    runtime.mkdir(parents=True)
    (root / "owner.json").write_text("{}\n")
    receipts = tmp_path / "forbidden-installed"
    with pytest.raises(ValueError, match="managed installed desktop version"):
        assemble_runtime(
            runtime, assembly_catalog_path, assembly_uv, receipts, assembly_seconds, apply=True
        )
    assert not receipts.exists()


def test_runtime_cannot_contain_its_selected_catalog(
    assembly_runtime: Path,
    assembly_catalog_path: Path,
    assembly_uv: Path,
    assembly_seconds: float,
    tmp_path: Path,
) -> None:
    """Inputs inside the replacement target are refused before any uninstall can erase them."""
    inside = assembly_runtime / "refused-assembly-catalog.json"
    assert not inside.exists()
    shutil.copyfile(assembly_catalog_path, inside)
    receipts = tmp_path / "forbidden-inside-input"
    try:
        with pytest.raises(ValueError, match="must be external"):
            assemble_runtime(
                assembly_runtime, inside, assembly_uv, receipts, assembly_seconds, apply=True
            )
        assert not receipts.exists()
    finally:
        inside.unlink()


def test_public_cli_replaces_complete_closure_and_moved_runtime_executes(
    assembly_runtime: Path,
    assembly_catalog_path: Path,
    assembly_catalog: WheelCatalog,
    assembly_uv: Path,
    assembly_seconds: float,
    tmp_path: Path,
) -> None:
    """The real complete CLI replacement retains original payloads and executes after movement."""
    receipts = tmp_path / "apply"
    result = run_assembly_command(
        [
            sys.executable,
            "-B",
            "-m",
            "tools.studio_desktop_install_wheels",
            "--runtime",
            str(assembly_runtime),
            "--catalog",
            str(assembly_catalog_path),
            "--uv",
            str(assembly_uv),
            "--receipts",
            str(receipts),
            "--timeout-seconds",
            str(assembly_seconds),
            "--apply",
        ],
        assembly_seconds,
        tmp_path / "actual-cli",
    )
    value = json.loads(result.stdout)
    assert value["state"] == "installed"
    assert value["bundle_sealed"] is False and value["desktop_acceptance"] is False
    assert value["packages"] == {wheel.name: wheel.version for wheel in assembly_catalog.wheels}
    assert value["schema_version"] == "studio.desktop.runtime-assembly.v2"
    assert len(value["shared_wheel_payloads"]) == 4
    assert all(len(item["archives"]) == 3 for item in value["shared_wheel_payloads"].values())
    policy = Path(__file__).resolve().parents[3] / "tools/studio_desktop_generated_bytecode.toml"
    policy_sha256 = hashlib.sha256(policy.read_bytes()).hexdigest()
    assert value["generated_bytecode_policy_sha256"] == policy_sha256
    assert set(value["uninstalled_upstream_bytecode"]) == {
        wheel.path.name for wheel in assembly_catalog.wheels if wheel.generated_bytecode
    }
    assert set(value["original_wheel_member_bindings"]) == {
        wheel.path.name for wheel in assembly_catalog.wheels
    }
    assert json.loads((receipts / "result.json").read_text()) == value
    inventory = inspect_runtime(
        assembly_runtime, assembly_catalog, assembly_seconds, tmp_path / "installed"
    )
    assert (
        verify_installed_wheels(
            assembly_runtime, assembly_catalog, inventory, require_pypa_cache_absence=True
        )
        == value["original_wheel_member_bindings"]
    )
    for record in inventory.records.values():
        assert not (Path(record).parent / "direct_url.json").exists()
    moved = assembly_runtime.with_name("moved ' runtime with spaces")
    assert not moved.exists()
    assembly_runtime.rename(moved)
    try:
        assert not assembly_runtime.exists()
        for name in ["pip", "sc-neurocore"]:
            command = [str(moved / "bin" / name), "--version" if name == "pip" else "--help"]
            observed = run_assembly_command(command, assembly_seconds, tmp_path / ("moved-" + name))
            assert observed.stdout
        python = moved / "bin" / inventory.executable.name
        imports = (
            "import json,sys,importlib.metadata as m; "
            "import sc_neurocore,sc_neurocore_engine,scpn_studio_platform; "
            "from PySide6 import QtCore,QtWebEngineCore; "
            "print(json.dumps({'prefix':sys.prefix,'qt':QtCore.qVersion(),"
            "'packages':{name:m.version(name) for name in "
            "['sc-neurocore','sc-neurocore-engine','scpn-studio-platform']},"
            "'core':sc_neurocore.__file__,'native':sc_neurocore_engine.__file__,"
            "'sdk':scpn_studio_platform.__file__}))"
        )
        observed = run_assembly_command(
            [str(python), "-I", "-B", "-c", imports], assembly_seconds, tmp_path / "moved-imports"
        )
        actual = json.loads(observed.stdout)
        assert Path(actual["prefix"]).resolve() == moved
        assert all(
            Path(actual[name]).resolve().is_relative_to(moved) for name in ["core", "native", "sdk"]
        )
        assert actual["packages"] == {
            name: inventory.packages[name]
            for name in ["sc-neurocore", "sc-neurocore-engine", "scpn-studio-platform"]
        }
        assert actual["qt"]
        moved_inventory = inspect_runtime(
            moved, assembly_catalog, assembly_seconds, tmp_path / "moved-inventory"
        )
        assert moved_inventory.record_sha256 == inventory.record_sha256
        assert (
            verify_installed_wheels(
                moved, assembly_catalog, moved_inventory, require_pypa_cache_absence=True
            )
            == value["original_wheel_member_bindings"]
        )
    finally:
        moved.rename(assembly_runtime)
    restored = inspect_runtime(
        assembly_runtime, assembly_catalog, assembly_seconds, tmp_path / "restored"
    )
    assert restored.record_sha256 == inventory.record_sha256


def test_public_cli_preserves_control_environment_refusal(
    assembly_catalog_path: Path,
    assembly_uv: Path,
    assembly_seconds: float,
    tmp_path: Path,
) -> None:
    """The documented apply entry point propagates its real pre-mutation refusal and status."""
    receipts = tmp_path / "forbidden-control-operation"
    with pytest.raises(subprocess.CalledProcessError) as raised:
        run_assembly_command(
            [
                sys.executable,
                "-B",
                "-m",
                "tools.studio_desktop_install_wheels",
                "--runtime",
                sys.prefix,
                "--catalog",
                str(assembly_catalog_path),
                "--uv",
                str(assembly_uv),
                "--receipts",
                str(receipts),
                "--timeout-seconds",
                str(assembly_seconds),
                "--apply",
            ],
            assembly_seconds,
            tmp_path / "control-refusal-command",
        )
    assert raised.value.returncode == 1
    assert b"must be external" in raised.value.stderr
    assert not receipts.exists()


@pytest.mark.parametrize("fault", ["record", "origin", "launcher"])
def test_real_producer_refuses_file_drift_after_its_live_probe(
    assembly_runtime: Path,
    assembly_catalog_path: Path,
    assembly_catalog: WheelCatalog,
    assembly_initial_inventory: RuntimeInventory,
    assembly_uv: Path,
    assembly_seconds: float,
    tmp_path: Path,
    fault: str,
) -> None:
    """Actual audited subprocess boundaries introduce real drift without replacing producer code."""
    control = Path(__file__).with_name("studio_desktop_install_controls.py")
    receipts = tmp_path / "file-fault"
    observed = run_assembly_command(
        [
            sys.executable,
            "-B",
            "-c",
            "import runpy,sys; path=sys.argv.pop(1); runpy.run_path(path,run_name='__main__')",
            str(control),
            "--runtime",
            str(assembly_runtime),
            "--catalog",
            str(assembly_catalog_path),
            "--uv",
            str(assembly_uv),
            "--receipts",
            str(receipts),
            "--seconds",
            str(assembly_seconds),
            "--fault",
            fault,
        ],
        assembly_seconds,
        tmp_path / "actual-file-fault-command",
    )
    result = json.loads(observed.stdout)
    assert result["fault"] == fault and result["injected"] is True
    assert result["tag_probes"] == (1 if fault == "record" else 2)
    assert result["bundle_sealed"] is False
    assert not (receipts / "assembly/result.json").exists()
    after = inspect_runtime(
        assembly_runtime, assembly_catalog, assembly_seconds, tmp_path / "restored-inventory"
    )
    assert after.packages == assembly_initial_inventory.packages
    assert after.record_sha256 == assembly_initial_inventory.record_sha256
    assert all(
        verify_installed_wheels(
            assembly_runtime, assembly_catalog, after, require_pypa_cache_absence=True
        ).values()
    )
    assert not list(assembly_runtime.rglob("*.audit-original"))
    if fault == "record":
        assert not (receipts / "assembly/uninstall").exists()
    else:
        assert (receipts / "assembly/uninstall/receipt.json").is_file()
        seed = json.loads((receipts / "actual-origin-seed/receipt.json").read_text())
        assert seed["cleanup"]["returncode"] == 0
