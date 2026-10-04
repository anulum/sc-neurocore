# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Engine ABI inventory CLI contracts

"""Exercise the inventory CLI against the actual engine and persisted captures."""

from __future__ import annotations

import importlib
import inspect
import json
import os
import pickle
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
TOOL = ROOT / "tools/engine_abi_inventory.py"


def run_cli(*args: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    """Execute the public CLI and retain its actual status and diagnostics.

    Parameters
    ----------
    *args : str
        CLI arguments.
    env : dict of str to str, optional
        Explicit child environment.

    Returns
    -------
    subprocess.CompletedProcess
        Reaped command result.
    """
    return subprocess.run(
        [sys.executable, str(TOOL), *args],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )


@pytest.fixture(scope="session")
def captured_inventory(tmp_path_factory: pytest.TempPathFactory) -> bytes:
    """Capture the real engine once and retain its immutable inventory bytes.

    Parameters
    ----------
    tmp_path_factory : pytest.TempPathFactory
        Session-owned temporary-directory factory.

    Returns
    -------
    bytes
        Successful public CLI capture, copied independently by each test.
    """
    path = tmp_path_factory.mktemp("engine-abi-inventory") / "engine.json"
    result = run_cli("capture", "--output", str(path))
    assert result.returncode == 0, result.stderr
    return path.read_bytes()


@pytest.fixture
def capture_path(tmp_path: Path, captured_inventory: bytes) -> Path:
    """Give one test its own persisted copy of the actual captured interface.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Test-owned output directory.
    captured_inventory : bytes
        Immutable bytes captured through the real public CLI.

    Returns
    -------
    pathlib.Path
        Independent capture file; mutations cannot affect another test.
    """
    path = tmp_path / "engine.json"
    path.write_bytes(captured_inventory)
    return path


def test_capture_preserves_live_global_reference_aliases(capture_path: Path) -> None:
    """Capture the actual native class identity through both public namespaces."""
    capture = json.loads(capture_path.read_text())
    namespaces = capture["inventory"]["modules"]
    facade = namespaces["sc_neurocore_engine"]["FixedPointLif"]
    native = namespaces["sc_neurocore_engine.sc_neurocore_engine"]["FixedPointLif"]
    assert facade == native
    assert facade["__module__"] == "sc_neurocore_engine.sc_neurocore_engine"
    assert facade["global_reference"]["same_object"] is True
    assert facade["global_reference"]["error"] is None
    assert "sc_neurocore_engine.FixedPointLif" in facade["aliases"]
    assert "step" in facade["members"]
    assert facade["members"]["step"]["signature"] == "(self, /, leak_k, gain_k, i_t, noise_in=0)"


@pytest.mark.parametrize(
    "namespace", ["sc_neurocore_engine", "sc_neurocore_engine.sc_neurocore_engine"]
)
def test_live_export_pickle_roundtrips_match_captured_identity(
    capture_path: Path, namespace: str
) -> None:
    """Roundtrip every live exported class/function using freshly encoded bytes.

    This separate runtime contract checks actual pickle behaviour; production
    capture never executes deserialisation. Stored inventory data provides only
    names to check against the already imported engine and is never unpickled.
    Instance-state support belongs to each binding's behavioural tests.
    """
    module = importlib.import_module(namespace)
    capture = json.loads(capture_path.read_text())
    symbols = capture["inventory"]["modules"][namespace]
    checked = 0
    for name, details in symbols.items():
        value: object = getattr(module, name)
        if inspect.isclass(value) or inspect.isroutine(value):
            encoded = pickle.dumps(value, protocol=4)
            restored: object = pickle.loads(encoded)
            assert restored is value, f"{namespace}.{name} changed global identity"
            assert details["global_reference"]["same_object"] is True
            checked += 1
    assert checked > 0, f"{namespace} has no exported class/function contracts"


def test_compare_accepts_identical_interface_with_different_provenance(capture_path: Path) -> None:
    """Allow measured installation paths to differ while preserving the full ABI."""
    after = capture_path.with_name("relocated.json")
    capture = json.loads(capture_path.read_text())
    capture["provenance"]["site_packages"] = str(after.parent / "other-venv")
    after.write_text(json.dumps(capture))
    result = run_cli("compare", "--before", str(capture_path), "--after", str(after))
    assert result.returncode == 0, result.stderr


def test_compare_rejects_missing_native_export(capture_path: Path) -> None:
    """Fail if a real captured native function disappears from the interface."""
    after = capture_path.with_name("missing-export.json")
    capture = json.loads(capture_path.read_text())
    del capture["inventory"]["modules"]["sc_neurocore_engine.sc_neurocore_engine"]["popcount"]
    after.write_text(json.dumps(capture))
    result = run_cli("compare", "--before", str(capture_path), "--after", str(after))
    assert result.returncode == 1
    assert "sc_neurocore_engine.sc_neurocore_engine.popcount: removed" in result.stderr


def test_capture_refuses_editable_facade(tmp_path: Path) -> None:
    """Refuse a checkout facade even when its extension loads from site-packages."""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "bridge")
    result = run_cli(
        "capture", "--require-installed", "--output", str(tmp_path / "bad.json"), env=env
    )
    assert result.returncode == 2
    assert "does not originate in installed site-packages" in result.stderr
    assert not (tmp_path / "bad.json").exists()


def test_capture_reports_unwritable_output(tmp_path: Path) -> None:
    """Report the actual filesystem refusal when output names a directory."""
    result = run_cli("capture", "--output", str(tmp_path))
    assert result.returncode == 2
    assert "Engine ABI inventory failed" in result.stderr


def test_compare_refuses_invalid_json(capture_path: Path) -> None:
    """Refuse malformed persisted data before claiming interface equality."""
    broken = capture_path.with_name("truncated.json")
    broken.write_bytes(capture_path.read_bytes()[:100])
    result = run_cli("compare", "--before", str(capture_path), "--after", str(broken))
    assert result.returncode == 2
    assert "Engine ABI inventory failed" in result.stderr
