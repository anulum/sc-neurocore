# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Engine inventory metadata admission

"""Exercise public metadata admission against live objects and actual package copies."""

from __future__ import annotations

import ast
import importlib
import json
import os
import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path

import pytest

from tools.engine_abi_inventory import global_reference_identity, symbol_details
from tests.test_tools.test_engine_abi_inventory import TOOL, run_cli


@pytest.fixture
def consumer_package(tmp_path: Path) -> Path:
    """Copy the real facade and live extension into a test-owned consumer.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Test-owned directory, outside the checkout.

    Returns
    -------
    pathlib.Path
        Consumer import root. The shared package/binary is never modified.
        This deliberately editable fixture is not a source-qualified wheel.
    """
    facade = importlib.import_module("sc_neurocore_engine")
    native = importlib.import_module("sc_neurocore_engine.sc_neurocore_engine")
    assert isinstance(facade.__file__, str)
    assert isinstance(native.__file__, str)
    source = Path(facade.__file__).parent
    consumer = tmp_path / "consumer"
    package = consumer / "sc_neurocore_engine"
    for path in source.rglob("*.py"):
        target = package / path.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    binary = Path(native.__file__)
    shutil.copyfile(binary, package / binary.name)
    return consumer


def test_symbol_details_preserves_measured_float(tmp_path: Path) -> None:
    """Preserve an actual scalar timestamp without address-based object repr."""
    measured = tmp_path.stat().st_mtime
    details = symbol_details(measured)
    assert details["kind"] == "float"
    assert details["constant"] == repr(measured)


def test_global_reference_refuses_native_method_descriptor() -> None:
    """Report a live method descriptor that has no independent global identity."""
    native = importlib.import_module("sc_neurocore_engine.sc_neurocore_engine")
    result = global_reference_identity(native.FixedPointLif.step)
    assert result["same_object"] is False
    assert (
        result["error"] == "ValueError: global reference requires named module and qualified name"
    )


def test_global_reference_reports_changed_live_qualname() -> None:
    """Refuse a real wrapper whose renamed qualified reference cannot resolve."""
    facade = importlib.import_module("sc_neurocore_engine")
    wrapper = facade.HDCVector
    original = wrapper.__qualname__
    try:
        wrapper.__qualname__ = original + ".missing_reference"
        result = global_reference_identity(wrapper)
    finally:
        wrapper.__qualname__ = original
    assert result["same_object"] is False
    assert isinstance(result["error"], str)
    assert "AttributeError" in result["error"]
    assert "missing_reference" in result["error"]


@pytest.mark.parametrize(
    "mutation,diagnostic",
    [
        ("__all__ = None", "__all__ must be a list of names"),
        ("__all__ = []", "__all__ must be a list of names"),
        ("__all__ = ['missing_export']", "advertises absent exports"),
        (
            "from . import sc_neurocore_engine as _abi_extension\ndel _abi_extension.__version__",
            "sc_neurocore_engine.sc_neurocore_engine has no __version__",
        ),
        ("__file__ = None", "has no file provenance"),
    ],
)
def test_capture_refuses_damaged_real_facade(
    consumer_package: Path, tmp_path: Path, mutation: str, diagnostic: str
) -> None:
    """Refuse actual package metadata damage through the public capture CLI."""
    facade = consumer_package / "sc_neurocore_engine/__init__.py"
    with facade.open("a") as stream:
        stream.write("\n" + mutation + "\n")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(consumer_package)
    output = tmp_path / "refused.json"
    result = run_cli("capture", "--output", str(output), env=env)
    assert result.returncode == 2
    assert diagnostic in result.stderr
    assert not output.exists()


def test_installed_guard_refuses_checkout_without_loading_extension(
    consumer_package: Path, tmp_path: Path
) -> None:
    """Reject a real checkout facade before its missing extension can execute."""
    package = consumer_package / "sc_neurocore_engine"
    native = importlib.import_module("sc_neurocore_engine.sc_neurocore_engine")
    assert isinstance(native.__file__, str)
    (package / Path(native.__file__).name).unlink()
    env = os.environ.copy()
    env["PYTHONPATH"] = str(consumer_package)
    output = tmp_path / "missing-extension.json"
    result = run_cli("capture", "--require-installed", "--output", str(output), env=env)
    assert result.returncode == 2
    assert "does not originate in installed site-packages" in result.stderr
    assert not output.exists()


@pytest.fixture
def consumer_environment(
    consumer_package: Path, tmp_path: Path
) -> tuple[Path, Path, dict[str, str]]:
    """Prepare an isolated interpreter for real site-packages origin admission.

    Parameters
    ----------
    consumer_package : pathlib.Path
        Test-owned copy of the actual package and extension.
    tmp_path : pathlib.Path
        Test-owned environment directory.

    Returns
    -------
    tuple
        Interpreter, installed package path and explicit dependency environment.
        File copying tests origin admission; it does not qualify a release wheel.
    """
    environment = tmp_path / "environment"
    subprocess.run(
        [sys.executable, "-m", "venv", "--without-pip", str(environment)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    )
    python = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    location = subprocess.run(
        [str(python), "-c", "import sysconfig; print(sysconfig.get_path('purelib'))"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    site_packages = Path(location.stdout.strip())
    package = site_packages / "sc_neurocore_engine"
    shutil.copytree(consumer_package / "sc_neurocore_engine", package)
    dependencies = Path(sysconfig.get_path("purelib"))
    if os.environ.get("COVERAGE_PROCESS_CONFIG"):
        for startup in dependencies.glob("*coverage.pth"):
            shutil.copyfile(startup, site_packages / startup.name)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(dependencies)
    return python, package, env


@pytest.mark.parametrize("external_extension", [False, True])
def test_installed_guard_checks_both_loaded_origins(
    consumer_environment: tuple[Path, Path, dict[str, str]],
    tmp_path: Path,
    external_extension: bool,
) -> None:
    """Accept both local origins and refuse an extension resolved outside them."""
    python, package, env = consumer_environment
    if external_extension:
        native = importlib.import_module("sc_neurocore_engine.sc_neurocore_engine")
        assert isinstance(native.__file__, str)
        copied = package / Path(native.__file__).name
        outside = tmp_path / "external-native"
        outside.mkdir()
        shutil.move(copied, outside / copied.name)
        facade = package / "__init__.py"
        source = facade.read_text()
        insert_after = 0
        for statement in ast.parse(source).body:
            if isinstance(statement, ast.ImportFrom) and statement.module == "__future__":
                insert_after = max(insert_after, statement.end_lineno or statement.lineno)
        lines = source.splitlines(keepends=True)
        lines.insert(insert_after, f"__path__.append({str(outside)!r})\n")
        facade.write_text("".join(lines))
    output = tmp_path / "origin-inventory.json"
    result = subprocess.run(
        [str(python), str(TOOL), "capture", "--require-installed", "--output", str(output)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    if external_extension:
        assert result.returncode == 2
        assert "sc_neurocore_engine.sc_neurocore_engine does not originate" in result.stderr
        assert not output.exists()
    else:
        assert result.returncode == 0, result.stderr
        capture = json.loads(output.read_text())
        assert all(record["installed"] for record in capture["provenance"]["modules"].values())
        assert str(package.parent.resolve()) in capture["provenance"]["installation_roots"]
