# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Declared complete desktop assembly test inputs

"""Admit real staging inputs only when a desktop runtime test requests them."""

from __future__ import annotations

import math
import os
import shutil
import stat
import sys
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager
from pathlib import Path

import pytest
from installer import install
from installer.sources import WheelFile

from tools.studio_desktop_assembly_process import run_assembly_command
from tools.studio_desktop_runtime_inventory import (
    RuntimeInventory,
    inspect_runtime,
    verify_installed_wheels,
)
from tools.studio_desktop_wheel_catalog import WheelCatalog, read_catalog
from tools.studio_desktop_wheel_destination import RelocatableWheelDestination


@pytest.fixture(scope="session")
def assembly_runtime() -> Path:
    """Select the explicitly supplied standalone staging prefix, never installed user data."""
    configured = os.environ.get("SC_NEUROCORE_STUDIO_ASSEMBLY_RUNTIME")
    if not configured:
        pytest.fail("SC_NEUROCORE_STUDIO_ASSEMBLY_RUNTIME must name the admitted staging prefix.")
    root = Path(configured).resolve(strict=True)
    assert root.is_dir()
    assert not Path(sys.prefix).resolve().is_relative_to(root)
    assert not Path(sys.executable).resolve().is_relative_to(root)
    assert not (
        root.parent.parent.name == "versions"
        and (root.parent.parent.parent / "owner.json").exists()
    )
    return root


@pytest.fixture(scope="session")
def assembly_catalog_path() -> Path:
    """Locate the explicitly supplied complete adjacent-archive catalog for API and CLI tests."""
    configured = os.environ.get("SC_NEUROCORE_STUDIO_ASSEMBLY_CATALOG")
    if not configured:
        pytest.fail("SC_NEUROCORE_STUDIO_ASSEMBLY_CATALOG must name the acquired complete catalog.")
    return Path(configured).resolve(strict=True)


@pytest.fixture(scope="session")
def assembly_catalog(assembly_catalog_path: Path) -> WheelCatalog:
    """Validate the original complete catalog and every selected archive before runtime tests."""
    return read_catalog(assembly_catalog_path)


@pytest.fixture(scope="session")
def assembly_incompatible_catalog() -> WheelCatalog:
    """Read the separately acquired genuine foreign-platform wheel used for ABI refusal."""
    configured = os.environ.get("SC_NEUROCORE_STUDIO_ASSEMBLY_INCOMPATIBLE_CATALOG")
    if not configured:
        pytest.fail(
            "SC_NEUROCORE_STUDIO_ASSEMBLY_INCOMPATIBLE_CATALOG must bind the foreign wheel."
        )
    catalog = read_catalog(Path(configured))
    assert len(catalog.wheels) == 1
    return catalog


@pytest.fixture(scope="session")
def assembly_seconds() -> float:
    """Use the executing job's declared finite command deadline without a host-size constant."""
    configured = os.environ.get("SC_NEUROCORE_STUDIO_ASSEMBLY_COMMAND_SECONDS")
    if not configured:
        pytest.fail("SC_NEUROCORE_STUDIO_ASSEMBLY_COMMAND_SECONDS must declare the job deadline.")
    seconds = float(configured)
    assert math.isfinite(seconds) and seconds > 0
    return seconds


@pytest.fixture(scope="session")
def assembly_uv() -> Path:
    """Resolve the actual external uv executable provisioned by the locked build profile."""
    executable = shutil.which("uv")
    assert executable is not None, "The locked build profile must provide uv."
    return Path(executable).resolve(strict=True)


@pytest.fixture(scope="session")
def assembly_initial_inventory(
    assembly_runtime: Path,
    assembly_catalog: WheelCatalog,
    assembly_seconds: float,
    tmp_path_factory: pytest.TempPathFactory,
) -> RuntimeInventory:
    """Read the actual standalone interpreter once to locate its original installed records."""
    return inspect_runtime(
        assembly_runtime,
        assembly_catalog,
        assembly_seconds,
        tmp_path_factory.mktemp("initial-runtime-inventory") / "probe",
    )


@pytest.fixture
def assembly_runtime_with_data_scripts(
    assembly_runtime: Path,
    assembly_catalog: WheelCatalog,
    assembly_initial_inventory: RuntimeInventory,
    assembly_seconds: float,
    assembly_uv: Path,
    tmp_path: Path,
) -> Iterator[tuple[WheelCatalog, RuntimeInventory]]:
    """Install genuine Python and ELF data-script wheels, then restore the original closure."""
    configured = os.environ.get("SC_NEUROCORE_STUDIO_ASSEMBLY_SCRIPT_CATALOG")
    if not configured:
        pytest.fail("SC_NEUROCORE_STUDIO_ASSEMBLY_SCRIPT_CATALOG must bind real script wheels.")
    scripts = read_catalog(Path(configured))
    assert {wheel.name for wheel in scripts.wheels} == {"docutils", "ninja"}
    assert scripts.python_version == assembly_catalog.python_version
    assert scripts.platform == assembly_catalog.platform
    assert not set(assembly_initial_inventory.packages).intersection(
        wheel.name for wheel in scripts.wheels
    )
    before = inspect_runtime(
        assembly_runtime, assembly_catalog, assembly_seconds, tmp_path / "before-script-wheels"
    )
    assert before.record_sha256 == assembly_initial_inventory.record_sha256
    combined = WheelCatalog(
        assembly_catalog.python_version,
        assembly_catalog.platform,
        (*assembly_catalog.wheels, *scripts.wheels),
    )
    destination = RelocatableWheelDestination(before.scheme, str(before.executable), "posix")
    try:
        for wheel in scripts.wheels:
            with WheelFile.open(wheel.path) as source:
                install(source, destination, {"INSTALLER": b"installer\n"})
        installed = inspect_runtime(
            assembly_runtime, combined, assembly_seconds, tmp_path / "with-script-wheels"
        )
        run_assembly_command(
            [
                str(assembly_uv),
                "--no-config",
                "--offline",
                "pip",
                "check",
                "--python",
                str(installed.executable),
            ],
            assembly_seconds,
            tmp_path / "script-dependency-check",
        )
        yield combined, installed
    finally:
        run_assembly_command(
            [
                str(assembly_uv),
                "--no-config",
                "--offline",
                "pip",
                "uninstall",
                "--python",
                str(before.executable),
                "--system",
                "docutils",
                "ninja",
            ],
            assembly_seconds,
            tmp_path / "uninstall-script-wheels",
        )
        restored = inspect_runtime(
            assembly_runtime, assembly_catalog, assembly_seconds, tmp_path / "restored-runtime"
        )
        assert restored.record_sha256 == before.record_sha256
        assert restored.packages == before.packages
        assert all(verify_installed_wheels(assembly_runtime, assembly_catalog, restored).values())


@pytest.fixture
def assembly_file_change(
    assembly_runtime: Path,
) -> Callable[[Path, bytes | None], AbstractContextManager[None]]:
    """Replace one real staging file temporarily and restore its original inode on every exit."""

    @contextmanager
    def change(path: Path, content: bytes | None) -> Iterator[None]:
        assert path.is_file() and not path.is_symlink()
        assert path.resolve(strict=True).is_relative_to(assembly_runtime)
        original = path.with_name(path.name + ".assembly-original")
        assert not original.exists() and not original.is_symlink()
        mode = stat.S_IMODE(path.stat().st_mode)
        path.rename(original)
        try:
            if content is not None:
                path.write_bytes(content)
                path.chmod(mode)
            yield
        finally:
            if path.exists() or path.is_symlink():
                path.unlink()
            original.rename(path)

    return change


@pytest.fixture(scope="session")
def assembly_probe_command() -> Callable[[Path, str], list[str]]:
    """Trace the actual isolated probe with external coverage while preserving its import paths."""
    import coverage

    probe = Path(__file__).resolve().parents[3] / "tools/studio_desktop_probe_runtime.py"
    active = coverage.Coverage.current()
    config = active.config.config_file if active is not None else None
    package = Path(coverage.__file__).resolve(strict=True)
    source = (
        "import importlib.util,runpy,sys\n"
        "paths=list(sys.path)\n"
        "package,config,probe,interface=sys.argv[1:]\n"
        "spec=importlib.util.spec_from_file_location('coverage',package,"
        "submodule_search_locations=[str(__import__('pathlib').Path(package).parent)])\n"
        "module=importlib.util.module_from_spec(spec)\n"
        "sys.modules['coverage']=module\n"
        "spec.loader.exec_module(module)\n"
        "cov=module.Coverage(config_file=config,data_suffix=True)\n"
        "cov.start()\n"
        "try:\n"
        "    assert sys.path==paths, 'Probe tracing changed the target import paths.'\n"
        "    sys.argv=[probe]\n"
        "    if interface=='cli':\n"
        "        runpy.run_path(probe,run_name='__main__')\n"
        "    else:\n"
        "        runpy.run_path(probe)['main']()\n"
        "finally:\n"
        "    cov.stop()\n"
        "    cov.save()\n"
    )

    def _command(executable: Path, interface: str) -> list[str]:
        arguments = [str(executable), "-I", "-B"]
        if active is None:
            if interface == "cli":
                return [*arguments, str(probe)]
            return [
                *arguments,
                "-c",
                "import runpy,sys; runpy.run_path(sys.argv[1])['main']()",
                str(probe),
            ]
        assert config is not None
        return [*arguments, "-c", source, str(package), config, str(probe), interface]

    return _command
