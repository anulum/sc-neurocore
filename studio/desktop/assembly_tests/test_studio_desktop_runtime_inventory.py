# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Actual desktop runtime inventory and original wheel bindings

"""Match the real standalone interpreter, installed files and selected original archives."""

from __future__ import annotations

import json
import math
import os
import sys
from collections.abc import Callable
from contextlib import AbstractContextManager
from dataclasses import replace
from pathlib import Path
from zipfile import ZipFile

import pytest

from tools.studio_desktop_assembly_process import run_assembly_command
from tools.studio_desktop_runtime_inventory import (
    RuntimeInventory,
    inspect_runtime,
    parse_runtime_inventory,
    verify_installed_wheels,
)
from tools.studio_desktop_wheel_catalog import WheelCatalog


@pytest.fixture(scope="module")
def captured_runtime_reports(
    assembly_initial_inventory: RuntimeInventory,
    assembly_seconds: float,
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[bytes, bytes]:
    """Capture the actual standalone probe and isolated pip diagnostic for protocol tests."""
    tags = run_assembly_command(
        [
            str(assembly_initial_inventory.executable),
            "-I",
            "-B",
            "-m",
            "pip",
            "--isolated",
            "debug",
            "--verbose",
        ],
        assembly_seconds,
        tmp_path_factory.mktemp("captured-runtime-reports") / "wheel-tags",
        environment={**os.environ, "PIP_CONFIG_FILE": os.devnull},
    )
    return assembly_initial_inventory.raw_stdout.encode("utf-8"), tags.stdout


def test_real_runtime_inventory_and_complete_original_wheel_payloads(
    assembly_runtime: Path,
    assembly_catalog: WheelCatalog,
    assembly_seconds: float,
    tmp_path: Path,
) -> None:
    """The real isolated probe and independent original-payload comparison cover the full set."""
    inventory = inspect_runtime(
        assembly_runtime, assembly_catalog, assembly_seconds, tmp_path / "actual-inventory"
    )
    assert inventory.packages == {wheel.name: wheel.version for wheel in assembly_catalog.wheels}
    bindings = verify_installed_wheels(assembly_runtime, assembly_catalog, inventory)
    assert set(bindings) == {wheel.path.name for wheel in assembly_catalog.wheels}
    assert all(count > 0 for count in bindings.values())
    assert set(inventory.records) == set(inventory.record_sha256) == set(inventory.packages)


@pytest.mark.parametrize("deadline", [0.0, -1.0, math.nan, math.inf, -math.inf])
def test_invalid_probe_deadline_refuses_before_start(tmp_path: Path, deadline: float) -> None:
    """A refused deadline cannot create a probe receipt or execute an interpreter."""
    runtime = tmp_path / "missing-runtime"
    receipts = tmp_path / "probe"
    declaration = WheelCatalog("3.12.16", "linux-x86_64", ())
    with pytest.raises(ValueError, match="finite and positive"):
        inspect_runtime(runtime, declaration, deadline, receipts)
    assert not receipts.exists()


@pytest.mark.parametrize("field", ["python_version", "platform", "packages"])
def test_catalog_must_match_actual_interpreter_and_complete_installed_set(
    assembly_runtime: Path,
    assembly_catalog: WheelCatalog,
    assembly_seconds: float,
    tmp_path: Path,
    field: str,
) -> None:
    """A genuine runtime refuses a wrong patch, wrong platform or incomplete selected closure."""
    if field == "python_version":
        parts = assembly_catalog.python_version.split(".")
        parts[-1] = str(int(parts[-1]) + 1)
        catalog = replace(assembly_catalog, python_version=".".join(parts))
    elif field == "platform":
        catalog = replace(assembly_catalog, platform="linux-incompatible-platform")
    else:
        catalog = replace(
            assembly_catalog,
            wheels=tuple(wheel for wheel in assembly_catalog.wheels if wheel.name != "pip"),
        )
        assert len(catalog.wheels) + 1 == len(assembly_catalog.wheels)
    reason = "distribution set differs" if field == "packages" else "runtime identity differs"
    with pytest.raises(ValueError, match=reason):
        inspect_runtime(assembly_runtime, catalog, assembly_seconds, tmp_path / "actual-probe")


def test_genuine_foreign_platform_wheel_refused_by_actual_interpreter_tags(
    assembly_runtime: Path,
    assembly_catalog: WheelCatalog,
    assembly_incompatible_catalog: WheelCatalog,
    assembly_seconds: float,
    tmp_path: Path,
) -> None:
    """An unchanged published foreign-platform archive cannot replace the selected native wheel."""
    foreign = assembly_incompatible_catalog.wheels[0]
    original = next(wheel for wheel in assembly_catalog.wheels if wheel.name == foreign.name)
    assert original.version == foreign.version
    assert not original.tags.intersection(foreign.tags)
    catalog = replace(
        assembly_catalog,
        wheels=tuple(
            foreign if wheel.name == foreign.name else wheel for wheel in assembly_catalog.wheels
        ),
    )
    with pytest.raises(ValueError, match="cannot install " + foreign.path.name):
        inspect_runtime(assembly_runtime, catalog, assembly_seconds, tmp_path / "foreign-abi-probe")


def test_original_archive_comparison_detects_installed_payload_change(
    assembly_runtime: Path,
    assembly_catalog: WheelCatalog,
    assembly_initial_inventory: RuntimeInventory,
    assembly_file_change: Callable[[Path, bytes | None], AbstractContextManager[None]],
) -> None:
    """Independent member hashes refuse a changed actual installed wheel payload."""
    member = Path(assembly_initial_inventory.records["pip"]).parent / "WHEEL"
    original = member.read_bytes()
    with (
        assembly_file_change(member, original + b"changed\n"),
        pytest.raises(ValueError, match="installed wheel payload differs"),
    ):
        verify_installed_wheels(assembly_runtime, assembly_catalog, assembly_initial_inventory)
    assert member.read_bytes() == original
    assert all(
        count > 0
        for count in verify_installed_wheels(
            assembly_runtime, assembly_catalog, assembly_initial_inventory
        ).values()
    )


def test_actual_compatibility_ignores_ambient_pip_platform_configuration(
    assembly_runtime: Path,
    assembly_catalog: WheelCatalog,
    assembly_seconds: float,
    tmp_path: Path,
) -> None:
    """Real foreign pip settings affect an unisolated command but cannot change admission."""
    config = tmp_path / "foreign-pip.conf"
    config.write_text("[global]\nplatform = win_amd64\n")
    selected = {"PIP_CONFIG_FILE": str(config), "PIP_PLATFORM": "win_amd64"}
    previous = {key: os.environ.get(key) for key in selected}
    try:
        os.environ.update(selected)
        python = (
            assembly_runtime
            / "bin"
            / ("python" + ".".join(assembly_catalog.python_version.split(".")[:2]))
        )
        changed = run_assembly_command(
            [str(python), "-I", "-B", "-m", "pip", "debug", "--verbose"],
            assembly_seconds,
            tmp_path / "actual-foreign-settings",
        )
        assert b"win_amd64" in changed.stdout
        inventory = inspect_runtime(
            assembly_runtime, assembly_catalog, assembly_seconds, tmp_path / "isolated-admission"
        )
        assert inventory.packages == {
            wheel.name: wheel.version for wheel in assembly_catalog.wheels
        }
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def test_captured_reports_decode_to_the_live_inventory(
    assembly_runtime: Path,
    assembly_catalog: WheelCatalog,
    assembly_initial_inventory: RuntimeInventory,
    captured_runtime_reports: tuple[bytes, bytes],
) -> None:
    """The public decoder retains every field of the real supervised inventory."""
    observed = parse_runtime_inventory(
        assembly_runtime, assembly_catalog, *captured_runtime_reports
    )
    assert observed == assembly_initial_inventory
    assert len(observed.packages) == len(assembly_catalog.wheels)
    assert observed.record_sha256 == assembly_initial_inventory.record_sha256


def test_captured_report_requires_an_object(
    assembly_runtime: Path,
    assembly_catalog: WheelCatalog,
    captured_runtime_reports: tuple[bytes, bytes],
) -> None:
    """A captured report wrapped in a new array protocol cannot pass as an inventory."""
    raw, tags = captured_runtime_reports
    replacement = json.dumps([json.loads(raw)]).encode()
    with pytest.raises(ValueError, match="did not return an object"):
        parse_runtime_inventory(assembly_runtime, assembly_catalog, replacement, tags)


@pytest.mark.parametrize("field", ["packages", "scheme", "records", "record_sha256"])
@pytest.mark.parametrize("fault", ["array", "non-string-value"])
def test_captured_report_mapping_schema_refuses_drift(
    assembly_runtime: Path,
    assembly_catalog: WheelCatalog,
    captured_runtime_reports: tuple[bytes, bytes],
    field: str,
    fault: str,
) -> None:
    """Real report fields cannot silently change from string mappings to arrays or numbers."""
    raw, tags = captured_runtime_reports
    report: dict[str, object] = json.loads(raw)
    original = report[field]
    assert isinstance(original, dict)
    if fault == "array":
        report[field] = list(original.items())
    else:
        original[next(iter(original))] = 1
    with pytest.raises(ValueError, match="probe mapping is invalid"):
        parse_runtime_inventory(
            assembly_runtime, assembly_catalog, json.dumps(report).encode(), tags
        )


@pytest.mark.parametrize("fault", ["missing-key", "outside-prefix"])
def test_captured_scheme_requires_all_confined_destinations(
    assembly_runtime: Path,
    assembly_catalog: WheelCatalog,
    captured_runtime_reports: tuple[bytes, bytes],
    tmp_path: Path,
    fault: str,
) -> None:
    """A reported scheme must contain every declared destination inside the actual prefix."""
    raw, tags = captured_runtime_reports
    report: dict[str, object] = json.loads(raw)
    scheme = report["scheme"]
    assert isinstance(scheme, dict)
    if fault == "missing-key":
        scheme.pop("headers")
    else:
        scheme["headers"] = str(tmp_path)
    with pytest.raises(ValueError, match="installation scheme is invalid or escapes"):
        parse_runtime_inventory(
            assembly_runtime, assembly_catalog, json.dumps(report).encode(), tags
        )


@pytest.mark.parametrize("count", [True, 0, -1, 1.5, "1", None])
def test_captured_record_counts_require_positive_integers(
    assembly_runtime: Path,
    assembly_catalog: WheelCatalog,
    captured_runtime_reports: tuple[bytes, bytes],
    count: object,
) -> None:
    """Boolean, empty and non-integral counters cannot describe checked RECORD rows."""
    raw, tags = captured_runtime_reports
    report: dict[str, object] = json.loads(raw)
    counts = report["record_entries"]
    assert isinstance(counts, dict)
    counts["pip"] = count
    with pytest.raises(ValueError, match="RECORD counts are invalid"):
        parse_runtime_inventory(
            assembly_runtime, assembly_catalog, json.dumps(report).encode(), tags
        )


def test_captured_record_counts_require_a_mapping(
    assembly_runtime: Path,
    assembly_catalog: WheelCatalog,
    captured_runtime_reports: tuple[bytes, bytes],
) -> None:
    """A list of actual counters does not satisfy the named-record protocol."""
    raw, tags = captured_runtime_reports
    report: dict[str, object] = json.loads(raw)
    counts = report["record_entries"]
    assert isinstance(counts, dict)
    report["record_entries"] = list(counts.items())
    with pytest.raises(ValueError, match="RECORD counts are invalid"):
        parse_runtime_inventory(
            assembly_runtime, assembly_catalog, json.dumps(report).encode(), tags
        )


@pytest.mark.parametrize("field", ["records", "record_sha256", "record_entries"])
def test_captured_record_bindings_cover_every_installed_distribution(
    assembly_runtime: Path,
    assembly_catalog: WheelCatalog,
    captured_runtime_reports: tuple[bytes, bytes],
    field: str,
) -> None:
    """Losing one real record binding cannot qualify the otherwise complete package set."""
    raw, tags = captured_runtime_reports
    report: dict[str, object] = json.loads(raw)
    mapping = report[field]
    assert isinstance(mapping, dict)
    mapping.pop("pip")
    with pytest.raises(ValueError, match="do not cover the whole inventory"):
        parse_runtime_inventory(
            assembly_runtime, assembly_catalog, json.dumps(report).encode(), tags
        )


@pytest.mark.parametrize("fault", ["missing", "duplicate", "zero", "wrong-count", "invalid-tag"])
def test_captured_pip_tag_protocol_refuses_drift(
    assembly_runtime: Path,
    assembly_catalog: WheelCatalog,
    captured_runtime_reports: tuple[bytes, bytes],
    fault: str,
) -> None:
    """The actual pip diagnostic must retain its single counted compatibility declaration."""
    raw, tags = captured_runtime_reports
    lines = tags.decode("utf-8").splitlines()
    index = next(i for i, line in enumerate(lines) if line.startswith("Compatible tags: "))
    count = int(lines[index].rsplit(" ", 1)[1])
    if fault == "missing":
        lines.pop(index)
    elif fault == "duplicate":
        lines.append(lines[index])
    elif fault == "zero":
        lines[index] = "Compatible tags: 0"
    elif fault == "wrong-count":
        lines[index] = "Compatible tags: " + str(count + 1)
    else:
        lines[index + 1] = "  invalidtag"
    replacement = ("\n".join(lines) + "\n").encode()
    with pytest.raises(ValueError):
        parse_runtime_inventory(assembly_runtime, assembly_catalog, raw, replacement)


@pytest.mark.parametrize("fault", ["missing", "external-link"])
def test_selected_runtime_executable_must_remain_internal(
    assembly_runtime: Path,
    assembly_catalog: WheelCatalog,
    assembly_initial_inventory: RuntimeInventory,
    assembly_file_change: Callable[[Path, bytes | None], AbstractContextManager[None]],
    assembly_seconds: float,
    tmp_path: Path,
    fault: str,
) -> None:
    """A missing or externally linked real interpreter is refused before any child starts."""
    executable = assembly_initial_inventory.executable
    original_inode = executable.stat().st_ino
    with assembly_file_change(executable, None):
        if fault == "external-link":
            executable.symlink_to(Path(sys.executable).resolve())
        with pytest.raises(ValueError, match="must contain its own selected Python"):
            inspect_runtime(
                assembly_runtime, assembly_catalog, assembly_seconds, tmp_path / "refused"
            )
        assert not (tmp_path / "refused").exists()
    assert executable.stat().st_ino == original_inode


def test_actual_data_script_wheels_bind_python_and_native_commands(
    assembly_runtime: Path,
    assembly_runtime_with_data_scripts: tuple[WheelCatalog, RuntimeInventory],
    assembly_catalog: WheelCatalog,
    assembly_seconds: float,
    assembly_file_change: Callable[[Path, bytes | None], AbstractContextManager[None]],
    tmp_path: Path,
) -> None:
    """Original Docutils and Ninja scripts execute, compare and refuse actual installed drift."""
    catalog, inventory = assembly_runtime_with_data_scripts
    assert len(catalog.wheels) == len(assembly_catalog.wheels) + 2
    bindings = verify_installed_wheels(assembly_runtime, catalog, inventory)
    assert set(bindings) == {wheel.path.name for wheel in catalog.wheels}
    scripts = Path(inventory.scheme["scripts"])
    for command, expected in [("rst2html.py", b"0.20.1"), ("ninja", b"1.13.2")]:
        result = run_assembly_command(
            [str(scripts / command), "--version"], assembly_seconds, tmp_path / command
        )
        assert expected in result.stdout + result.stderr
    python_script = scripts / "rst2html.py"
    original = python_script.read_bytes()
    header, separator, body = original.partition(b"\n")
    assert separator == b"\n"
    for changed in [b"#!/usr/bin/python\n" + body, header, header + b"\n" + body + b"changed\n"]:
        with (
            assembly_file_change(python_script, changed),
            pytest.raises(ValueError, match="Python data script differs"),
        ):
            verify_installed_wheels(assembly_runtime, catalog, inventory)
        assert python_script.read_bytes() == original
    native_script = scripts / "ninja"
    original_native = native_script.read_bytes()
    assert original_native.startswith(b"\x7fELF")
    with (
        assembly_file_change(native_script, original_native + b"changed"),
        pytest.raises(ValueError, match="installed wheel payload differs"),
    ):
        verify_installed_wheels(assembly_runtime, catalog, inventory)
    assert native_script.read_bytes() == original_native
    assert verify_installed_wheels(assembly_runtime, catalog, inventory) == bindings


def test_original_cache_disposition_and_retained_bytes_are_enforced(
    assembly_runtime: Path,
    assembly_catalog: WheelCatalog,
    assembly_initial_inventory: RuntimeInventory,
) -> None:
    """Exact upstream caches may be absent or byte-exact, and PyPA assembly requires absence."""
    wheel = next(wheel for wheel in assembly_catalog.wheels if wheel.generated_bytecode)
    cache = wheel.generated_bytecode[0]
    target = Path(assembly_initial_inventory.scheme["platlib"], cache.path)
    assert not target.exists() and not target.is_symlink()
    missing_dirs: list[Path] = []
    parent = target.parent
    while not parent.exists():
        missing_dirs.append(parent)
        parent = parent.parent
    for directory in reversed(missing_dirs):
        directory.mkdir()
    try:
        with ZipFile(wheel.path) as archive, target.open("xb") as stream:
            stream.write(archive.read(cache.path))
        bindings = verify_installed_wheels(
            assembly_runtime, assembly_catalog, assembly_initial_inventory
        )
        assert set(bindings) == {wheel.path.name for wheel in assembly_catalog.wheels}
        with pytest.raises(ValueError, match="generated bytecode must be absent"):
            verify_installed_wheels(
                assembly_runtime,
                assembly_catalog,
                assembly_initial_inventory,
                require_pypa_cache_absence=True,
            )
        target.write_bytes(target.read_bytes() + b"changed")
        with pytest.raises(ValueError, match="installed wheel payload differs"):
            verify_installed_wheels(assembly_runtime, assembly_catalog, assembly_initial_inventory)
    finally:
        target.unlink(missing_ok=True)
        for directory in missing_dirs:
            directory.rmdir()
    changed_catalog = replace(
        assembly_catalog,
        wheels=tuple(
            replace(item, generated_bytecode=()) if item == wheel else item
            for item in assembly_catalog.wheels
        ),
    )
    with pytest.raises(ValueError, match="bytecode disposition differs"):
        verify_installed_wheels(assembly_runtime, changed_catalog, assembly_initial_inventory)
    assert not target.exists()
    assert all(
        verify_installed_wheels(
            assembly_runtime,
            assembly_catalog,
            assembly_initial_inventory,
            require_pypa_cache_absence=True,
        ).values()
    )


def test_original_data_member_requires_its_declared_scheme(
    assembly_runtime: Path,
    assembly_catalog: WheelCatalog,
    assembly_initial_inventory: RuntimeInventory,
) -> None:
    """The real SymPy data member cannot be verified against an incomplete destination scheme."""
    scheme = dict(assembly_initial_inventory.scheme)
    scheme.pop("data")
    with pytest.raises(ValueError, match="invalid data scheme"):
        verify_installed_wheels(
            assembly_runtime, assembly_catalog, replace(assembly_initial_inventory, scheme=scheme)
        )


def test_original_member_refuses_an_installed_link_outside_runtime(
    assembly_runtime: Path,
    assembly_catalog: WheelCatalog,
    assembly_initial_inventory: RuntimeInventory,
    assembly_file_change: Callable[[Path, bytes | None], AbstractContextManager[None]],
    tmp_path: Path,
) -> None:
    """A genuine installed member may not resolve to an otherwise identical external file."""
    target = Path(assembly_initial_inventory.records["pip"]).parent / "WHEEL"
    external = tmp_path / "outside-wheel-metadata"
    external.write_bytes(target.read_bytes())
    original_inode = target.stat().st_ino
    with assembly_file_change(target, None):
        target.symlink_to(external)
        with pytest.raises(ValueError, match="installed wheel member escapes"):
            verify_installed_wheels(assembly_runtime, assembly_catalog, assembly_initial_inventory)
    assert target.stat().st_ino == original_inode


def test_original_cache_refuses_destination_outside_runtime(
    assembly_runtime: Path,
    assembly_catalog: WheelCatalog,
    assembly_initial_inventory: RuntimeInventory,
    tmp_path: Path,
) -> None:
    """Even a qualified absent upstream cache must remain confined to the actual prefix."""
    wheel = next(wheel for wheel in assembly_catalog.wheels if wheel.generated_bytecode)
    target = Path(assembly_initial_inventory.scheme["platlib"], wheel.generated_bytecode[0].path)
    assert not target.exists() and not target.is_symlink()
    missing_dirs: list[Path] = []
    parent = target.parent
    while not parent.exists():
        missing_dirs.append(parent)
        parent = parent.parent
    for directory in reversed(missing_dirs):
        directory.mkdir()
    try:
        target.symlink_to(tmp_path / "external-cache")
        with pytest.raises(ValueError, match="cache destination escapes"):
            verify_installed_wheels(assembly_runtime, assembly_catalog, assembly_initial_inventory)
    finally:
        target.unlink(missing_ok=True)
        for directory in missing_dirs:
            directory.rmdir()
