# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Real standalone desktop interpreter probe contracts

"""Execute the actual probe under the supplied standalone runtime and control interpreter."""

from __future__ import annotations

import csv
import io
import json
import shutil
import subprocess
import sys
from collections.abc import Callable
from contextlib import AbstractContextManager
from pathlib import Path

import pytest

from tools.studio_desktop_assembly_process import run_assembly_command
from tools.studio_desktop_runtime_inventory import RuntimeInventory
from tools.studio_desktop_wheel_catalog import WheelCatalog

PROBE = Path(__file__).resolve().parents[3] / "tools/studio_desktop_probe_runtime.py"


@pytest.mark.parametrize("interface", ["cli", "api"])
def test_actual_standalone_probe_reports_complete_inventory(
    assembly_runtime: Path,
    assembly_catalog: WheelCatalog,
    assembly_initial_inventory: RuntimeInventory,
    assembly_seconds: float,
    assembly_probe_command: Callable[[Path, str], list[str]],
    tmp_path: Path,
    interface: str,
) -> None:
    """The real isolated interpreter reports every selected distribution and checked record."""
    result = run_assembly_command(
        assembly_probe_command(assembly_initial_inventory.executable, interface),
        assembly_seconds,
        tmp_path / "actual-probe",
    )
    value = json.loads(result.stdout)
    direct = run_assembly_command(
        [str(assembly_initial_inventory.executable), "-I", "-B", str(PROBE)],
        assembly_seconds,
        tmp_path / "direct-probe",
    )
    assert result.stdout == direct.stdout
    assert value["record_sha256"] == assembly_initial_inventory.record_sha256
    assert value["prefix"] == str(assembly_runtime)
    assert value["python_version"] == assembly_catalog.python_version
    assert value["platform"] == assembly_catalog.platform
    expected = {wheel.name: wheel.version for wheel in assembly_catalog.wheels}
    assert value["packages"] == expected
    assert set(value["records"]) == set(value["record_entries"]) == set(expected)
    assert set(value["record_sha256"]) == set(expected)
    assert all(count > 0 for count in value["record_entries"].values())
    assert all(
        Path(path).resolve().is_relative_to(assembly_runtime) for path in value["records"].values()
    )


def test_virtual_environment_refuses_before_inventory(
    assembly_seconds: float,
    assembly_probe_command: Callable[[Path, str], list[str]],
    tmp_path: Path,
) -> None:
    """The existing external-base control environment cannot qualify as standalone Python."""
    assert Path(sys.prefix).resolve() != Path(sys.base_prefix).resolve(), (
        "The test profile must use its separate complete control virtual environment."
    )
    with pytest.raises(subprocess.CalledProcessError) as raised:
        run_assembly_command(
            assembly_probe_command(Path(sys.executable), "cli"),
            assembly_seconds,
            tmp_path / "external-base",
        )
    assert raised.value.returncode == 1
    assert b"depends on an external Python prefix" in raised.value.stderr
    assert raised.value.stdout == b""


@pytest.mark.parametrize(
    ("change", "reason"),
    [
        ("content", b"Installed RECORD content differs"),
        ("size", b"Installed RECORD content differs"),
        ("duplicate", b"RECORD repeats a path"),
        ("missing-record-row", b"exactly one RECORD"),
        ("missing-payload", b"FileNotFoundError"),
        ("missing-record", b"lacks a RECORD"),
    ],
)
def test_real_installed_record_failures_are_refused_and_restored(
    assembly_initial_inventory: RuntimeInventory,
    assembly_seconds: float,
    assembly_file_change: Callable[[Path, bytes | None], AbstractContextManager[None]],
    assembly_probe_command: Callable[[Path, str], list[str]],
    tmp_path: Path,
    change: str,
    reason: bytes,
) -> None:
    """A real installed record or file fault cannot pass the normal isolated probe."""
    record = Path(assembly_initial_inventory.records["pip"])
    original = record.read_bytes()
    rows = list(csv.reader(io.StringIO(original.decode("utf-8"))))
    wheel_row = next(row for row in rows if row[0].endswith(".dist-info/WHEEL"))
    wheel_file = record.parent.parent / wheel_row[0]
    target = record
    replacement: bytes | None = original
    if change == "content":
        target, replacement = wheel_file, wheel_file.read_bytes() + b"changed\n"
    elif change == "size":
        wheel_row[2] = str(int(wheel_row[2]) + 1)
    elif change == "duplicate":
        rows.append(wheel_row.copy())
    elif change == "missing-record-row":
        rows = [row for row in rows if not row[0].endswith(".dist-info/RECORD")]
    elif change == "missing-payload":
        target, replacement = wheel_file, None
    elif change == "missing-record":
        replacement = None
    if change in {"size", "duplicate", "missing-record-row"}:
        stream = io.StringIO()
        csv.writer(stream).writerows(rows)
        replacement = stream.getvalue().encode("utf-8")
    target_original = target.read_bytes()
    with assembly_file_change(target, replacement):
        with pytest.raises(subprocess.CalledProcessError) as raised:
            run_assembly_command(
                assembly_probe_command(assembly_initial_inventory.executable, "cli"),
                assembly_seconds,
                tmp_path / "refused-probe",
            )
        assert raised.value.returncode == 1
        assert reason in raised.value.stderr
        assert raised.value.stdout == b""
    assert target.read_bytes() == target_original
    assert record.read_bytes() == original
    run_assembly_command(
        assembly_probe_command(assembly_initial_inventory.executable, "cli"),
        assembly_seconds,
        tmp_path / "restored-probe",
    )


def test_real_scheme_link_outside_runtime_is_refused_and_restored(
    assembly_runtime: Path,
    assembly_initial_inventory: RuntimeInventory,
    assembly_probe_command: Callable[[Path, str], list[str]],
    assembly_seconds: float,
    tmp_path: Path,
) -> None:
    """A real include-directory link cannot move an installed scheme outside its prefix."""
    include = Path(assembly_initial_inventory.scheme["headers"])
    original = include.with_name(include.name + ".assembly-original")
    assert include.is_dir() and not include.is_symlink() and not original.exists()
    outside = tmp_path / "outside-include"
    outside.mkdir()
    include.rename(original)
    try:
        include.symlink_to(outside, target_is_directory=True)
        with pytest.raises(subprocess.CalledProcessError) as raised:
            run_assembly_command(
                assembly_probe_command(assembly_initial_inventory.executable, "cli"),
                assembly_seconds,
                tmp_path / "scheme-refused",
            )
        assert b"installation scheme escapes" in raised.value.stderr
        assert raised.value.stdout == b""
    finally:
        if include.is_symlink():
            include.unlink()
        original.rename(include)
    assert include.resolve().is_relative_to(assembly_runtime)


def test_real_record_link_outside_runtime_is_refused_and_restored(
    assembly_initial_inventory: RuntimeInventory,
    assembly_file_change: Callable[[Path, bytes | None], AbstractContextManager[None]],
    assembly_probe_command: Callable[[Path, str], list[str]],
    assembly_seconds: float,
    tmp_path: Path,
) -> None:
    """An unchanged original RECORD outside the prefix cannot qualify the installation."""
    record = Path(assembly_initial_inventory.records["pip"])
    original = record.read_bytes()
    outside = tmp_path / "external-record.csv"
    outside.write_bytes(original)
    with assembly_file_change(record, None):
        record.symlink_to(outside)
        with pytest.raises(subprocess.CalledProcessError) as raised:
            run_assembly_command(
                assembly_probe_command(assembly_initial_inventory.executable, "cli"),
                assembly_seconds,
                tmp_path / "record-link-refused",
            )
        assert b"installed RECORD escapes" in raised.value.stderr
        assert raised.value.stdout == b""
    assert record.read_bytes() == original and not record.is_symlink()


@pytest.mark.parametrize("fault", ["directory", "external-file"])
def test_real_payload_destination_must_be_an_internal_regular_file(
    assembly_initial_inventory: RuntimeInventory,
    assembly_file_change: Callable[[Path, bytes | None], AbstractContextManager[None]],
    assembly_probe_command: Callable[[Path, str], list[str]],
    assembly_seconds: float,
    tmp_path: Path,
    fault: str,
) -> None:
    """Neither a directory nor an external payload link may satisfy an installed record row."""
    record = Path(assembly_initial_inventory.records["pip"])
    member = record.parent / "WHEEL"
    original = member.read_bytes()
    outside = tmp_path / "external-wheel-metadata"
    outside.write_bytes(original)
    with assembly_file_change(member, None):
        if fault == "directory":
            member.mkdir()
        else:
            member.symlink_to(outside)
        try:
            with pytest.raises(subprocess.CalledProcessError) as raised:
                run_assembly_command(
                    assembly_probe_command(assembly_initial_inventory.executable, "cli"),
                    assembly_seconds,
                    tmp_path / "payload-destination-refused",
                )
            assert b"record path escapes or is not a file" in raised.value.stderr
            assert raised.value.stdout == b""
        finally:
            if fault == "directory":
                member.rmdir()
    assert member.read_bytes() == original and not member.is_symlink()


def test_duplicate_genuine_distribution_metadata_is_refused(
    assembly_initial_inventory: RuntimeInventory,
    assembly_probe_command: Callable[[Path, str], list[str]],
    assembly_seconds: float,
    tmp_path: Path,
) -> None:
    """Two real copies of the same installed metadata cannot become a unique inventory."""
    original = Path(assembly_initial_inventory.records["pip"]).parent
    duplicate = original.with_name("pip.assembly-duplicate.dist-info")
    assert not duplicate.exists()
    shutil.copytree(original, duplicate)
    try:
        with pytest.raises(subprocess.CalledProcessError) as raised:
            run_assembly_command(
                assembly_probe_command(assembly_initial_inventory.executable, "cli"),
                assembly_seconds,
                tmp_path / "duplicate-refused",
            )
        assert b"distribution is duplicate or lacks a RECORD" in raised.value.stderr
        assert raised.value.stdout == b""
    finally:
        shutil.rmtree(duplicate)
