# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Relative wheel destination and generated launcher contracts

"""Install real pip and exercise its moved scripts with an explicit external Python.

This destination contract does not qualify a self-contained desktop runtime.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import stat
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING
from zipfile import ZipFile

import pytest
from installer import install
from installer.records import RecordEntry
from installer.sources import WheelFile
from installer.utils import Scheme

from tools.studio_desktop_wheel_catalog import (
    SharedWheelPayload,
    WheelCatalog,
    plan_shared_payloads,
)
from tools.studio_desktop_wheel_destination import RelocatableWheelDestination

if TYPE_CHECKING:
    from installer.scripts import LauncherKind, ScriptSection


@pytest.fixture(scope="module")
def installed_pip(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, dict[str, str]]:
    """Install the declared genuine pip wheel, then move its complete test payload."""
    configured = os.environ.get("SC_NEUROCORE_STUDIO_ASSEMBLY_INPUTS")
    if not configured:
        pytest.fail("SC_NEUROCORE_STUDIO_ASSEMBLY_INPUTS must name the acquired wheel directory.")
    wheel = Path(configured) / "pip-26.2.2-py3-none-any.whl"
    with wheel.open("rb") as stream:
        assert hashlib.file_digest(stream, "sha256").hexdigest() == (
            "fd7f4dfa8190e014013b9f08f3e973f4e4c26ffd179a204895533a7254813a70"
        )
    root = tmp_path_factory.mktemp("actual-pip-destination")
    original = root / "initial installation"
    scripts = original / "bin"
    scripts.mkdir(parents=True)
    (scripts / "python3.12").symlink_to(sys.executable)
    scheme = {
        "purelib": str(original / "lib"),
        "platlib": str(original / "lib"),
        "scripts": str(scripts),
        "headers": str(original / "include"),
        "data": str(original),
    }
    destination = RelocatableWheelDestination(scheme, str(scripts / "python3.12"), "posix")
    with WheelFile.open(wheel) as source:
        install(source, destination, {"INSTALLER": b"installer\n"})
    moved = root / "moved ' installation with spaces"
    original.rename(moved)
    assert not original.exists()
    return moved, {key: value.replace(str(original), str(moved)) for key, value in scheme.items()}


def _pip_version(
    script: Path, prefix: Path, *, explicit_python: bool = False
) -> subprocess.CompletedProcess[str]:
    environment = {**os.environ, "PYTHONPATH": str(prefix / "lib"), "PYTHONDONTWRITEBYTECODE": "1"}
    command = ([sys.executable] if explicit_python else []) + [str(script), "--version"]
    receipts = (
        prefix.parent / "commands" / (script.name + ("-explicit-python" if explicit_python else ""))
    )
    receipts.mkdir(parents=True)
    (receipts / "command.json").write_text(
        json.dumps({"argv": command, "cwd": "/", "PYTHONPATH": str(prefix / "lib")}) + "\n"
    )
    result = subprocess.run(
        command,
        cwd="/",
        env=environment,
        capture_output=True,
        text=True,
        timeout=float(os.environ["SC_NEUROCORE_STUDIO_ASSEMBLY_COMMAND_SECONDS"]),
    )
    (receipts / "stdout").write_text(result.stdout)
    (receipts / "stderr").write_text(result.stderr)
    (receipts / "returncode").write_text(str(result.returncode) + "\n")
    return result


@pytest.fixture(scope="module")
def shared_qt_plan(
    assembly_catalog: WheelCatalog, tmp_path_factory: pytest.TempPathFactory
) -> dict[Path, SharedWheelPayload]:
    """Derive shared ownership from every genuine original archive, without a runtime copy."""
    root = tmp_path_factory.mktemp("real-shared-qt-plan")
    scheme = {
        "purelib": str(root / "lib"),
        "platlib": str(root / "lib"),
        "scripts": str(root / "bin"),
        "headers": str(root / "include"),
        "data": str(root),
    }
    result = plan_shared_payloads(assembly_catalog, scheme)
    assert len(result) == 4
    return result


@pytest.fixture
def shared_qt_destination(
    shared_qt_plan: dict[Path, SharedWheelPayload], tmp_path: Path
) -> RelocatableWheelDestination:
    """Bind the real shared declarations to a fresh ordinary filesystem destination."""
    original_library = next(iter(shared_qt_plan)).parents[1]
    library = tmp_path / "lib"
    scheme = {
        "purelib": str(library),
        "platlib": str(library),
        "scripts": str(tmp_path / "bin"),
        "headers": str(tmp_path / "include"),
        "data": str(tmp_path),
    }
    return RelocatableWheelDestination(
        scheme,
        str(tmp_path / "bin/python3.12"),
        "posix",
        shared_payloads={
            library / path.relative_to(original_library): item
            for path, item in shared_qt_plan.items()
        },
    )


@pytest.fixture(scope="module")
def shared_qt_bytes(assembly_catalog: WheelCatalog) -> dict[str, bytes]:
    """Read the four original shared payloads directly from the admitted Qt archive."""
    wheel = next(w for w in assembly_catalog.wheels if w.name == "pyside6-addons")
    with ZipFile(wheel.path) as source:
        return {
            name: source.read(name)
            for name in [
                "PySide6/__init__.py",
                "PySide6/_config.py",
                "PySide6/_git_pyside_version.py",
                "PySide6/py.typed",
            ]
        }


def test_real_shared_qt_payloads_keep_one_inode_and_each_owners_record(
    shared_qt_destination: RelocatableWheelDestination, shared_qt_bytes: dict[str, bytes]
) -> None:
    """Three genuine owners receive byte-valid records without rewriting any shared file."""
    destination = shared_qt_destination
    for member, raw in shared_qt_bytes.items():
        target = Path(destination.scheme_dict["platlib"], member)
        first = destination.write_file(Scheme("platlib"), member, io.BytesIO(raw), False)
        inode = (target.stat().st_dev, target.stat().st_ino)
        assert len(destination.shared_payloads[target].archives) == 3
        for _ in destination.shared_payloads[target].archives[1:]:
            record = destination.write_file(Scheme("platlib"), member, io.BytesIO(raw), False)
            assert record == first
            assert record.validate_stream(io.BytesIO(target.read_bytes()))
            assert (target.stat().st_dev, target.stat().st_ino) == inode
        assert target.read_bytes() == raw
        assert target.stat().st_nlink == 1


@pytest.mark.parametrize("origin", ["pre-existing", "other-destination-instance"])
def test_admitted_shared_path_cannot_claim_an_existing_file(
    shared_qt_destination: RelocatableWheelDestination,
    shared_qt_bytes: dict[str, bytes],
    origin: str,
) -> None:
    """Even byte-identical existing files need current-instance creation provenance."""
    destination = shared_qt_destination
    member = "PySide6/__init__.py"
    raw = shared_qt_bytes[member]
    target = Path(destination.scheme_dict["platlib"], member)
    if origin == "pre-existing":
        target.parent.mkdir(parents=True)
        target.write_bytes(raw)
    else:
        destination.write_file(Scheme("platlib"), member, io.BytesIO(raw), False)
        destination = RelocatableWheelDestination(
            destination.scheme_dict,
            destination.interpreter,
            "posix",
            shared_payloads=destination.shared_payloads,
        )
    inode = target.stat().st_ino
    with pytest.raises(FileExistsError):
        destination.write_file(Scheme("platlib"), member, io.BytesIO(raw), False)
    assert target.read_bytes() == raw and target.stat().st_ino == inode


@pytest.mark.parametrize(
    "fault", ["incoming-bytes", "installed-bytes", "replaced-inode", "hardlink", "mode"]
)
def test_shared_payload_fault_is_refused_without_rewriting(
    shared_qt_destination: RelocatableWheelDestination,
    shared_qt_bytes: dict[str, bytes],
    fault: str,
) -> None:
    """Real source, destination, inode, link-count and mode faults cannot inherit ownership."""
    destination = shared_qt_destination
    member = "PySide6/__init__.py"
    raw = shared_qt_bytes[member]
    target = Path(destination.scheme_dict["platlib"], member)
    destination.write_file(Scheme("platlib"), member, io.BytesIO(raw), False)
    incoming = raw
    if fault == "incoming-bytes":
        incoming += b"changed\n"
    elif fault == "installed-bytes":
        target.write_bytes(raw + b"changed\n")
    elif fault == "replaced-inode":
        target.rename(target.with_suffix(".original"))
        target.write_bytes(raw)
    elif fault == "hardlink":
        os.link(target, target.with_suffix(".other-owner"))
    else:
        target.chmod(stat.S_IMODE(target.stat().st_mode) | 0o111)
    before, inode = target.read_bytes(), target.stat().st_ino
    with pytest.raises(ValueError, match="shared wheel payload"):
        destination.write_file(Scheme("platlib"), member, io.BytesIO(incoming), False)
    assert target.read_bytes() == before and target.stat().st_ino == inode


def test_first_shared_payload_mismatch_does_not_admit_the_created_file(
    shared_qt_destination: RelocatableWheelDestination, shared_qt_bytes: dict[str, bytes]
) -> None:
    """An incorrect first stream leaves unqualified staging and cannot be retried as sharing."""
    destination = shared_qt_destination
    member = "PySide6/__init__.py"
    raw = shared_qt_bytes[member]
    with pytest.raises(ValueError, match="new shared wheel payload differs"):
        destination.write_file(Scheme("platlib"), member, io.BytesIO(raw + b"changed\n"), False)
    with pytest.raises(FileExistsError):
        destination.write_file(Scheme("platlib"), member, io.BytesIO(raw), False)


def test_overwrite_option_and_shared_mode_mismatch_refuse_before_writing(
    shared_qt_destination: RelocatableWheelDestination, shared_qt_bytes: dict[str, bytes]
) -> None:
    """Neither a broad overwrite option nor a conflicting executable declaration is accepted."""
    destination = shared_qt_destination
    member = "PySide6/__init__.py"
    target = Path(destination.scheme_dict["platlib"], member)
    destination.overwrite_existing = True
    with pytest.raises(ValueError, match="never permits overwriting"):
        destination.write_file(
            Scheme("platlib"), member, io.BytesIO(shared_qt_bytes[member]), False
        )
    destination.overwrite_existing = False
    with pytest.raises(ValueError, match="executable declaration differs"):
        destination.write_file(Scheme("platlib"), member, io.BytesIO(shared_qt_bytes[member]), True)
    assert not target.exists()


def test_symbolic_shared_destination_is_refused_without_touching_its_target(
    shared_qt_destination: RelocatableWheelDestination, shared_qt_bytes: dict[str, bytes]
) -> None:
    """An admitted member path cannot redirect a write through an actual symbolic link."""
    destination = shared_qt_destination
    member = "PySide6/__init__.py"
    raw = shared_qt_bytes[member]
    target = Path(destination.scheme_dict["platlib"], member)
    target.parent.mkdir(parents=True)
    original = target.parent / "original.py"
    original.write_bytes(raw)
    target.symlink_to(original)
    with pytest.raises(ValueError, match="symbolic link"):
        destination.write_file(Scheme("platlib"), member, io.BytesIO(raw), False)
    assert target.is_symlink() and original.read_bytes() == raw


def test_current_instance_cannot_share_an_unplanned_original_member(
    shared_qt_destination: RelocatableWheelDestination, assembly_catalog: WheelCatalog
) -> None:
    """Creating a genuine pip member grants no sharing permission outside the selected plan."""
    destination = shared_qt_destination
    wheel = next(w for w in assembly_catalog.wheels if w.name == "pip")
    with WheelFile.open(wheel.path) as source:
        member = source.dist_info_dir + "/WHEEL"
        raw = source.read_dist_info("WHEEL").encode("utf-8")
    target = Path(destination.scheme_dict["platlib"], member)
    first = destination.write_file(Scheme("platlib"), member, io.BytesIO(raw), False)
    inode = target.stat().st_ino
    with pytest.raises(FileExistsError):
        destination.write_file(Scheme("platlib"), member, io.BytesIO(raw), False)
    assert first.validate_stream(io.BytesIO(target.read_bytes()))
    assert target.stat().st_ino == inode


def test_complete_installer_record_after_movement(
    installed_pip: tuple[Path, dict[str, str]],
) -> None:
    """Every normal installer-generated record still matches its moved installed file."""
    prefix, _ = installed_pip
    record = prefix / "lib/pip-26.2.2.dist-info/RECORD"
    rows = list(csv.reader(io.StringIO(record.read_text())))
    assert len(rows) == 481
    for row in rows:
        entry = RecordEntry.from_elements(*row)
        target = record.parent.parent / entry.path
        assert target.resolve().is_relative_to(prefix.resolve())
        with target.open("rb") as stream:
            assert entry.validate_stream(stream), entry.path
    assert not (record.parent / "direct_url.json").exists()


@pytest.mark.parametrize("name", ["pip", "pip3"])
def test_original_console_commands_after_movement(
    installed_pip: tuple[Path, dict[str, str]], name: str
) -> None:
    """Original declared pip entry points execute the moved payload from outside it."""
    prefix, _ = installed_pip
    script = prefix / "bin" / name
    assert str(prefix.parent / "initial installation").encode() not in script.read_bytes()
    result = _pip_version(script, prefix)
    assert result.returncode == 0, result.stderr
    assert "pip 26.2.2 from " + str(prefix / "lib/pip") in result.stdout


def test_external_script_symlink_uses_its_real_destination(
    installed_pip: tuple[Path, dict[str, str]], tmp_path: Path
) -> None:
    """A symlink outside the installation dispatches to Python beside the real script."""
    prefix, _ = installed_pip
    link = tmp_path / "external pip link"
    link.symlink_to(prefix / "bin/pip")
    result = _pip_version(link, prefix)
    assert result.returncode == 0, result.stderr
    assert str(prefix / "lib/pip") in result.stdout


@pytest.mark.parametrize("section", ["console", "gui"])
def test_generated_declared_entry_point_has_valid_record_and_executes(
    installed_pip: tuple[Path, dict[str, str]], section: ScriptSection
) -> None:
    """Both wheel entry-point sections retain executable modes and actual byte records."""
    prefix, scheme = installed_pip
    destination = RelocatableWheelDestination(scheme, str(prefix / "bin/python3.12"), "posix")
    name = "declared-pip-" + section
    entry = destination.write_script(name, "pip._internal.cli.main", "main", section)
    script = prefix / "bin" / name
    assert script.stat().st_mode & stat.S_IXUSR
    with script.open("rb") as stream:
        assert entry.validate_stream(stream)
    result = _pip_version(script, prefix)
    assert result.returncode == 0, result.stderr
    assert str(prefix / "lib/pip") in result.stdout


@pytest.mark.parametrize("kind", ["python", "pythonw"])
@pytest.mark.parametrize("encoding", ["utf-8", "iso-8859-1"])
def test_generic_python_scripts_retain_nonascii_source_encoding(
    installed_pip: tuple[Path, dict[str, str]], kind: str, encoding: str
) -> None:
    """Real pip runs behind generic shebangs with UTF8 or Latin1 module literals."""
    prefix, scheme = installed_pip
    destination = RelocatableWheelDestination(scheme, str(prefix / "bin/python3.12"), "posix")
    body = (prefix / "bin/pip").read_bytes().partition(b"\n")[2]
    source = (
        f"#!{kind}\r\n# coding: {encoding}\n".encode("ascii")
        + '__doc__ = "Source encoding witness: café"\n'.encode(encoding)
        + body
    )
    name = kind + "-" + encoding
    entry = destination.write_file(Scheme("scripts"), name, io.BytesIO(source), True)
    script = prefix / "bin" / name
    with script.open("rb") as stream:
        assert entry.validate_stream(stream)
    result = _pip_version(script, prefix)
    assert result.returncode == 0, result.stderr
    assert str(prefix / "lib/pip") in result.stdout
    if encoding == "iso-8859-1":
        wrong = prefix / "bin" / (name + "-wrong-encoding")
        wrong.write_bytes(
            script.read_bytes().replace(b"# coding: iso-8859-1", b"# coding: utf-8", 1)
        )
        wrong.chmod(script.stat().st_mode)
        refused = _pip_version(wrong, prefix)
        assert refused.returncode != 0
        assert "SyntaxError" in refused.stderr


@pytest.mark.parametrize("scheme_name", ["purelib", "scripts"])
def test_ordinary_payload_keeps_original_bytes(
    installed_pip: tuple[Path, dict[str, str]], scheme_name: str
) -> None:
    """A non-Python-script wheel member is preserved by the normal destination write."""
    prefix, scheme = installed_pip
    destination = RelocatableWheelDestination(scheme, str(prefix / "bin/python3.12"), "posix")
    payload = (prefix / "lib/pip-26.2.2.dist-info/WHEEL").read_bytes()
    entry = destination.write_file(
        Scheme(scheme_name), "wheel-metadata-copy", io.BytesIO(payload), False
    )
    target = Path(scheme[scheme_name], entry.path)
    assert target.read_bytes() == payload
    with target.open("rb") as stream:
        assert entry.validate_stream(stream)


def test_generic_script_preserves_docstring_and_future_import(
    installed_pip: tuple[Path, dict[str, str]],
) -> None:
    """A valid generic script can retain its module docstring before a future import."""
    prefix, scheme = installed_pip
    destination = RelocatableWheelDestination(scheme, str(prefix / "bin/python3.12"), "posix")
    body = (prefix / "bin/pip").read_bytes().partition(b"\n")[2]
    source = (
        b'#!python\n# coding: utf-8\n"""Original module documentation."""\n'
        b'from __future__ import annotations\nassert __doc__ == "Original module documentation."\n'
        + body
    )
    compile(source, "original-generic-pip", "exec")
    entry = destination.write_file(Scheme("scripts"), "pip-with-future", io.BytesIO(source), True)
    script = prefix / "bin/pip-with-future"
    with script.open("rb") as stream:
        assert entry.validate_stream(stream)
    result = _pip_version(script, prefix)
    assert result.returncode == 0, result.stderr
    assert str(prefix / "lib/pip") in result.stdout
    direct = _pip_version(script, prefix, explicit_python=True)
    assert direct.returncode == 0, direct.stderr
    assert str(prefix / "lib/pip") in direct.stdout


@pytest.mark.parametrize(
    ("interpreter", "platform"),
    [("python3.12", "win-amd64"), ("unrecognised-interpreter", "posix")],
)
def test_unsupported_destination_refuses_before_writing_launcher(
    installed_pip: tuple[Path, dict[str, str]], interpreter: str, platform: LauncherKind
) -> None:
    """A declared unsupported destination cannot create a misleading runnable script."""
    prefix, scheme = installed_pip
    destination = RelocatableWheelDestination(scheme, interpreter, platform)
    name = "refused-" + platform + "-" + interpreter
    with pytest.raises(ValueError):
        destination.write_script(name, "pip._internal.cli.main", "main", "console")
    assert not (prefix / "bin" / name).exists()
