# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Exact selected desktop wheel catalog contracts

"""Read genuine published wheels and refuse invalid catalog or archive bindings."""

from __future__ import annotations

import base64
import csv
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import zipfile
from copy import copy
from dataclasses import replace
from pathlib import Path

import pytest

from tools.studio_desktop_assembly_process import run_assembly_command
from tools.studio_desktop_wheel_catalog import (
    WheelCatalog,
    create_catalog,
    plan_shared_payloads,
    read_catalog,
    verify_wheel_input,
)


@pytest.fixture
def catalog_path(tmp_path: Path) -> Path:
    """Bind the declared CI-produced published pip and locked Pint wheel archives."""
    configured = os.environ.get("SC_NEUROCORE_STUDIO_ASSEMBLY_INPUTS")
    if not configured:
        pytest.fail("SC_NEUROCORE_STUDIO_ASSEMBLY_INPUTS must name the acquired wheel directory.")
    inputs = Path(configured)
    declarations = []
    for filename, checksum in [
        (
            "pip-26.2.2-py3-none-any.whl",
            "fd7f4dfa8190e014013b9f08f3e973f4e4c26ffd179a204895533a7254813a70",
        ),
        (
            "pint-0.26.2-py3-none-any.whl",
            "c94124cdb19baa1a84bb65c8bb9a620a9636b9b9cd80f32f30ad7ff795ea2eb8",
        ),
    ]:
        source = inputs / filename
        with source.open("rb") as stream:
            assert hashlib.file_digest(stream, "sha256").hexdigest() == checksum
        target = tmp_path / filename
        shutil.copyfile(source, target)
        declarations.append(
            {"filename": filename, "sha256": checksum, "size_bytes": target.stat().st_size}
        )
    path = tmp_path / "catalog.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "studio.desktop.wheel-catalog.v1",
                "python_version": "3.12.16",
                "platform": "linux-x86_64",
                "wheels": declarations,
            }
        )
    )
    return path


def test_published_wheel_identities_and_original_bytes(catalog_path: Path) -> None:
    """Actual archives agree with filename identities, selected hashes and wheel RECORDs."""
    catalog = read_catalog(catalog_path)
    assert catalog.python_version == "3.12.16"
    assert catalog.platform == "linux-x86_64"
    assert [(wheel.name, wheel.version) for wheel in catalog.wheels] == [
        ("pip", "26.2.2"),
        ("pint", "0.26.2"),
    ]
    for wheel in catalog.wheels:
        assert wheel.path.parent == catalog_path.parent
        assert wheel.tags == frozenset({"py3-none-any"})
        verify_wheel_input(wheel)


def test_original_requires_python_rejects_incompatible_selection(catalog_path: Path) -> None:
    """A universal wheel is still refused when its genuine Python floor is not met."""
    declaration = json.loads(catalog_path.read_text())
    declaration["python_version"] = "3.11.15"
    catalog_path.write_text(json.dumps(declaration))
    with pytest.raises(ValueError, match="does not admit the selected Python"):
        read_catalog(catalog_path)
    declaration["wheels"] = declaration["wheels"][:1]
    catalog_path.write_text(json.dumps(declaration))
    assert read_catalog(catalog_path).wheels[0].name == "pip"


@pytest.mark.parametrize(
    ("field", "value", "reason"),
    [
        ("schema_version", "old", "schema is unsupported"),
        ("python_version", "3.10.16", "at least 3.11"),
        ("python_version", "3.12", "full stable"),
        ("python_version", "3.12.16rc1", "full stable"),
        ("python_version", 31216, "full stable"),
        ("platform", "win-amd64", "Linux platform"),
        ("platform", None, "Linux platform"),
        ("wheels", [], "complete nonempty"),
        ("wheels", {}, "complete nonempty"),
    ],
)
def test_invalid_runtime_and_closure_declarations(
    catalog_path: Path, field: str, value: object, reason: str
) -> None:
    """The public reader refuses unsupported runtime, schema and missing input sets."""
    declaration = json.loads(catalog_path.read_text())
    declaration[field] = value
    catalog_path.write_text(json.dumps(declaration))
    with pytest.raises(ValueError, match=reason):
        read_catalog(catalog_path)


@pytest.mark.parametrize("declaration", [None, [], {}, {"schema_version": "old"}])
def test_incomplete_catalog_fields(catalog_path: Path, declaration: object) -> None:
    """Neither a non-object nor a partial object can represent the selected closure."""
    catalog_path.write_text(json.dumps(declaration))
    with pytest.raises(ValueError, match="fields do not match"):
        read_catalog(catalog_path)


def test_duplicate_json_fields_are_not_silently_overwritten(catalog_path: Path) -> None:
    """Duplicate selected values are refused before JSON can discard either value."""
    original = catalog_path.read_text()
    catalog_path.write_text('{"python_version":"3.11.15",' + original[1:])
    with pytest.raises(ValueError, match="repeats field 'python_version'"):
        read_catalog(catalog_path)


def test_duplicate_distribution_identity_is_refused(catalog_path: Path) -> None:
    """A wheel cannot be selected twice even when each individual archive is valid."""
    declaration = json.loads(catalog_path.read_text())
    declaration["wheels"].append(declaration["wheels"][0])
    catalog_path.write_text(json.dumps(declaration))
    with pytest.raises(ValueError, match="duplicate distribution identities"):
        read_catalog(catalog_path)


@pytest.mark.parametrize(
    ("field", "value", "reason"),
    [
        ("filename", "../pip.whl", "single archive basename"),
        ("filename", "directory\\pip.whl", "single archive basename"),
        ("filename", "", "single archive basename"),
        ("filename", 1, "single archive basename"),
        ("sha256", "0" * 63, "complete lowercase SHA-256"),
        ("sha256", "A" * 64, "complete lowercase SHA-256"),
        ("sha256", None, "complete lowercase SHA-256"),
        ("sha256", "0" * 64, "SHA-256 differs"),
        ("size_bytes", True, "positive integer"),
        ("size_bytes", 0, "positive integer"),
        ("size_bytes", -1, "positive integer"),
        ("size_bytes", 1.0, "positive integer"),
        ("size_bytes", 1, "byte length differs"),
    ],
)
def test_invalid_wheel_declarations(
    catalog_path: Path, field: str, value: object, reason: str
) -> None:
    """The archive basename and complete byte identity are mandatory admission inputs."""
    declaration = json.loads(catalog_path.read_text())
    declaration["wheels"][0][field] = value
    catalog_path.write_text(json.dumps(declaration))
    with pytest.raises(ValueError, match=reason):
        read_catalog(catalog_path)


@pytest.mark.parametrize("entry", [None, [], {}, {"filename": "pip.whl"}])
def test_incomplete_wheel_fields(catalog_path: Path, entry: object) -> None:
    """A missing archive binding cannot be inferred from an incomplete wheel row."""
    declaration = json.loads(catalog_path.read_text())
    declaration["wheels"][0] = entry
    catalog_path.write_text(json.dumps(declaration))
    with pytest.raises(ValueError, match="requires filename, sha256 and size_bytes only"):
        read_catalog(catalog_path)


@pytest.mark.parametrize("change", ["symlink", "missing", "bytes", "size"])
def test_verified_archive_is_rechecked_before_installation(catalog_path: Path, change: str) -> None:
    """The public recheck detects substitutions after the original successful admission."""
    wheel = read_catalog(catalog_path).wheels[0]
    if change == "symlink":
        moved = wheel.path.with_suffix(".original")
        wheel.path.rename(moved)
        wheel.path.symlink_to(moved)
        reason = "regular file"
    elif change == "missing":
        wheel.path.unlink()
        reason = "regular file"
    elif change == "bytes":
        with wheel.path.open("r+b") as stream:
            byte = stream.read(1)
            stream.seek(0)
            stream.write(bytes([byte[0] ^ 1]))
        reason = "SHA-256 differs"
    else:
        with wheel.path.open("ab") as stream:
            stream.write(b"changed")
        reason = "byte length differs"
    with pytest.raises(ValueError, match=reason):
        verify_wheel_input(wheel)


def test_catalog_creation_binds_actual_selected_archives(catalog_path: Path) -> None:
    """A new deterministic catalog contains every original archive and no derived wheel."""
    output = catalog_path.with_name("created.json")
    catalog = create_catalog(catalog_path.parent, output, "3.12.16", "linux-x86_64")
    assert catalog == read_catalog(output)
    assert {wheel.name: wheel.version for wheel in catalog.wheels} == {
        "pip": "26.2.2",
        "pint": "0.26.2",
    }
    first_bytes = output.read_bytes()
    repeated = output.with_name("repeat.json")
    assert create_catalog(catalog_path.parent, repeated, "3.12.16", "linux-x86_64") == catalog
    assert repeated.read_bytes() == first_bytes
    original = read_catalog(catalog_path)
    assert {wheel.sha256 for wheel in original.wheels} == {wheel.sha256 for wheel in catalog.wheels}


def test_creation_preserves_existing_catalog(catalog_path: Path) -> None:
    """An existing original declaration is never overwritten by a new acquisition snapshot."""
    original = catalog_path.read_bytes()
    with pytest.raises(FileExistsError):
        create_catalog(catalog_path.parent, catalog_path, "3.12.16", "linux-x86_64")
    assert catalog_path.read_bytes() == original


def test_creation_refuses_catalog_outside_wheel_directory(catalog_path: Path) -> None:
    """Catalog placement must preserve the reader's archive-relative path contract."""
    output = catalog_path.parent.parent / "outside.json"
    with pytest.raises(ValueError, match="must be adjacent"):
        create_catalog(catalog_path.parent, output, "3.12.16", "linux-x86_64")
    assert not output.exists()


def test_creation_refuses_incompatible_original_python_requirement(catalog_path: Path) -> None:
    """The new producer uses the same actual Pint Python floor before writing its output."""
    output = catalog_path.with_name("incompatible.json")
    with pytest.raises(ValueError, match="does not admit the selected Python"):
        create_catalog(catalog_path.parent, output, "3.11.15", "linux-x86_64")
    assert not output.exists()


@pytest.mark.parametrize("kind", ["symlink", "directory"])
def test_creation_refuses_nonregular_wheel_inputs(catalog_path: Path, kind: str) -> None:
    """A wheel-shaped path must be an original regular archive before it is hashed."""
    source = catalog_path.parent / "pip-26.2.2-py3-none-any.whl"
    if kind == "symlink":
        original = source.with_suffix(".original")
        source.rename(original)
        source.symlink_to(original)
    else:
        source.unlink()
        source.mkdir()
    output = catalog_path.with_name("refused.json")
    with pytest.raises(ValueError, match="regular file"):
        create_catalog(catalog_path.parent, output, "3.12.16", "linux-x86_64")
    assert not output.exists()


def test_creation_refuses_an_empty_selection(tmp_path: Path) -> None:
    """A new directory without selected archives cannot yield a usable catalog."""
    output = tmp_path / "empty.json"
    with pytest.raises(ValueError, match="complete nonempty"):
        create_catalog(tmp_path, output, "3.12.16", "linux-x86_64")
    assert not output.exists()


def test_public_creation_command_and_refusal(catalog_path: Path, assembly_seconds: float) -> None:
    """The actual tools CLI creates the verified file and returns an invalid-input refusal."""
    output = catalog_path.with_name("cli-created.json")
    arguments = [
        sys.executable,
        "-B",
        "-m",
        "tools.studio_desktop_wheel_catalog",
        "--directory",
        str(catalog_path.parent),
        "--output",
        str(output),
        "--python-version",
        "3.12.16",
        "--platform",
        "linux-x86_64",
    ]
    result = run_assembly_command(
        arguments, assembly_seconds, catalog_path.parent / "creation-command"
    )
    assert result.stdout.decode("utf-8").strip() == str(output.resolve())
    assert len(read_catalog(output).wheels) == 2
    arguments[arguments.index("--python-version") + 1] = "3.11.15"
    arguments[arguments.index("--output") + 1] = str(catalog_path.with_name("cli-refused.json"))
    with pytest.raises(subprocess.CalledProcessError) as raised:
        run_assembly_command(arguments, assembly_seconds, catalog_path.parent / "refusal-command")
    assert raised.value.returncode == 1
    assert b"does not admit the selected Python" in raised.value.stderr
    assert not catalog_path.with_name("cli-refused.json").exists()


def test_exact_upstream_generated_bytecode_disposition(assembly_catalog: WheelCatalog) -> None:
    """Only the exact original Qt cache and source record pairs are qualified."""
    wheel = next(w for w in assembly_catalog.wheels if w.name == "pyside6-essentials")
    assert wheel.sha256 == "16f4b5e41daf49235ae048458243da9efe5f3d8040f9cc5aa454ce69893d3bd9"
    assert verify_wheel_input(wheel) == wheel.generated_bytecode
    assert [
        (entry.path, entry.size_bytes, entry.source_path, entry.source_size_bytes)
        for entry in wheel.generated_bytecode
    ] == [
        (
            "PySide6/support/__pycache__/__init__.cpython-312.pyc",
            315,
            "PySide6/support/__init__.py",
            223,
        ),
        (
            "PySide6/support/__pycache__/deprecated.cpython-312.pyc",
            271,
            "PySide6/support/deprecated.py",
            605,
        ),
    ]
    assert all(not other.generated_bytecode for other in assembly_catalog.wheels if other != wheel)


def test_changed_archive_identity_does_not_inherit_cache_disposition(
    assembly_catalog: WheelCatalog, tmp_path: Path
) -> None:
    """Changing only a genuine ZIP container comment cannot inherit its reviewed archive pin."""
    original = next(w for w in assembly_catalog.wheels if w.generated_bytecode)
    source_hash = hashlib.sha256(original.path.read_bytes()).hexdigest()
    changed = tmp_path / original.path.name
    shutil.copyfile(original.path, changed)
    with zipfile.ZipFile(changed, "a") as packed:
        packed.comment = b"Deliberate archive identity counterexample; original members unchanged."
    with changed.open("rb") as stream:
        checksum = hashlib.file_digest(stream, "sha256").hexdigest()
    assert checksum != source_hash
    catalog = tmp_path / "catalog.json"
    catalog.write_text(
        json.dumps(
            {
                "schema_version": "studio.desktop.wheel-catalog.v1",
                "python_version": assembly_catalog.python_version,
                "platform": assembly_catalog.platform,
                "wheels": [
                    {
                        "filename": changed.name,
                        "sha256": checksum,
                        "size_bytes": changed.stat().st_size,
                    }
                ],
            }
        )
    )
    with pytest.raises(ValueError, match="unqualified upstream bytecode"):
        read_catalog(catalog)
    assert hashlib.sha256(original.path.read_bytes()).hexdigest() == source_hash


def test_cache_disposition_requires_exact_source_companion_record(
    assembly_catalog: WheelCatalog, assembly_catalog_path: Path, tmp_path: Path
) -> None:
    """A real malformed policy cannot claim a different source companion for unchanged Qt bytes."""
    policy = Path(__file__).resolve().parents[3] / "tools/studio_desktop_generated_bytecode.toml"
    original = policy.read_bytes()
    (tmp_path / "original-policy.toml").write_bytes(original)
    malformed = original.replace(
        b"sha256=VEzYGbtVB2wirZUgh9Dc70FNxk9vCaSsZa0e4CFrVpU",
        b"sha256=AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
        1,
    )
    assert malformed != original
    (tmp_path / "invalid-source-policy.toml").write_bytes(malformed)
    try:
        policy.write_bytes(malformed)
        with pytest.raises(ValueError, match="exact RECORD-verified source companion"):
            read_catalog(assembly_catalog_path)
    finally:
        policy.write_bytes(original)
    assert policy.read_bytes() == original
    assert read_catalog(assembly_catalog_path) == assembly_catalog


def test_real_complete_closure_declares_exact_shared_qt_library_owners(
    assembly_catalog: WheelCatalog, tmp_path: Path
) -> None:
    """All genuine selected archives agree on four shared Qt files before any target write."""
    scheme = {
        "purelib": str(tmp_path / "lib"),
        "platlib": str(tmp_path / "lib"),
        "scripts": str(tmp_path / "bin"),
        "headers": str(tmp_path / "include"),
        "data": str(tmp_path),
    }
    planned = plan_shared_payloads(assembly_catalog, scheme)
    assert {str(path.relative_to(tmp_path / "lib")) for path in planned} == {
        "PySide6/__init__.py",
        "PySide6/_config.py",
        "PySide6/_git_pyside_version.py",
        "PySide6/py.typed",
    }
    for item in planned.values():
        assert item.record_hash.startswith("sha256=") and item.size_bytes > 0
        assert item.executable is False
        assert len(item.archives) == 3
        assert all(name.startswith("pyside6_") for name in item.archives)
    assert not (tmp_path / "lib").exists()


def test_real_archive_mode_disagreement_refuses_shared_ownership_before_writing(
    assembly_catalog: WheelCatalog, tmp_path: Path
) -> None:
    """A genuine Qt container with one changed mode cannot share otherwise unchanged payloads."""
    original = next(w for w in assembly_catalog.wheels if w.name == "pyside6-addons")
    changed = tmp_path / original.path.name
    shutil.copyfile(original.path, changed)
    with zipfile.ZipFile(changed, "a") as packed:
        packed.getinfo("PySide6/__init__.py").external_attr |= 0o111 << 16
        packed.comment = b"Deliberate shared-mode counterexample; original payloads unchanged."
    with changed.open("rb") as stream:
        checksum = hashlib.file_digest(stream, "sha256").hexdigest()
    assert checksum != original.sha256
    variant = replace(original, path=changed, sha256=checksum, size_bytes=changed.stat().st_size)
    assert verify_wheel_input(variant) == ()
    catalog = replace(
        assembly_catalog,
        wheels=tuple(variant if w == original else w for w in assembly_catalog.wheels),
    )
    scheme = {
        "purelib": str(tmp_path / "lib"),
        "platlib": str(tmp_path / "lib"),
        "scripts": str(tmp_path / "bin"),
        "headers": str(tmp_path / "include"),
        "data": str(tmp_path),
    }
    with pytest.raises(ValueError, match="owners disagree on a shared payload"):
        plan_shared_payloads(catalog, scheme)
    assert not (tmp_path / "lib").exists()
    with original.path.open("rb") as stream:
        assert hashlib.file_digest(stream, "sha256").hexdigest() == original.sha256


@pytest.mark.parametrize("fault", ["layout", "duplicate-python-requirement"])
def test_genuine_archive_metadata_faults_are_refused(
    catalog_path: Path, tmp_path: Path, fault: str
) -> None:
    """Located metadata faults in full pip copies retain every other original member and record."""
    original = read_catalog(catalog_path).wheels[0]
    changed = tmp_path / fault / original.path.name
    changed.parent.mkdir()
    with zipfile.ZipFile(original.path) as source:
        member = next(
            name
            for name in source.namelist()
            if name.endswith(".dist-info/" + ("WHEEL" if fault == "layout" else "METADATA"))
        )
        raw = source.read(member)
        if fault == "layout":
            assert b"Wheel-Version: 1.0" in raw
            replacement = raw.replace(b"Wheel-Version: 1.0", b"Wheel-Version: 2.0", 1)
            reason = "supported wheel layout"
        else:
            header, separator, body = raw.partition(b"\n\n")
            assert separator and header.count(b"Requires-Python:") == 1
            replacement = header + b"\nRequires-Python: >=3.11\n\n" + body
            reason = "repeats its Requires-Python"
        record = next(name for name in source.namelist() if name.endswith(".dist-info/RECORD"))
        rows = list(csv.reader(io.StringIO(source.read(record).decode("utf-8"))))
        checksum = base64.urlsafe_b64encode(hashlib.sha256(replacement).digest()).rstrip(b"=")
        matching = [row for row in rows if row[0] == member]
        assert len(matching) == 1
        matching[0][1:] = ["sha256=" + checksum.decode("ascii"), str(len(replacement))]
        buffer = io.StringIO(newline="")
        csv.writer(buffer).writerows(rows)
        replacements = {member: replacement, record: buffer.getvalue().encode("utf-8")}
        with zipfile.ZipFile(changed, "w") as destination:
            for entry in source.infolist():
                destination.writestr(
                    copy(entry), replacements.get(entry.filename, source.read(entry))
                )
        with zipfile.ZipFile(changed) as destination:
            assert destination.namelist() == source.namelist()
            assert all(
                destination.read(name) == source.read(name)
                for name in source.namelist()
                if name not in replacements
            )
    output = changed.parent / "refused.json"
    with pytest.raises(ValueError, match=reason):
        create_catalog(changed.parent, output, "3.12.16", "linux-x86_64")
    assert not output.exists()
    assert hashlib.sha256(original.path.read_bytes()).hexdigest() == original.sha256


def test_original_metadata_refuses_a_different_selected_identity(catalog_path: Path) -> None:
    """The unchanged real pip archive cannot acquire a different declared distribution name."""
    original = read_catalog(catalog_path).wheels[0]
    with pytest.raises(ValueError, match="metadata does not match"):
        verify_wheel_input(replace(original, name="different-distribution"))
    assert hashlib.sha256(original.path.read_bytes()).hexdigest() == original.sha256


@pytest.mark.parametrize(
    ("fault", "reason"),
    [
        ("schema", "policy schema is invalid"),
        ("no-members", "has no exact members"),
        ("member-set", "member set differs"),
        ("incomplete-member", "policy member is incomplete"),
        ("invalid-member", "invalid record fields"),
        ("cache-record", "cache RECORD hash or byte length differs"),
    ],
)
def test_original_qt_archive_requires_its_complete_exact_cache_policy(
    assembly_catalog: WheelCatalog, tmp_path: Path, fault: str, reason: str
) -> None:
    """Malformed real policy files are refused through the public original-archive verifier."""
    wheel = next(wheel for wheel in assembly_catalog.wheels if wheel.generated_bytecode)
    policy = Path(__file__).resolve().parents[3] / "tools/studio_desktop_generated_bytecode.toml"
    original = policy.read_bytes()
    if fault == "schema":
        changed = original.replace(b"source_url = ", b"origin_url = ", 1)
    elif fault == "no-members":
        changed = original.partition(b"[[generated_bytecode]]")[0] + b"generated_bytecode = []\n"
    elif fault == "member-set":
        header, first, _ = original.split(b"[[generated_bytecode]]")
        changed = header + b"[[generated_bytecode]]" + first
    elif fault == "incomplete-member":
        changed = original.replace(b"source_path = ", b"source_file = ", 1)
    elif fault == "invalid-member":
        changed = original.replace(b"size_bytes = 315", b"size_bytes = true", 1)
    else:
        changed = original.replace(
            b"sha256=1u7uDwSk7ky3Bet7a08vF5ZFLxLSvhUj4Kh_SOBjzmg",
            b"sha256=AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
            1,
        )
    assert changed != original
    (tmp_path / "original-policy.toml").write_bytes(original)
    (tmp_path / "refused-policy.toml").write_bytes(changed)
    try:
        policy.write_bytes(changed)
        with pytest.raises(ValueError, match=reason):
            verify_wheel_input(wheel)
    finally:
        policy.write_bytes(original)
    assert policy.read_bytes() == original
    assert hashlib.sha256(wheel.path.read_bytes()).hexdigest() == wheel.sha256


@pytest.mark.parametrize("fault", ["cache-disposition", "missing-data-scheme", "external-link"])
def test_original_closure_refuses_invalid_ownership_destinations(
    assembly_catalog: WheelCatalog, tmp_path: Path, fault: str
) -> None:
    """Planning original full-closure ownership refuses changed cache or destination bindings."""
    scheme = {
        "purelib": str(tmp_path / "lib"),
        "platlib": str(tmp_path / "lib"),
        "scripts": str(tmp_path / "bin"),
        "headers": str(tmp_path / "include"),
        "data": str(tmp_path),
    }
    catalog = assembly_catalog
    if fault == "cache-disposition":
        catalog = replace(
            catalog,
            wheels=tuple(replace(wheel, generated_bytecode=()) for wheel in catalog.wheels),
        )
        reason = "cache disposition changed"
    elif fault == "missing-data-scheme":
        scheme.pop("data")
        reason = "invalid installation destination"
    else:
        library = tmp_path / "lib"
        library.mkdir()
        external = tmp_path / "outside-library"
        external.mkdir()
        (library / "pip").symlink_to(external, target_is_directory=True)
        reason = "escapes its installation scheme"
    with pytest.raises(ValueError, match=reason):
        plan_shared_payloads(catalog, scheme)
    assert not (tmp_path / "bin").exists()
    assert not (tmp_path / "include").exists()


def test_original_data_scripts_cannot_have_duplicate_owners(
    assembly_catalog: WheelCatalog, tmp_path: Path
) -> None:
    """A typed caller repeating genuine script wheels cannot plan shared script ownership."""
    configured = os.environ.get("SC_NEUROCORE_STUDIO_ASSEMBLY_SCRIPT_CATALOG")
    assert configured, "The normal input producer must provide the genuine script-wheel catalog."
    scripts = read_catalog(Path(configured))
    catalog = replace(
        assembly_catalog,
        wheels=(*assembly_catalog.wheels, *scripts.wheels, *scripts.wheels),
    )
    scheme = {
        "purelib": str(tmp_path / "lib"),
        "platlib": str(tmp_path / "lib"),
        "scripts": str(tmp_path / "bin"),
        "headers": str(tmp_path / "include"),
        "data": str(tmp_path),
    }
    with pytest.raises(ValueError, match="Only original library payloads"):
        plan_shared_payloads(catalog, scheme)
    assert not (tmp_path / "lib").exists()
    assert not (tmp_path / "bin").exists()
