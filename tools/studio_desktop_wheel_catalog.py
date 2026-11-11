# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Exact desktop wheel closure catalog

"""Validate the complete selected wheel inputs before changing a bundled runtime."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections.abc import Mapping
from dataclasses import dataclass, replace
from email.parser import Parser
from pathlib import Path

import tomllib
from installer.sources import WheelFile
from packaging.specifiers import SpecifierSet
from packaging.utils import canonicalize_name, parse_wheel_filename
from packaging.version import Version


@dataclass(frozen=True, slots=True)
class GeneratedBytecode:
    """An exact upstream cache member excluded by the standard installer.

    Attributes
    ----------
    path, record_hash : str
        Original cache member path and its complete wheel RECORD hash.
    size_bytes : int
        Original cache payload byte length.
    source_path, source_record_hash : str
        Required original Python source companion and its complete RECORD hash.
    source_size_bytes : int
        Original source companion byte length.
    """

    path: str
    record_hash: str
    size_bytes: int
    source_path: str
    source_record_hash: str
    source_size_bytes: int


@dataclass(frozen=True, slots=True)
class WheelInput:
    """One complete, verified wheel archive in the selected closure.

    Attributes
    ----------
    path : pathlib.Path
        Regular archive next to its catalog, never a symbolic link.
    name, version : str
        Normalised distribution identity verified against wheel metadata.
    sha256 : str
        Complete archive SHA-256, as declared by the selected source channel.
    size_bytes : int
        Complete archive byte length.
    tags : frozenset of str
        Expanded compatibility tags from the original wheel filename.
    generated_bytecode : tuple of GeneratedBytecode
        Exact qualified upstream caches with RECORD-verified source companions;
        their original bytes remain in the unchanged wheel, never forcibly installed.
    """

    path: Path
    name: str
    version: str
    sha256: str
    size_bytes: int
    tags: frozenset[str]
    generated_bytecode: tuple[GeneratedBytecode, ...] = ()


@dataclass(frozen=True, slots=True)
class WheelCatalog:
    """Complete exact inputs for one embedded Linux Python runtime.

    Attributes
    ----------
    python_version : str
        Selected full stable Python 3 version, at least 3.11.
    platform : str
        Actual embedded interpreter's ``sysconfig.get_platform()`` value.
    wheels : tuple of WheelInput
        Exactly one verified archive per installed distribution.
    """

    python_version: str
    platform: str
    wheels: tuple[WheelInput, ...]


@dataclass(frozen=True, slots=True)
class SharedWheelPayload:
    """An identical regular payload owned by multiple selected wheel archives.

    Attributes
    ----------
    record_hash : str
        Complete original RECORD hash shared by every owner.
    size_bytes : int
        Original byte length shared by every owner.
    executable : bool
        Original executable declaration, equal for every owner.
    archives : tuple of str
        Exact selected archive basenames whose RECORDs own this file.
    """

    record_hash: str
    size_bytes: int
    executable: bool
    archives: tuple[str, ...]


def plan_shared_payloads(
    catalog: WheelCatalog, scheme: Mapping[str, str]
) -> dict[Path, SharedWheelPayload]:
    """Identify byte-identical library ownership before uninstalling any package.

    Parameters
    ----------
    catalog : WheelCatalog
        Complete selected closure; every original archive is rechecked.
    scheme : mapping of str to str
        Actual target installation destinations from its isolated inventory.

    Returns
    -------
    dict of pathlib.Path to SharedWheelPayload
        Confined resolved destinations with multiple identical original owners.
        This plan grants no permission to replace an existing runtime file.

    Raises
    ------
    ValueError
        Original ownership differs in hash, size or mode, a shared script would
        require generation, or a member escapes its declared installation scheme.
    OSError
        An original selected archive cannot be read.
    """
    owners: dict[Path, SharedWheelPayload] = {}
    library_owners: set[Path] = set()
    for wheel in catalog.wheels:
        if verify_wheel_input(wheel) != wheel.generated_bytecode:
            raise ValueError("The upstream cache disposition changed before ownership planning.")
        excluded = {entry.path for entry in wheel.generated_bytecode}
        with WheelFile.open(wheel.path) as source:
            header = Parser().parsestr(source.read_dist_info("WHEEL"))
            root_scheme = "purelib" if header.get("Root-Is-Purelib") == "true" else "platlib"
            for row, stream, executable in source.get_contents():
                stream.close()
                member = row[0]
                if member == source.dist_info_dir + "/RECORD" or member in excluded:
                    continue
                destination, relative = root_scheme, member
                if member.startswith(source.data_dir + "/"):
                    destination, _, relative = member[len(source.data_dir) + 1 :].partition("/")
                if destination not in scheme or not relative:
                    raise ValueError("A selected wheel has an invalid installation destination.")
                base = Path(scheme[destination]).resolve()
                target = (base / relative).resolve()
                if not target.is_relative_to(base):
                    raise ValueError("A selected wheel member escapes its installation scheme.")
                item = SharedWheelPayload(row[1], int(row[2]), executable, (wheel.path.name,))
                previous = owners.get(target)
                if previous is not None:
                    if destination not in {"purelib", "platlib"} or target not in library_owners:
                        raise ValueError(
                            "Only original library payloads may have shared ownership."
                        )
                    if (
                        previous.record_hash != item.record_hash
                        or previous.size_bytes != item.size_bytes
                        or previous.executable != item.executable
                    ):
                        raise ValueError("Selected wheel owners disagree on a shared payload.")
                    item = replace(item, archives=previous.archives + item.archives)
                elif destination in {"purelib", "platlib"}:
                    library_owners.add(target)
                owners[target] = item
    return {path: value for path, value in owners.items() if len(value.archives) > 1}


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    """Reject duplicate JSON fields before ordinary schema validation."""
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"The wheel catalog repeats field {key!r}.")
        result[key] = value
    return result


def verify_wheel_input(wheel: WheelInput) -> tuple[GeneratedBytecode, ...]:
    """Recheck the actual archive and its full original wheel RECORD.

    Parameters
    ----------
    wheel : WheelInput
        Declared exact archive identity.

    Returns
    -------
    tuple of GeneratedBytecode
        Only exact pinned upstream cache members, with original source companion
        records verified. Any unqualified cache member or archive is refused.

    Raises
    ------
    ValueError
        The archive kind, byte length, SHA-256 or original wheel RECORD differs.
    OSError
        The original archive cannot be read.
    """
    if wheel.path.is_symlink() or not wheel.path.is_file():
        raise ValueError("A catalog wheel must be a regular file, not a symbolic link.")
    if wheel.path.stat().st_size != wheel.size_bytes:
        raise ValueError(f"Wheel byte length differs: {wheel.path.name}")
    with wheel.path.open("rb") as stream:
        checksum = hashlib.file_digest(stream, "sha256").hexdigest()
    if checksum != wheel.sha256:
        raise ValueError(f"Wheel SHA-256 differs: {wheel.path.name}")
    with WheelFile.open(wheel.path) as source:
        source.validate_record(validate_contents=True)
        header = Parser().parsestr(source.read_dist_info("WHEEL"))
        wheel_version = header.get("Wheel-Version")
        if (
            not isinstance(wheel_version, str)
            or not wheel_version.startswith("1.")
            or header.get("Root-Is-Purelib") not in {"true", "false"}
        ):
            raise ValueError("The archive does not declare a supported wheel layout.")
        metadata = Parser().parsestr(source.read_dist_info("METADATA"))
        name = metadata.get("Name")
        version = metadata.get("Version")
        if (
            not isinstance(name, str)
            or not isinstance(version, str)
            or len(metadata.get_all("Name", [])) != 1
            or len(metadata.get_all("Version", [])) != 1
            or canonicalize_name(name, validate=True) != wheel.name
            or Version(version) != Version(wheel.version)
        ):
            raise ValueError("Wheel metadata does not match its selected filename identity.")
        records = {row[0]: (row[0], row[1], row[2]) for row, _, _ in source.get_contents()}
        return _generated_bytecode(wheel, records)


def _generated_bytecode(
    wheel: WheelInput, records: dict[str, tuple[str, str, str]]
) -> tuple[GeneratedBytecode, ...]:
    """Admit only the exact reviewed upstream archive and cache/source record pairs."""
    caches = {path for path in records if "__pycache__" in Path(path).parts[:-1]}
    if not caches:
        return ()
    policy: object = tomllib.loads(
        Path(__file__)
        .with_name("studio_desktop_generated_bytecode.toml")
        .read_text(encoding="utf-8")
    )
    if not isinstance(policy, dict) or set(policy) != {
        "schema_version",
        "filename",
        "sha256",
        "size_bytes",
        "source_url",
        "generated_bytecode",
    }:
        raise ValueError("The upstream generated-bytecode policy schema is invalid.")
    if (
        policy["schema_version"] != "studio.desktop.generated-bytecode.v1"
        or policy["filename"] != wheel.path.name
        or policy["sha256"] != wheel.sha256
        or policy["size_bytes"] != wheel.size_bytes
        or not isinstance(policy["source_url"], str)
        or not policy["source_url"].startswith("https://")
    ):
        raise ValueError("A selected wheel contains unqualified upstream bytecode caches.")
    values = policy["generated_bytecode"]
    if not isinstance(values, list) or not values:
        raise ValueError("The upstream generated-bytecode policy has no exact members.")
    qualified = tuple(_bytecode_entry(value) for value in values)
    if len(qualified) != len(caches) or {entry.path for entry in qualified} != caches:
        raise ValueError("The upstream cache member set differs from its exact policy.")
    for entry in qualified:
        if records.get(entry.path) != (entry.path, entry.record_hash, str(entry.size_bytes)):
            raise ValueError("An upstream cache RECORD hash or byte length differs.")
        if records.get(entry.source_path) != (
            entry.source_path,
            entry.source_record_hash,
            str(entry.source_size_bytes),
        ):
            raise ValueError("An upstream cache lacks its exact RECORD-verified source companion.")
    return qualified


def _bytecode_entry(value: object) -> GeneratedBytecode:
    """Validate one complete cache/source pair from the shipped exact-pin policy."""
    if not isinstance(value, dict) or set(value) != {
        "path",
        "record_hash",
        "size_bytes",
        "source_path",
        "source_record_hash",
        "source_size_bytes",
    }:
        raise ValueError("An upstream cache policy member is incomplete.")
    path, checksum, size = value["path"], value["record_hash"], value["size_bytes"]
    source, source_hash, source_size = (
        value["source_path"],
        value["source_record_hash"],
        value["source_size_bytes"],
    )
    if (
        not isinstance(path, str)
        or "__pycache__" not in Path(path).parts[:-1]
        or not isinstance(checksum, str)
        or not checksum.startswith("sha256=")
        or not isinstance(source, str)
        or not source.endswith(".py")
        or "__pycache__" in Path(source).parts[:-1]
        or not isinstance(source_hash, str)
        or not source_hash.startswith("sha256=")
        or not isinstance(size, int)
        or isinstance(size, bool)
        or size < 0
        or not isinstance(source_size, int)
        or isinstance(source_size, bool)
        or source_size < 0
    ):
        raise ValueError("An upstream cache policy member has invalid record fields.")
    return GeneratedBytecode(path, checksum, size, source, source_hash, source_size)


def _read_wheel(value: object, directory: Path, python: Version) -> WheelInput:
    """Parse and verify one exact wheel declaration relative to its catalog."""
    if not isinstance(value, dict) or set(value) != {
        "filename",
        "sha256",
        "size_bytes",
    }:
        raise ValueError("A wheel entry requires filename, sha256 and size_bytes only.")
    filename = value["filename"]
    checksum = value["sha256"]
    size = value["size_bytes"]
    if (
        not isinstance(filename, str)
        or not filename
        or "/" in filename
        or "\\" in filename
        or Path(filename).name != filename
    ):
        raise ValueError("A wheel filename must be a single archive basename.")
    if not isinstance(checksum, str) or re.fullmatch(r"[0-9a-f]{64}", checksum) is None:
        raise ValueError("A wheel requires its complete lowercase SHA-256.")
    if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
        raise ValueError("A wheel byte length must be a positive integer.")
    name, version, _, tags = parse_wheel_filename(filename)
    name = canonicalize_name(str(name), validate=True)
    wheel = WheelInput(
        directory / filename,
        str(name),
        str(version),
        checksum,
        size,
        frozenset(str(tag) for tag in tags),
    )
    generated = verify_wheel_input(wheel)
    with WheelFile.open(wheel.path) as source:
        metadata = Parser().parsestr(source.read_dist_info("METADATA"))
        requires_python = metadata.get("Requires-Python")
        if len(metadata.get_all("Requires-Python", [])) > 1:
            raise ValueError("A wheel repeats its Requires-Python declaration.")
        if requires_python is not None and (
            not isinstance(requires_python, str)
            or not SpecifierSet(requires_python).contains(python, prereleases=False)
        ):
            raise ValueError("A wheel does not admit the selected Python version.")
    return replace(wheel, generated_bytecode=generated)


def read_catalog(path: Path) -> WheelCatalog:
    """Read the exact selected closure and verify every archive before mutation.

    Parameters
    ----------
    path : pathlib.Path
        UTF-8 catalog with archives in the same directory. The schema is
        ``studio.desktop.wheel-catalog.v1`` with Python version, platform and
        a nonempty wheel list; unknown or repeated fields are refused.

    Returns
    -------
    WheelCatalog
        Exact complete input declarations, with original archive RECORDs checked.

    Raises
    ------
    ValueError
        The schema, runtime declaration or wheel closure is invalid.
    OSError
        The catalog or a selected archive cannot be read.
    """
    value: object = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique_object)
    return _catalog(value, path.resolve().parent)


def _catalog(value: object, directory: Path) -> WheelCatalog:
    """Validate the same schema and original archives for reading and creation."""
    if not isinstance(value, dict) or set(value) != {
        "schema_version",
        "python_version",
        "platform",
        "wheels",
    }:
        raise ValueError("The wheel catalog fields do not match the supported schema.")
    if value["schema_version"] != "studio.desktop.wheel-catalog.v1":
        raise ValueError("The desktop wheel catalog schema is unsupported.")
    python = value["python_version"]
    platform = value["platform"]
    wheels = value["wheels"]
    if not isinstance(python, str) or re.fullmatch(r"3\.[0-9]+\.[0-9]+", python) is None:
        raise ValueError("The catalog requires a full stable Python 3 version.")
    version = Version(python)
    if version < Version("3.11"):
        raise ValueError("The catalog requires a stable Python version at least 3.11.")
    if not isinstance(platform, str) or not platform.startswith("linux-"):
        raise ValueError("This desktop wheel producer requires a Linux platform.")
    if not isinstance(wheels, list) or not wheels:
        raise ValueError("The catalog must declare the complete nonempty wheel closure.")
    selected = tuple(_read_wheel(item, directory, version) for item in wheels)
    names = [wheel.name for wheel in selected]
    if len(set(names)) != len(names):
        raise ValueError("The catalog contains duplicate distribution identities.")
    return WheelCatalog(python, platform, selected)


def create_catalog(directory: Path, path: Path, python_version: str, platform: str) -> WheelCatalog:
    """Bind every already selected wheel archive to a new adjacent catalog.

    Parameters
    ----------
    directory : pathlib.Path
        Existing wheel directory populated by the hashlocked input acquisition.
        This command does not resolve, download or establish source provenance.
    path : pathlib.Path
        New catalog path in that same directory. Existing files are refused.
    python_version, platform : str
        Actual standalone interpreter version and Linux installation platform.

    Returns
    -------
    WheelCatalog
        Original archives validated through the same complete reader contract.

    Raises
    ------
    ValueError
        Placement, runtime or any selected archive declaration is invalid.
    OSError
        Inputs cannot be read or the new catalog cannot be created.
    """
    directory = directory.resolve(strict=True)
    if not directory.is_dir() or path.parent.resolve(strict=True) != directory:
        raise ValueError("The new catalog must be adjacent to its selected wheel directory.")
    wheels = []
    for archive in sorted(directory.glob("*.whl")):
        if archive.is_symlink() or not archive.is_file():
            raise ValueError("A catalog wheel must be a regular file, not a symbolic link.")
        with archive.open("rb") as stream:
            checksum = hashlib.file_digest(stream, "sha256").hexdigest()
        wheels.append(
            {"filename": archive.name, "sha256": checksum, "size_bytes": archive.stat().st_size}
        )
    declaration = {
        "schema_version": "studio.desktop.wheel-catalog.v1",
        "python_version": python_version,
        "platform": platform,
        "wheels": wheels,
    }
    catalog = _catalog(declaration, directory)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(declaration, indent=2, sort_keys=True) + "\n")
    return catalog


def main() -> int:
    """Create a verified catalog through the public build-tools command.

    Returns
    -------
    int
        Zero after writing the complete catalog; one after an ordinary input
        or filesystem refusal. Argument errors retain argparse's normal status.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--python-version", required=True)
    parser.add_argument("--platform", required=True)
    args = parser.parse_args()
    try:
        create_catalog(args.directory, args.output, args.python_version, args.platform)
    except (ValueError, OSError) as error:
        print(str(error), file=sys.stderr)
        return 1
    print(str(args.output.resolve()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
