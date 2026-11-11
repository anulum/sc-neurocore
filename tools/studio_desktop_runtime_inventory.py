# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Embedded desktop runtime inventory and record integrity

"""Inspect the actual embedded interpreter and every installed distribution RECORD."""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import re
import tokenize
from dataclasses import dataclass
from email.parser import Parser
from pathlib import Path

from installer.sources import WheelFile
from packaging.tags import parse_tag

from tools.studio_desktop_assembly_process import run_assembly_command
from tools.studio_desktop_wheel_catalog import WheelCatalog, verify_wheel_input


@dataclass(frozen=True, slots=True)
class RuntimeInventory:
    """Runtime report fields matched to the selected interpreter and wheel closure.

    Attributes
    ----------
    executable : pathlib.Path
        Actual bundled interpreter, resolving inside its standalone prefix.
    scheme : dict of str to str
        Normal wheel installation destinations, confined to the runtime.
    packages : dict of str to str
        Exactly the catalog's distribution names and versions.
    records : dict of str to str
        Actual installed RECORD paths, keyed by distribution name.
    record_entries : dict of str to int
        Number of validated rows per installed RECORD.
    record_sha256 : dict of str to str
        Exact original installed RECORD content identities.
    raw_stdout : str
        Original JSON probe output for the producer receipt.
    """

    executable: Path
    scheme: dict[str, str]
    packages: dict[str, str]
    records: dict[str, str]
    record_entries: dict[str, int]
    record_sha256: dict[str, str]
    raw_stdout: str


def _string_map(value: object) -> dict[str, str]:
    """Validate one string mapping from the isolated interpreter probe."""
    if not isinstance(value, dict) or not all(
        isinstance(k, str) and isinstance(v, str) for k, v in value.items()
    ):
        raise ValueError("An embedded runtime probe mapping is invalid.")
    return dict(value)


def inspect_runtime(
    runtime: Path, catalog: WheelCatalog, timeout_seconds: float, receipts: Path
) -> RuntimeInventory:
    """Match the actual standalone runtime and all installed records to the catalog.

    Parameters
    ----------
    runtime : pathlib.Path
        Existing embedded standalone Python prefix.
    catalog : WheelCatalog
        Exact selected input closure, runtime version and Linux platform.
    timeout_seconds : float
        Finite positive deadline for the isolated interpreter probe.
    receipts : pathlib.Path
        New directory for original interpreter probe output and group custody.

    Returns
    -------
    RuntimeInventory
        Actual matched distribution set, schemes and fully checked installed records.

    Raises
    ------
    ValueError
        The prefix, interpreter, input compatibility or inventory does not match.
    subprocess.SubprocessError
        The real bounded interpreter probe fails or exceeds its deadline.
    OSError
        The runtime interpreter cannot be read or executed.
    """
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ValueError("The runtime probe deadline must be finite and positive.")
    runtime = runtime.resolve(strict=True)
    version = catalog.python_version.split(".")
    executable = runtime / "bin" / f"python{version[0]}.{version[1]}"
    if not executable.is_file() or not executable.resolve().is_relative_to(runtime):
        raise ValueError("The runtime must contain its own selected Python executable.")
    result = run_assembly_command(
        [
            str(executable),
            "-I",
            "-B",
            str(Path(__file__).with_name("studio_desktop_probe_runtime.py")),
        ],
        timeout_seconds,
        receipts,
    )
    tags_result = run_assembly_command(
        [str(executable), "-I", "-B", "-m", "pip", "--isolated", "debug", "--verbose"],
        timeout_seconds,
        receipts / "wheel-tags",
        environment={**os.environ, "PIP_CONFIG_FILE": os.devnull},
    )
    return parse_runtime_inventory(runtime, catalog, result.stdout, tags_result.stdout)


def parse_runtime_inventory(
    runtime: Path, catalog: WheelCatalog, probe_stdout: bytes, tag_stdout: bytes
) -> RuntimeInventory:
    """Decode captured interpreter reports and validate their selected input bindings.

    Parameters
    ----------
    runtime : pathlib.Path
        Existing prefix named by the captured interpreter report.
    catalog : WheelCatalog
        Selected wheel closure, Python version and platform.
    probe_stdout : bytes
        Original UTF-8 JSON from the isolated installed-record probe.
    tag_stdout : bytes
        Original UTF-8 output of the selected pip compatibility diagnostic.

    Returns
    -------
    RuntimeInventory
        Reported identity, scheme and record fields matched to the selected
        contract. Use ``inspect_runtime`` to execute the live file verification
        and collect these reports with bounded process custody.

    Raises
    ------
    ValueError
        A report, runtime identity, wheel tag, mapping or inventory count is invalid.
    OSError
        The declared prefix cannot be resolved.
    """
    runtime = runtime.resolve(strict=True)
    version = catalog.python_version.split(".")
    executable = runtime / "bin" / f"python{version[0]}.{version[1]}"
    value: object = json.loads(probe_stdout)
    if not isinstance(value, dict):
        raise ValueError("The embedded runtime probe did not return an object.")
    if (
        value.get("prefix") != str(runtime)
        or value.get("python_version") != catalog.python_version
        or value.get("platform") != catalog.platform
    ):
        raise ValueError("The actual embedded runtime identity differs from the catalog.")
    lines = tag_stdout.decode("utf-8").splitlines()
    declarations = [
        (index, int(match.group(1)))
        for index, line in enumerate(lines)
        if (match := re.fullmatch(r"Compatible tags: ([0-9]+)", line)) is not None
    ]
    if len(declarations) != 1:
        raise ValueError("The embedded pip compatibility declaration is invalid.")
    index, count = declarations[0]
    reported = lines[index + 1 :]
    if count <= 0 or len(reported) != count:
        raise ValueError("The embedded pip compatibility tag count differs.")
    tags = {str(tag) for line in reported for tag in parse_tag(line.strip())}
    for wheel in catalog.wheels:
        if not wheel.tags.intersection(tags):
            raise ValueError(f"The embedded runtime cannot install {wheel.path.name}.")
    packages = _string_map(value.get("packages"))
    if packages != {wheel.name: wheel.version for wheel in catalog.wheels}:
        raise ValueError("The installed distribution set differs from the complete catalog.")
    scheme = _string_map(value.get("scheme"))
    if set(scheme) != {"purelib", "platlib", "scripts", "headers", "data"} or any(
        not Path(path).resolve().is_relative_to(runtime) for path in scheme.values()
    ):
        raise ValueError("The actual installation scheme is invalid or escapes the runtime.")
    records = _string_map(value.get("records"))
    hashes = _string_map(value.get("record_sha256"))
    counts = value.get("record_entries")
    if not isinstance(counts, dict) or not all(
        isinstance(name, str)
        and isinstance(count, int)
        and not isinstance(count, bool)
        and count > 0
        for name, count in counts.items()
    ):
        raise ValueError("The installed RECORD counts are invalid.")
    if not (set(records) == set(hashes) == set(counts) == set(packages)):
        raise ValueError("Installed RECORD identities do not cover the whole inventory.")
    return RuntimeInventory(
        executable,
        scheme,
        packages,
        records,
        dict(counts),
        hashes,
        probe_stdout.decode("utf-8"),
    )


def verify_installed_wheels(
    runtime: Path,
    catalog: WheelCatalog,
    inventory: RuntimeInventory,
    *,
    require_pypa_cache_absence: bool = False,
) -> dict[str, int]:
    """Compare original wheel payloads with their installed scheme destinations.

    Parameters
    ----------
    runtime : pathlib.Path
        Actual embedded runtime prefix.
    catalog : WheelCatalog
        Complete verified original wheel closure.
    inventory : RuntimeInventory
        Actual post-install interpreter and scheme bindings.
    require_pypa_cache_absence : bool
        Require exact qualified upstream caches to be absent after PyPA
        installation. Other installation paths may retain them only byte-exact.

    Returns
    -------
    dict of str to int
        Original member bindings checked per archive, including the expected
        absence of exact qualified generated bytecode. Every other payload is
        byte-exact, apart from the declared relative Python script shebang.
        Installed RECORD is generated and checked separately.

    Raises
    ------
    ValueError
        A payload, destination or script transformation differs.
    OSError
        An original member or installed file cannot be read.
    """
    runtime = runtime.resolve(strict=True)
    compared: dict[str, int] = {}
    for wheel in catalog.wheels:
        if verify_wheel_input(wheel) != wheel.generated_bytecode:
            raise ValueError("The declared upstream bytecode disposition differs from its archive.")
        qualified_caches = {entry.path for entry in wheel.generated_bytecode}
        count = 0
        with WheelFile.open(wheel.path) as source:
            header = Parser().parsestr(source.read_dist_info("WHEEL"))
            root_scheme = "purelib" if header.get("Root-Is-Purelib") == "true" else "platlib"
            for record, stream, _ in source.get_contents():
                member = record[0]
                if member == source.dist_info_dir + "/RECORD":
                    continue
                scheme, relative = root_scheme, member
                if member.startswith(source.data_dir + "/"):
                    scheme, _, relative = member[len(source.data_dir) + 1 :].partition("/")
                    if scheme not in inventory.scheme or not relative:
                        raise ValueError("A wheel payload has an invalid data scheme.")
                target_path = Path(inventory.scheme[scheme], relative)
                if member in qualified_caches:
                    if not target_path.resolve().is_relative_to(runtime):
                        raise ValueError("An upstream cache destination escapes its runtime.")
                    present = target_path.exists() or target_path.is_symlink()
                    if require_pypa_cache_absence and present:
                        raise ValueError(
                            "PyPA-installed upstream generated bytecode must be absent."
                        )
                    if not present:
                        count += 1
                        continue
                target = target_path.resolve(strict=True)
                if not target.is_relative_to(runtime) or not target.is_file():
                    raise ValueError("An installed wheel member escapes its runtime.")
                if scheme == "scripts":
                    original = stream.read()
                    first, _, body = original.partition(b"\n")
                    if first.rstrip(b"\r") in {b"#!python", b"#!pythonw"}:
                        written = target.read_bytes()
                        launcher_header, separator, installed_body = written.partition(b"\n")
                        expected = (
                            "#!/usr/bin/env -S /bin/sh -c 'exec "
                            '"$(dirname -- "$(realpath -- "$0")")"/'
                            + inventory.executable.name
                            + ' "$0" "$@"\''
                        ).encode("ascii")
                        if (
                            launcher_header != expected
                            or separator != b"\n"
                            or installed_body != body
                            or tokenize.detect_encoding(io.BytesIO(original).readline)[0]
                            != tokenize.detect_encoding(io.BytesIO(written).readline)[0]
                        ):
                            raise ValueError("An installed Python data script differs.")
                        count += 1
                        continue
                    expected_digest = hashlib.sha256(original).hexdigest()
                else:
                    digest = hashlib.sha256()
                    while chunk := stream.read(1024 * 1024):
                        digest.update(chunk)
                    expected_digest = digest.hexdigest()
                with target.open("rb") as installed:
                    actual_digest = hashlib.file_digest(installed, "sha256").hexdigest()
                if actual_digest != expected_digest:
                    raise ValueError(f"An installed wheel payload differs: {member}")
                count += 1
        compared[wheel.path.name] = count
    return compared
