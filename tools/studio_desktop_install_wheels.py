# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Complete standalone desktop wheel closure installation

"""Replace a staging runtime's exact wheel closure with recorded relative launchers."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

from installer import install
from installer.sources import WheelFile

from tools.studio_desktop_assembly_process import run_assembly_command
from tools.studio_desktop_runtime_inventory import (
    inspect_runtime,
    verify_installed_wheels,
)
from tools.studio_desktop_wheel_catalog import (
    plan_shared_payloads,
    read_catalog,
    verify_wheel_input,
)
from tools.studio_desktop_wheel_destination import RelocatableWheelDestination


def assemble_runtime(
    runtime: Path,
    catalog_path: Path,
    uv: Path,
    receipts: Path,
    timeout_seconds: float,
    *,
    apply: bool,
) -> dict[str, object]:
    """Replace the complete selected closure in one existing build-staging prefix.

    Parameters
    ----------
    runtime : pathlib.Path
        Existing standalone embedded Python, never the controlling environment
        or a managed installed desktop version.
    catalog_path : pathlib.Path
        Exact verified wheel catalog and selected archives outside the runtime.
    uv : pathlib.Path
        Existing external uv executable used for normal uninstall and pip check.
    receipts : pathlib.Path
        New external directory for original inputs, commands and output.
    timeout_seconds : float
        Finite positive deadline for each bounded command. The caller's admitted
        whole-process-group job also bounds the full producer lifetime/resources.
    apply : bool
        False retains a verified plan and normal uninstall dry run. True performs
        the declared full replacement; failure leaves unqualified build staging.

    Returns
    -------
    dict
        Actual planned or installed runtime result. No bundle manifest or delivery
        archive is sealed by this command and no desktop acceptance is inferred.

    Raises
    ------
    ValueError
        Input, runtime, placement, inventory or content bindings differ.
    subprocess.SubprocessError
        A real bounded interpreter, uninstaller or dependency check fails.
    RuntimeError
        A command's owned group cannot be completely reaped.
    OSError
        Reading inputs, creating immutable receipts or installing a payload fails.
    """
    runtime = runtime.resolve(strict=True)
    catalog_path = catalog_path.resolve(strict=True)
    uv = uv.resolve(strict=True)
    receipts = receipts.resolve()
    if (
        Path(sys.prefix).resolve().is_relative_to(runtime)
        or Path(sys.executable).resolve().is_relative_to(runtime)
        or catalog_path.is_relative_to(runtime)
        or uv.is_relative_to(runtime)
        or receipts.is_relative_to(runtime)
    ):
        raise ValueError("The control environment, inputs and receipts must be external.")
    if (
        runtime.parent.parent.name == "versions"
        and (runtime.parent.parent.parent / "owner.json").exists()
    ):
        raise ValueError("A managed installed desktop version is not build staging.")
    catalog = read_catalog(catalog_path)
    receipts.mkdir(parents=True)
    (receipts / "selected-catalog.json").write_bytes(catalog_path.read_bytes())
    before = inspect_runtime(runtime, catalog, timeout_seconds, receipts / "before")
    shared_payloads = plan_shared_payloads(catalog, before.scheme)
    records = receipts / "original-installed-records"
    records.mkdir()
    for name, filename in before.records.items():
        raw = Path(filename).read_bytes()
        if hashlib.sha256(raw).hexdigest() != before.record_sha256[name]:
            raise ValueError("An installed RECORD changed after the actual probe.")
        (records / (name + ".csv")).write_bytes(raw)
    uv_version = run_assembly_command(
        [str(uv), "--version"], timeout_seconds, receipts / "uv-version"
    )
    uninstall = [
        str(uv),
        "--no-config",
        "--offline",
        "pip",
        "uninstall",
        "--python",
        str(before.executable),
        "--system",
        *sorted(before.packages),
    ]
    run_assembly_command([*uninstall, "--dry-run"], timeout_seconds, receipts / "uninstall-dry-run")
    result: dict[str, object] = {
        "schema_version": "studio.desktop.runtime-assembly.v2",
        "state": "planned",
        "runtime": str(runtime),
        "python_version": catalog.python_version,
        "platform": catalog.platform,
        "packages": before.packages,
        "uv_version": uv_version.stdout.decode("utf-8").strip(),
        "bundle_sealed": False,
        "desktop_acceptance": False,
        "shared_wheel_payloads": {
            str(path.relative_to(runtime)): asdict(value) for path, value in shared_payloads.items()
        },
    }
    (receipts / "plan.json").write_text(json.dumps(result, indent=2) + "\n")
    if not apply:
        return result
    # A normal uninstall unlinks old owned files before any new payload is written;
    # opening existing uv-hardlinked files with overwrite would mutate cache inodes.
    run_assembly_command(uninstall, timeout_seconds, receipts / "uninstall")
    destination = RelocatableWheelDestination(
        before.scheme, str(before.executable), "posix", shared_payloads=shared_payloads
    )
    installed = receipts / "installed-wheels"
    installed.mkdir()
    for wheel in catalog.wheels:
        verify_wheel_input(wheel)
        with WheelFile.open(wheel.path) as source:
            install(source, destination, {"INSTALLER": b"installer\n"})
        (installed / (wheel.path.name + ".json")).write_text(
            json.dumps(
                {
                    "filename": wheel.path.name,
                    "sha256": wheel.sha256,
                    "bytes": wheel.size_bytes,
                },
                indent=2,
            )
            + "\n"
        )
    after = inspect_runtime(runtime, catalog, timeout_seconds, receipts / "after")
    bindings = verify_installed_wheels(runtime, catalog, after, require_pypa_cache_absence=True)
    for record in after.records.values():
        if (Path(record).parent / "direct_url.json").exists():
            raise ValueError("Installed origin metadata is not permitted in this bundle.")
    for path in Path(after.scheme["scripts"]).iterdir():
        if path.is_file() and not path.is_symlink():
            with path.open("rb") as stream:
                prefix = stream.read(4096)
            if prefix.startswith(b"#!") and str(runtime).encode() in prefix:
                raise ValueError("An installed launcher retains the builder prefix.")
    run_assembly_command(
        [
            str(uv),
            "--no-config",
            "--offline",
            "pip",
            "check",
            "--python",
            str(after.executable),
        ],
        timeout_seconds,
        receipts / "dependency-check",
    )
    for wheel in catalog.wheels:
        verify_wheel_input(wheel)
    result.update(
        {
            "state": "installed",
            "original_wheel_member_bindings": bindings,
            "uninstalled_upstream_bytecode": {
                wheel.path.name: {
                    "archive_sha256": wheel.sha256,
                    "members": [asdict(entry) for entry in wheel.generated_bytecode],
                }
                for wheel in catalog.wheels
                if wheel.generated_bytecode
            },
            "generated_bytecode_policy_sha256": hashlib.sha256(
                Path(__file__).with_name("studio_desktop_generated_bytecode.toml").read_bytes()
            ).hexdigest(),
            "installed_record_entries": after.record_entries,
            "installed_record_sha256": after.record_sha256,
        }
    )
    (receipts / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> int:
    """Run the complete selected-runtime producer through its public tools command.

    Returns
    -------
    int
        Zero after a verified plan or complete installed closure; one after an
        ordinary input, filesystem or bounded-command refusal. Parser errors
        retain argparse's normal status.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", required=True, type=Path)
    parser.add_argument("--catalog", required=True, type=Path)
    parser.add_argument("--uv", required=True, type=Path)
    parser.add_argument("--receipts", required=True, type=Path)
    parser.add_argument("--timeout-seconds", required=True, type=float)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    try:
        result = assemble_runtime(
            args.runtime,
            args.catalog,
            args.uv,
            args.receipts,
            args.timeout_seconds,
            apply=args.apply,
        )
    except (ValueError, OSError, subprocess.SubprocessError, RuntimeError) as error:
        print(str(error), file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
