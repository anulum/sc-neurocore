# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Audited real desktop installation file-fault controls

"""Introduce actual file drift at a live subprocess boundary in one isolated controller."""

from __future__ import annotations

import argparse
import json
import stat
import sys
from pathlib import Path

from tools.studio_desktop_assembly_process import run_assembly_command
from tools.studio_desktop_install_wheels import assemble_runtime
from tools.studio_desktop_runtime_inventory import inspect_runtime
from tools.studio_desktop_wheel_catalog import read_catalog


def main() -> int:
    """Exercise a real producer refusal with deterministic filesystem drift.

    Returns
    -------
    int
        Zero only after the requested file was changed, the exact public
        refusal was observed and the owned temporary file change was restored.
        Audit hooks remain confined to this controller process. Every actual
        subprocess still executes without an implementation replacement.

    Raises
    ------
    AssertionError
        A boundary, original seed, expected refusal or restoration differs.
    ValueError
        The producer refuses for a reason other than the requested control.
    OSError
        An owned file cannot be read, changed or restored.
    subprocess.SubprocessError
        A real seed command or inventory probe fails.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--uv", type=Path, required=True)
    parser.add_argument("--receipts", type=Path, required=True)
    parser.add_argument("--seconds", type=float, required=True)
    parser.add_argument("--fault", choices=["record", "origin", "launcher"], required=True)
    args = parser.parse_args()
    runtime = args.runtime.resolve(strict=True)
    catalog = read_catalog(args.catalog)
    inventory = inspect_runtime(runtime, catalog, args.seconds, args.receipts / "initial")
    record = Path(inventory.records["pip"])
    launcher = Path(inventory.scheme["scripts"]) / "pip"
    origin = record.parent / "direct_url.json"
    payload = b""
    if args.fault != "record":
        pip = next(wheel for wheel in catalog.wheels if wheel.name == "pip")
        assert not origin.exists()
        seed = args.receipts / "original-seed.txt"
        seed.write_text(pip.path.as_uri() + " --hash=sha256:" + pip.sha256 + "\n")
        run_assembly_command(
            [
                str(args.uv),
                "--no-config",
                "--offline",
                "pip",
                "install",
                "--python",
                str(inventory.executable),
                "--system",
                "--no-deps",
                "--require-hashes",
                "--only-binary",
                ":all:",
                "--reinstall",
                "--link-mode",
                "copy",
                "-r",
                str(seed),
            ],
            args.seconds,
            args.receipts / "actual-origin-seed",
        )
        actual_origin = origin.read_bytes()
        assert json.loads(actual_origin)["url"] == pip.path.as_uri()
        actual_launcher = launcher.read_bytes()
        assert actual_launcher.startswith(b"#!")
        assert str(runtime).encode() in actual_launcher[:4096]
        payload = actual_origin if args.fault == "origin" else actual_launcher
    expected_tags = [
        str(inventory.executable),
        "-I",
        "-B",
        "-m",
        "pip",
        "--isolated",
        "debug",
        "--verbose",
    ]
    operation = args.receipts / "assembly"
    target: Path | None = None
    saved: Path | None = None
    tag_calls = 0
    fired = False

    def audit(event: str, arguments: tuple[object, ...]) -> None:
        """Change one real file only after the selected actual record probe has exited."""
        nonlocal target, saved, tag_calls, fired
        if event != "subprocess.Popen" or len(arguments) < 2 or arguments[1] != expected_tags:
            return
        tag_calls += 1
        selected = 1 if args.fault == "record" else 2
        if tag_calls != selected:
            return
        phase = "before" if selected == 1 else "after"
        probe = json.loads((operation / phase / "receipt.json").read_text())
        assert probe["cleanup"]["reaped"] and probe["cleanup"]["returncode"] == 0
        target = {"record": record, "origin": origin, "launcher": launcher}[args.fault]
        assert target.resolve().is_relative_to(runtime) and not target.is_symlink()
        if args.fault == "origin":
            assert not target.exists()
            with target.open("xb") as stream:
                stream.write(payload)
        else:
            mode = stat.S_IMODE(target.stat().st_mode)
            changed = target.read_bytes() + b"\n" if args.fault == "record" else payload
            saved = target.with_name(target.name + ".audit-original")
            assert not saved.exists() and not saved.is_symlink()
            target.rename(saved)
            target.write_bytes(changed)
            target.chmod(mode)
        fired = True

    expected = {
        "record": "An installed RECORD changed after the actual probe.",
        "origin": "Installed origin metadata is not permitted in this bundle.",
        "launcher": "An installed launcher retains the builder prefix.",
    }[args.fault]
    sys.addaudithook(audit)
    try:
        try:
            assemble_runtime(
                runtime,
                args.catalog,
                args.uv,
                operation,
                args.seconds,
                apply=args.fault != "record",
            )
        except ValueError as error:
            assert fired and str(error) == expected
        else:
            raise AssertionError("The producer accepted the actual injected file drift.")
    finally:
        if target is not None:
            if saved is not None:
                target.unlink(missing_ok=True)
                saved.rename(target)
            elif args.fault == "origin":
                target.unlink(missing_ok=True)
    assert not (operation / "result.json").exists()
    if args.fault == "record":
        assert not (operation / "uninstall").exists()
    print(
        json.dumps(
            {
                "fault": args.fault,
                "injected": fired,
                "tag_probes": tag_calls,
                "refusal": expected,
                "bundle_sealed": False,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
