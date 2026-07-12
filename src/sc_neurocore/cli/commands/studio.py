# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Studio operator commands

"""Launch Studio and manage its local operator deployment state."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Literal, cast


def add_studio_commands(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    """Register Studio launch and operator commands.

    Parameters
    ----------
    subparsers : argparse._SubParsersAction[argparse.ArgumentParser]
        Top-level command registry.
    """
    launch = subparsers.add_parser(
        "studio",
        help="Launch the visual SNN design Studio",
        description="Start the local Studio application and open it in a browser.",
    )
    launch.add_argument("--port", type=int, default=8001, help="Studio port")
    launch.set_defaults(handler=run_studio)

    backup = subparsers.add_parser(
        "studio-backup-plan",
        help="Emit the Studio durable-state backup plan",
        description="Inspect configured durable state and emit a backup/restore manifest.",
    )
    backup.add_argument("--output", "-o", default=None, help="Optional JSON output path")
    backup.add_argument(
        "--include-local-paths",
        action="store_true",
        help="Include resolved local paths for an internal handoff",
    )
    backup.set_defaults(handler=run_studio_backup_plan)

    profile = subparsers.add_parser(
        "studio-deployment-profile",
        help="Emit a local, lab, or server deployment profile",
        description="Build a path-free Studio deployment package in JSON or env format.",
    )
    profile.add_argument("--studio-profile", choices=["local", "lab", "server"], default="local")
    profile.add_argument("--format", choices=["json", "env"], default="json")
    profile.add_argument("--output", "-o", default=None, help="Optional output path")
    profile.set_defaults(handler=run_studio_deployment_profile)

    preflight = subparsers.add_parser(
        "studio-preflight",
        help="Run the Studio release-readiness preflight",
        description="Evaluate Studio deployment posture and emit a path-free JSON report.",
    )
    preflight.add_argument("--output", "-o", default=None, help="Optional JSON output path")
    preflight.set_defaults(handler=run_studio_preflight)

    bootstrap = subparsers.add_parser(
        "studio-bootstrap-admin",
        help="Create the first Studio service-account identity",
        description="Create an initial service principal in a local Studio identity file.",
    )
    _add_identity_file_argument(bootstrap)
    bootstrap.add_argument("--principal-id", default="svc-studio-admin")
    bootstrap.add_argument("--role", dest="roles", action="append", default=None)
    bootstrap.add_argument("--token-bytes", type=int, default=32)
    bootstrap.add_argument("--expires-at-utc", default=None)
    bootstrap.add_argument("--allow-overwrite", action="store_true")
    bootstrap.set_defaults(handler=run_studio_bootstrap_admin)

    browser_user = subparsers.add_parser(
        "studio-add-browser-user",
        help="Add a persistent Studio browser-login user",
        description="Add a password-authenticated browser user to a Studio identity file.",
    )
    _add_identity_file_argument(browser_user)
    browser_user.add_argument("--principal-id", default="svc-studio-admin")
    browser_user.add_argument("--username", default=None)
    browser_user.add_argument("--role", dest="roles", action="append", default=None)
    browser_user.add_argument("--password-stdin", action="store_true")
    browser_user.add_argument("--expires-at-utc", default=None)
    browser_user.set_defaults(handler=run_studio_add_browser_user)


def _add_identity_file_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--identity-file",
        default=None,
        help="Studio identity JSON path",
    )


def run_studio(args: argparse.Namespace) -> int:
    """Launch the local Studio application.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed ``studio`` arguments.

    Returns
    -------
    int
        Zero after a clean server shutdown, otherwise one when Studio extras are absent.
    """
    try:
        import uvicorn
    except ImportError:
        print("Error: Studio requires FastAPI + Uvicorn.")
        print("Install with: pip install sc-neurocore[studio]")
        return 1

    import webbrowser

    from sc_neurocore.studio.app import create_app

    app = create_app()
    url = f"http://127.0.0.1:{int(args.port)}"
    print(f"SC-NeuroCore Studio starting at {url}")
    webbrowser.open(url)
    uvicorn.run(app, host="127.0.0.1", port=int(args.port), log_level="warning")
    return 0


def run_studio_backup_plan(args: argparse.Namespace) -> int:
    """Emit the Studio durable-state backup and restore plan.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed ``studio-backup-plan`` arguments.

    Returns
    -------
    int
        Zero when required durable targets exist, otherwise one.
    """
    from sc_neurocore.studio.platform import build_studio_backup_plan

    try:
        plan = build_studio_backup_plan(include_local_paths=bool(args.include_local_paths))
    except ValueError as exc:
        print(f"Error: {exc}")
        return 1
    payload = json.dumps(plan.to_public_dict(), indent=2, sort_keys=True)
    _write_or_print(payload, args.output)
    return 0 if plan.missing_required_count == 0 else 1


def run_studio_bootstrap_admin(args: argparse.Namespace) -> int:
    """Create the first local Studio service-account identity file.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed ``studio-bootstrap-admin`` arguments.

    Returns
    -------
    int
        Zero on success, otherwise one for invalid input or I/O failure.
    """
    from sc_neurocore.studio.platform import (
        DEFAULT_STUDIO_ADMIN_ROLES,
        bootstrap_studio_admin_identity,
    )

    if args.identity_file is None:
        print(
            "Error: studio-bootstrap-admin requires --identity-file /path/to/studio-identities.json"
        )
        return 1
    roles = (
        tuple(str(role) for role in args.roles)
        if args.roles is not None
        else DEFAULT_STUDIO_ADMIN_ROLES
    )
    try:
        result = bootstrap_studio_admin_identity(
            Path(args.identity_file),
            principal_id=str(args.principal_id),
            roles=roles,
            token_bytes=int(args.token_bytes),
            expires_at_utc=args.expires_at_utc,
            overwrite=bool(args.allow_overwrite),
        )
    except (FileExistsError, OSError, ValueError) as exc:
        print(f"Error: {exc}")
        return 1
    output = result.to_public_dict()
    output["bearer_token"] = result.bearer_token
    output["environment"] = f"SC_NEUROCORE_STUDIO_IDENTITY_FILE={result.identity_file_path}"
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


def run_studio_deployment_profile(args: argparse.Namespace) -> int:
    """Emit a Studio deployment profile package.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed ``studio-deployment-profile`` arguments.

    Returns
    -------
    int
        Zero on success, otherwise one for an invalid profile.
    """
    from sc_neurocore.studio.platform import build_studio_deployment_profile_package

    try:
        profile = cast(Literal["local", "lab", "server"], args.studio_profile)
        package = build_studio_deployment_profile_package(profile)
    except ValueError as exc:
        print(f"Error: {exc}")
        return 1
    payload = (
        "\n".join(package.to_env_lines())
        if args.format == "env"
        else json.dumps(package.to_public_dict(), indent=2, sort_keys=True)
    )
    _write_or_print(payload, args.output)
    return 0


def run_studio_preflight(args: argparse.Namespace) -> int:
    """Run the Studio release-readiness preflight.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed ``studio-preflight`` arguments.

    Returns
    -------
    int
        Zero when the report passes, otherwise one.
    """
    from sc_neurocore.studio.platform import run_studio_preflight as build_report

    report = build_report()
    payload = json.dumps(report.to_public_dict(), indent=2, sort_keys=True)
    _write_or_print(payload, args.output)
    return 0 if report.passed else 1


def run_studio_add_browser_user(args: argparse.Namespace) -> int:
    """Add a persistent browser-login user to a Studio identity file.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed ``studio-add-browser-user`` arguments.

    Returns
    -------
    int
        Zero on success, otherwise one for invalid input or I/O failure.
    """
    from sc_neurocore.studio.platform import add_studio_browser_user_record

    if args.identity_file is None:
        print(
            "Error: studio-add-browser-user requires --identity-file "
            "/path/to/studio-identities.json"
        )
        return 1
    if args.username is None:
        print("Error: studio-add-browser-user requires --username <browser-user>")
        return 1
    if args.roles is None:
        print("Error: studio-add-browser-user requires at least one --role")
        return 1
    if not args.password_stdin:
        print("Error: studio-add-browser-user requires --password-stdin")
        return 1
    password = sys.stdin.readline().removesuffix("\n")
    try:
        record = add_studio_browser_user_record(
            Path(args.identity_file),
            username=str(args.username),
            principal_id=str(args.principal_id),
            roles=tuple(str(role) for role in args.roles),
            password=password,
            expires_at_utc=args.expires_at_utc,
        )
    except (OSError, ValueError) as exc:
        print(f"Error: {exc}")
        return 1
    output = {
        "browser_user": record.to_public_dict(),
        "environment": (
            f"SC_NEUROCORE_STUDIO_IDENTITY_FILE={Path(args.identity_file).expanduser()}"
        ),
        "schema_version": "sc-neurocore.studio.identity.browser-user.add.v1",
    }
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


def _write_or_print(payload: str, output: str | None) -> None:
    if output is None:
        print(payload)
        return
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(f"{payload}\n", encoding="utf-8")
