# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Self-hosted hub command

"""Generate a local self-hosted hub Compose bundle."""

from __future__ import annotations

import argparse


def add_hub_command(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    """Register the self-hosted hub bundle command.

    Parameters
    ----------
    subparsers : argparse._SubParsersAction[argparse.ArgumentParser]
        Top-level command registry.
    """
    parser = subparsers.add_parser(
        "hub-init",
        help="Generate a self-hosted hub Compose bundle",
        description="Write a local Studio hub service and its deployment manifest.",
    )
    parser.add_argument("--output", "-o", default="build", help="Bundle output directory")
    parser.add_argument("--port", type=int, default=8001, help="Studio service port")
    parser.add_argument("--bind-host", default="127.0.0.1", help="Studio bind host")
    parser.add_argument("--hub-image", default="sc-neurocore-hub:local", help="Container image tag")
    parser.add_argument("--online", action="store_true", help="Clear generated offline-mode flags")
    parser.set_defaults(handler=run_hub_init)


def run_hub_init(args: argparse.Namespace) -> int:
    """Write the requested self-hosted hub bundle.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed ``hub-init`` arguments.

    Returns
    -------
    int
        Zero on success, otherwise one for invalid configuration or I/O failure.
    """
    from sc_neurocore.hub import HubBundleConfig, write_hub_bundle

    try:
        paths = write_hub_bundle(
            str(args.output),
            HubBundleConfig(
                studio_port=int(args.port),
                bind_host=str(args.bind_host),
                image=str(args.hub_image),
                offline=not bool(args.online),
            ),
        )
    except (OSError, ValueError) as exc:
        print(f"Error: {exc}")
        return 1

    print("SC-NeuroCore hub bundle generated")
    print(f"  Directory: {args.output}")
    print(f"  Compose:   {paths['compose']}")
    print(f"  Manifest:  {paths['manifest']}")
    return 0
