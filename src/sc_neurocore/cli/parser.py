# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Command parser and dispatcher

"""Build the command parser and dispatch requests to command modules."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from typing import cast

from .commands.compile import add_compile_commands
from .commands.deploy import add_deploy_command
from .commands.formal import add_formal_command
from .commands.hub import add_hub_command
from .commands.info import add_info_command
from .commands.maintenance import add_maintenance_commands
from .commands.mapping import add_mapping_command
from .commands.scnir import add_scnir_command
from .commands.serve import add_serve_command
from .commands.studio import add_studio_commands
from .commands.synthesis import add_synthesis_command

_HELP_EPILOG = """\
Modes and first steps:
  Model     info, compile, compile-nir, serve, map-nir
  Hardware  deploy, collect-synthesis, scnir, formal, hub-init
  Studio    studio and studio-* operator commands
  Maintain  benchmark, preflight

Start with `sc-neurocore info`, then run `sc-neurocore COMMAND --help`
for the options and examples belonging to one command.
"""


_Command = Callable[[argparse.Namespace], int]


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level parser and all command-specific parsers.

    Returns
    -------
    argparse.ArgumentParser
        Parser for the installed ``sc-neurocore`` console script.
    """
    parser = argparse.ArgumentParser(
        prog="sc-neurocore",
        description="SC-NeuroCore — stochastic computing and neuromorphic hardware toolkit",
        epilog=_HELP_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="store_true", help="Print version and exit")
    subparsers = parser.add_subparsers(
        dest="command",
        metavar="COMMAND",
        title="commands",
        description="Use `sc-neurocore COMMAND --help` for command-specific options.",
    )

    add_info_command(subparsers)
    add_compile_commands(subparsers)
    add_serve_command(subparsers)
    add_mapping_command(subparsers)
    add_deploy_command(subparsers)
    add_synthesis_command(subparsers)
    add_scnir_command(subparsers)
    add_formal_command(subparsers)
    add_hub_command(subparsers)
    add_studio_commands(subparsers)
    add_maintenance_commands(subparsers)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command-line interface and return a process exit status.

    Parameters
    ----------
    argv : Sequence[str] | None
        Argument vector without the executable name. ``None`` reads
        ``sys.argv`` through :mod:`argparse`.

    Returns
    -------
    int
        Zero on success, otherwise the command-specific failure status.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.version:
        from sc_neurocore import __version__

        print(f"sc-neurocore {__version__}")
        return 0

    handler = getattr(args, "handler", None)
    if handler is None:
        parser.print_help()
        return 0
    return cast(_Command, handler)(args)
