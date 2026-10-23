# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Desktop installation management entry point

"""Install, roll back or remove application code while retaining research data."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from sc_neurocore.refusals import AuthoredRefusal
from sc_neurocore.studio.desktop.installation import DesktopInstallation


def main() -> int:
    """Run the per-user installation lifecycle through the bundled interpreter.

    Returns
    -------
    int
        Zero after a verified operation, one after a filesystem or authored refusal.
        Argument errors use argparse's normal exit status. No download, package
        manager, system service or administrative installation is invoked.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    operations = parser.add_subparsers(dest="operation", required=True)
    install = operations.add_parser("install", help="Install an extracted desktop bundle.")
    install.add_argument("bundle", type=Path)
    rollback = operations.add_parser("rollback", help="Select a compatible retained build.")
    rollback.add_argument("build_id")
    operations.add_parser("uninstall", help="Remove application code and keep all user data.")
    args = parser.parse_args()
    try:
        installation = DesktopInstallation()
        if args.operation == "install":
            manifest = installation.install(args.bundle)
            print(f"Installed SNN Studio {manifest.version} ({manifest.build_id}).")
        elif args.operation == "rollback":
            manifest = installation.rollback(args.build_id)
            print(f"Selected SNN Studio {manifest.version} ({manifest.build_id}).")
        else:
            installation.uninstall()
            print(
                "Removed SNN Studio application files. Research data and configuration are preserved."
            )
    except (AuthoredRefusal, OSError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
