#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Record evidence and refresh its catalogue views

"""Record a facet receipt, its ownership sidecar and all derived readiness views.

The record action accepts the existing facet recorder's arguments, except for
an alternate receipt directory: catalogue publication uses the canonical store.
The refresh action repairs derived views without repeating a validator.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    """Execute the receipt publication or derived-view recovery command.

    Parameters
    ----------
    argv : list of str, optional
        Existing ``record`` arguments, or ``refresh`` without other arguments.
        Process arguments are used when omitted.

    Returns
    -------
    int
        Zero after a creditable receipt and refreshed views, one after an
        uncreditable receipt and refreshed views, or two on refusal or a
        publication failure. Existing receipts are never removed on failure.
    """
    sys.path.insert(0, str(REPO_ROOT))
    sys.path.insert(0, str(REPO_ROOT / "src"))
    from sc_neurocore.neurons.facet_receipts import RECEIPT_DIR
    from tools.facet_receipt import main as record_facet
    from tools.readiness_artifact_refresh import (
        ReadinessArtifactError,
        refresh_readiness_artifacts,
        write_receipt_license,
    )

    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments or arguments[0] in ("-h", "--help"):
        parser = argparse.ArgumentParser(description=__doc__)
        actions = parser.add_subparsers(dest="action", required=True)
        actions.add_parser("record", help="run a validator and publish its receipt and views")
        actions.add_parser("refresh", help="refresh views without another validator run")
        parser.parse_args(arguments)
    options = arguments[: arguments.index("--")] if "--" in arguments else arguments
    if any(argument.partition("=")[0] == "--receipt-dir" for argument in options):
        print("error: readiness publication requires the canonical receipt store", file=sys.stderr)
        return 2
    try:
        if arguments and arguments[0] == "refresh":
            parser = argparse.ArgumentParser(
                description="Refresh all receipt-derived readiness views"
            )
            parser.parse_args(arguments[1:])
            outcome = 0
        else:
            before = set(RECEIPT_DIR.glob("*.json"))
            outcome = record_facet(arguments)
            if outcome not in (0, 1):
                return outcome
            created = set(RECEIPT_DIR.glob("*.json")) - before
            if len(created) != 1:
                raise ReadinessArtifactError(
                    "could not identify exactly one newly recorded receipt"
                )
            write_receipt_license(created.pop())
        for path in refresh_readiness_artifacts():
            print(f"refreshed {path.relative_to(REPO_ROOT)}")
    except ReadinessArtifactError as error:
        print(f"error: {error}; recorded receipts are retained", file=sys.stderr)
        return 2
    return outcome


if __name__ == "__main__":
    raise SystemExit(main())
