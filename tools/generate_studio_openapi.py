#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — deterministic Studio OpenAPI reference generator

"""Generate or verify the committed Studio OpenAPI contract."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from collections.abc import Sequence
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPO_ROOT / "docs" / "_generated" / "studio_openapi.json"

sys.path.insert(0, str(REPO_ROOT / "src"))


def render_studio_openapi() -> str:
    """Return the current Studio OpenAPI document as deterministic JSON."""
    app_module = importlib.import_module("sc_neurocore.studio.app")
    return (
        json.dumps(
            app_module.create_app().openapi(),
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output path for the generated OpenAPI JSON.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Fail when the committed output differs from the runtime contract.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Generate or check the Studio OpenAPI reference.

    Parameters
    ----------
    argv:
        Optional command arguments excluding the executable name.

    Returns
    -------
    int
        Zero on success and one when check mode detects drift.
    """
    args = _parser().parse_args(argv)
    output = args.output.resolve()
    rendered = render_studio_openapi()
    if args.check:
        if not output.is_file() or output.read_text(encoding="utf-8") != rendered:
            print(f"Studio OpenAPI reference is stale: {output}", file=sys.stderr)
            return 1
        print(f"Studio OpenAPI reference is current: {output}")
        return 0
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered, encoding="utf-8")
    print(f"Wrote Studio OpenAPI reference: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
