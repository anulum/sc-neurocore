# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Write or check the Studio verified-readiness seal

"""Write or check the Studio's sealed verified readiness.

The seal (``src/sc_neurocore/studio/verified_readiness.json``) is what an
installation shows as verified readiness, because an installation cannot
re-verify receipts whose subjects it does not carry. It must be rewritten
whenever a facet receipt, a descriptor or a receipt subject changes, which the
readiness evidence ledger also tracks.

Usage::

    python tools/studio_readiness_seal.py --write
    python tools/studio_readiness_seal.py --check
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _rendered_seal() -> tuple[Path, str]:
    """Return where the seal lives and the seal the checkout derives now.

    Imported here rather than at module level so the checkout's own ``src`` is
    put on the path before the package is imported.
    """
    sys.path.insert(0, str(REPO_ROOT / "src"))
    from sc_neurocore.studio.model_catalogue import readiness_seal_payload
    from sc_neurocore.studio.readiness_seal import SEAL_PATH, render_seal

    return SEAL_PATH, render_seal(readiness_seal_payload())


def seal_problems(path: Path | None = None) -> list[str]:
    """Return the reasons the seal at ``path`` (the tracked one by default) is missing or stale."""
    tracked, rendered = _rendered_seal()
    path = tracked if path is None else path
    if not path.is_file():
        return [f"missing readiness seal: {path.name}"]
    if path.read_text(encoding="utf-8") != rendered:
        return [f"stale readiness seal: {path.name} (run tools/studio_readiness_seal.py --write)"]
    return []


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true", help="write the seal")
    mode.add_argument("--check", action="store_true", help="fail if the tracked seal is stale")
    args = parser.parse_args(argv)
    if args.write:
        path, rendered = _rendered_seal()
        path.write_text(rendered, encoding="utf-8")
        print(f"Wrote {path}")
        return 0
    problems = seal_problems()
    for problem in problems:
        print(problem, file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
