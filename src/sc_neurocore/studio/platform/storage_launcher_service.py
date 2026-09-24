# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — worker launcher service entry

"""Run the direct-spawn worker launcher as an operator-installed service."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
import signal
from types import FrameType

from sc_neurocore.studio.platform.storage_launcher_configuration import (
    load_launcher_configuration,
)
from sc_neurocore.studio.platform.storage_worker_launcher import WorkerLauncher


def main(argv: Sequence[str] | None = None) -> int:
    """Run the launcher service until SIGTERM or SIGINT.

    Parameters
    ----------
    argv : Sequence[str] or None
        ``--configuration PATH``; ``None`` reads ``sys.argv``.

    Returns
    -------
    int
        ``0`` after an orderly stop. Startup refusal raises instead.

    Notes
    -----
    ``ready`` is printed once the endpoint accepts connections. A refused or
    malformed connection is closed and the service continues; it never
    stops running workers on shutdown.
    """
    parser = argparse.ArgumentParser(prog="studio-worker-launcher")
    parser.add_argument("--configuration", required=True)
    args = parser.parse_args(argv)
    configuration = load_launcher_configuration(Path(args.configuration))
    stopping = False

    def request_stop(signum: int, frame: FrameType | None) -> None:
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    with WorkerLauncher(configuration) as launcher:
        print("ready", flush=True)
        while not stopping:
            try:
                launcher.serve_once()
            except (PermissionError, ValueError, TimeoutError, EOFError, OSError):
                continue
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
