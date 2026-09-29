# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Installed five-runtime comparison entry point

"""Launch the owning complete comparison from an installed package or source checkout."""

import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path


def main(argv: Sequence[str] | None = None) -> int:
    """Run the actual bundled comparison with the current interpreter and explicit owner settings.

    Parameters
    ----------
    argv : sequence of str or None
        Comparison arguments; None reads the actual command line.

    Returns
    -------
    int
        Actual comparison process exit status; no dependency/build side effects.

    Raises
    ------
    FileNotFoundError
        Installed resources and the source checkout's owning script are absent.
    """
    directory = Path(__file__).resolve().parent
    script = directory / "benchmark_resources" / "bench_ann_to_snn_replay.py"
    if not script.is_file():
        script = directory.parents[2] / "benchmarks" / "bench_ann_to_snn_replay.py"
    if not script.is_file():
        raise FileNotFoundError("the complete dense IF comparison resource is absent")
    arguments = list(sys.argv[1:] if argv is None else argv)
    return subprocess.call([sys.executable, str(script), *arguments])


if __name__ == "__main__":
    raise SystemExit(main())
