# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Brunel parity check: does a scaling recording keep the published rate agreement

"""Check a scaling-benchmark recording against the published rate agreement.

The public pages say that the dense NumPy lane of the Brunel network fires at
the Brian2 rate within :data:`RATE_TOLERANCE`. This tool reads one recording
written by ``benchmarks/scaling_benchmark.py`` and fails when a matched network
size is further off. Run on a fresh recording it tests the current code; run
on the committed recording it tests only that the recording still says what
the pages say.

Mean firing rate is one number per network. Two networks can share it and
still differ in timing, synchrony or correlation; this check does not look at
those.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import json
from pathlib import Path
import sys

RATE_TOLERANCE = 0.05
"""Largest relative difference of mean rates that the public pages allow."""
REFERENCE = "brian2"
PARITY_LANE = "sc_numpy_dense"


def rate_ratios(recording: object) -> dict[int, float]:
    """Return the parity lane's mean rate over the reference's, per network size.

    Parameters
    ----------
    recording : object
        Parsed recording of the scaling benchmark.

    Returns
    -------
    dict
        Network size to rate ratio, for every size that both lanes ran in the
        same regime; when a size has several regimes the first is used.

    Raises
    ------
    ValueError
        The recording has no row list, a row lacks a field, the reference
        rate is not positive, or no network size was run by both lanes.
    """
    rows = recording.get("data") if isinstance(recording, dict) else None
    if not isinstance(rows, list):
        raise ValueError("Recording has no row list.")
    rates: dict[tuple[str, int, str], float] = {}
    for row in rows:
        try:
            key = (str(row["simulator"]), int(row["n_neurons"]), str(row["regime"]))
            rates[key] = float(row["rate_mean_hz"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("Recording row lacks simulator, size, regime or rate.") from error
    ratios: dict[int, float] = {}
    for (simulator, size, regime), rate in sorted(rates.items()):
        reference = rates.get((REFERENCE, size, regime))
        if simulator != PARITY_LANE or reference is None or size in ratios:
            continue
        if reference <= 0.0:
            raise ValueError("Reference rate is not positive.")
        ratios[size] = rate / reference
    if not ratios:
        raise ValueError("No network size was run by both the parity lane and the reference.")
    return ratios


def main(argv: Sequence[str] | None = None) -> int:
    """Check one recording and report each matched size.

    Parameters
    ----------
    argv : sequence of str, optional
        Command-line arguments; the process arguments when omitted.

    Returns
    -------
    int
        Zero when every matched size is within the tolerance, one when a
        size is further off, two when the recording cannot be read or holds
        fewer matched sizes than required.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recording", type=Path)
    parser.add_argument("--minimum-sizes", type=int, default=1)
    arguments = parser.parse_args(argv)
    try:
        ratios = rate_ratios(json.loads(arguments.recording.read_text(encoding="utf-8")))
    except (OSError, ValueError) as error:
        print(str(error), file=sys.stderr)
        return 2
    if len(ratios) < arguments.minimum_sizes:
        print(f"Only {len(ratios)} matched network size(s).", file=sys.stderr)
        return 2
    failed = False
    for size, ratio in ratios.items():
        within = abs(ratio - 1.0) <= RATE_TOLERANCE
        failed |= not within
        print(
            f"N={size}: {PARITY_LANE}/{REFERENCE} rate ratio {ratio:.4f} {'ok' if within else 'OUTSIDE'}"
        )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
