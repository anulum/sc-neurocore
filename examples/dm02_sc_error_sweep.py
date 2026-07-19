#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — DM-02 SC reconstruction error sweep

"""DM-02: SC unipolar encode/decode error vs bitstream length.

Uses a rate proxy derived from high-fidelity Hodgkin–Huxley mid-drive
firing, then measures |decode − value| for increasing stream lengths.

Honesty: pedagogical SC accuracy curve only — not silicon energy/power.

Usage::

    PYTHONPATH=src python examples/dm02_sc_error_sweep.py
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("examples/output/dm02_sc_error_sweep.csv"),
    )
    parser.add_argument(
        "--png",
        type=Path,
        default=Path("examples/output/dm02_sc_error_sweep.png"),
    )
    args = parser.parse_args()

    from sc_neurocore import BitstreamEncoder, bitstream_to_probability
    from sc_neurocore.neurons.models import HodgkinHuxleyNeuron

    # Mid-drive HH rate proxy → [0,1]
    currents = np.linspace(0.0, 20.0, 9)
    rates: list[float] = []
    for i in currents:
        _v, spikes = HodgkinHuxleyNeuron().simulate(4000, current=float(i))
        duration_s = 4000 * 0.01 / 1000.0
        rates.append(spikes / duration_s if duration_s > 0 else 0.0)
    rate_arr = np.asarray(rates, dtype=float)
    proxy = float(rate_arr[len(rate_arr) // 2] / max(float(rate_arr.max()), 1.0))
    proxy = float(np.clip(proxy, 0.0, 1.0))

    lengths = [64, 128, 256, 512, 1024, 2048, 4096, 8192]
    rows: list[tuple[int, float]] = []
    for length in lengths:
        enc = BitstreamEncoder(0.0, 1.0, length=length, seed=length)
        bits = enc.encode(proxy)
        est = float(bitstream_to_probability(bits))
        rows.append((length, abs(est - proxy)))

    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["bitstream_length", "abs_error", "proxy_value"])
        for length, err in rows:
            w.writerow([length, f"{err:.8f}", f"{proxy:.8f}"])
    print(f"wrote {args.csv}")
    for length, err in rows:
        print(f"L={length:5d}  |err|={err:.5f}")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7, 3.5))
        ax.loglog([r[0] for r in rows], [r[1] for r in rows], "o-")
        ax.set_xlabel("bitstream length")
        ax.set_ylabel("|decode − proxy|")
        ax.set_title(f"SC unipolar error (HH-derived proxy={proxy:.3f})")
        ax.grid(True, which="both", alpha=0.3)
        fig.tight_layout()
        fig.savefig(args.png, dpi=120)
        plt.close(fig)
        print(f"wrote {args.png}")
    except ImportError:
        print("matplotlib missing — CSV only")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
