#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — DM-01 short spike-raster animation (high-fidelity HH)

"""DM-01: ~60s-style multi-frame spike raster from polyglot-complete Hodgkin–Huxley.

Honesty
-------
Proves: local PNG/GIF frames of HH voltage/spikes under constant current.
Does not prove: power, FPGA timing, or polyglot parity (see model_fidelity_status).

Usage::

    PYTHONPATH=src python examples/dm01_spike_raster_gif.py
    PYTHONPATH=src python examples/dm01_spike_raster_gif.py --out /tmp/hh_raster.gif
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("examples/output/dm01_hh_raster.gif"),
        help="Output GIF path (PNG fallback if pillow missing)",
    )
    parser.add_argument("--n-steps", type=int, default=4000)
    parser.add_argument("--current", type=float, default=10.0)
    parser.add_argument("--frames", type=int, default=24)
    args = parser.parse_args()

    from sc_neurocore.neurons.models import HodgkinHuxleyNeuron

    neuron = HodgkinHuxleyNeuron()
    voltage, spikes = neuron.simulate(args.n_steps, current=args.current)
    t = np.arange(len(voltage)) * float(neuron.dt)

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib import animation
        from matplotlib.artist import Artist
    except ImportError as exc:
        raise SystemExit(f"matplotlib required: {exc}") from exc

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 2.8))
    (line,) = ax.plot([], [], lw=0.9, color="#2563eb")
    ax.set_xlim(float(t[0]), float(t[-1]))
    ax.set_ylim(float(voltage.min()) - 5.0, float(voltage.max()) + 5.0)
    ax.set_xlabel("time")
    ax.set_ylabel("v")
    ax.set_title(f"HodgkinHuxleyNeuron I={args.current} spikes={spikes} (local demo, not a claim)")
    ax.grid(True, alpha=0.3)
    chunk = max(len(voltage) // args.frames, 1)

    def init() -> tuple[Artist]:
        line.set_data([], [])
        return (line,)

    def update(frame: int) -> tuple[Artist]:
        end = min((frame + 1) * chunk, len(voltage))
        line.set_data(t[:end], voltage[:end])
        return (line,)

    anim = animation.FuncAnimation(
        fig,
        update,
        init_func=init,
        frames=args.frames,
        interval=80,
        blit=True,
    )
    if animation.writers.is_available("pillow"):
        anim.save(args.out, writer="pillow", fps=12)
        print(f"wrote GIF {args.out}")
    else:
        png = args.out.with_suffix(".png")
        fig.savefig(png, dpi=120)
        print(f"pillow writer unavailable; wrote static PNG {png}")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
