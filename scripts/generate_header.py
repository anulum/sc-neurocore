# SPDX-License-Identifier: AGPL-3.0-or-later
"""Generate the SC-NeuroCore technical header image for GitHub README."""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Configuration for 1280x640 (GitHub optimized)
WIDTH, HEIGHT = 12.8, 6.4
DPI = 100


def generate_neurocore_header():
    np.random.seed(42)  # Reproducible output

    fig = plt.figure(figsize=(WIDTH, HEIGHT), dpi=DPI, facecolor="#050a15")
    ax = fig.add_axes([0, 0, 1, 1], frameon=False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # 1. Background: The Stochastic Grid
    grid_size = 40
    x_grid = np.linspace(0, 1, grid_size)
    y_grid = np.linspace(0, 1, grid_size)
    for i in range(grid_size):
        alpha = np.random.uniform(0.02, 0.08)
        ax.axhline(y_grid[i], color="cyan", lw=0.5, alpha=alpha)
        ax.axvline(x_grid[i], color="cyan", lw=0.5, alpha=alpha)

    # 2. Stochastic Bitstreams (The L1-L4 Data)
    for i in range(12):
        y_pos = 0.15 + (i * 0.06)
        bits = np.random.choice([0, 1], size=100, p=[0.7, 0.3])
        x_bits = np.linspace(0.1, 0.9, 100)
        for j, bit in enumerate(bits):
            if bit == 1:
                ax.plot(
                    [x_bits[j], x_bits[j]],
                    [y_pos, y_pos + 0.02],
                    color="#00ffff",
                    alpha=0.4,
                    lw=1.5,
                )

    # 3. The Petri Net Nodes (SCPN Logic)
    places = [(0.3, 0.8), (0.7, 0.8), (0.5, 0.6)]
    transitions = [(0.5, 0.8), (0.4, 0.7), (0.6, 0.7)]

    for px, py in places:
        circle = plt.Circle((px, py), 0.02, color="magenta", fill=False, lw=2, alpha=0.6)
        ax.add_artist(circle)
    for tx, ty in transitions:
        rect = plt.Rectangle(
            (tx - 0.015, ty - 0.015), 0.03, 0.03, color="cyan", fill=True, alpha=0.3
        )
        ax.add_artist(rect)

    # 4. Branding Text
    ax.text(
        0.5,
        0.45,
        "SC-NEUROCORE",
        color="white",
        fontsize=48,
        fontweight="bold",
        fontfamily="monospace",
        ha="center",
        alpha=0.9,
        transform=ax.transAxes,
    )
    ax.text(
        0.5,
        0.38,
        "STOCHASTIC COMPUTING & NEUROMORPHIC ENGINE v3.7",
        color="cyan",
        fontsize=14,
        fontfamily="monospace",
        ha="center",
        alpha=0.7,
        transform=ax.transAxes,
    )

    # 5. Spiking Neural Signal (Bottom Accent)
    t = np.linspace(0, 10, 1000)
    signal = np.sin(t) * np.exp(-0.1 * t) + np.random.normal(0, 0.05, 1000)
    ax.plot(np.linspace(0, 1, 1000), 0.05 + 0.05 * signal, color="magenta", lw=1, alpha=0.5)

    out_path = "docs/assets/sc_neurocore_header.png"
    plt.savefig(out_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    print(f"Generated {out_path} ({WIDTH * DPI:.0f}x{HEIGHT * DPI:.0f})")


if __name__ == "__main__":
    generate_neurocore_header()
