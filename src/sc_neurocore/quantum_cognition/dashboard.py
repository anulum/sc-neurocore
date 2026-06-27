# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Terminal dashboard for quantum cognition

"""ANSI terminal dashboard for real-time quantum cognition monitoring.

Renders a compact, colour-coded view of the GOTM Brain state directly
in the terminal.  Uses only stdlib (``os.get_terminal_size``, ANSI
escape codes) — no ``curses`` dependency.

Components:
    - Entanglement heatmap (1D bar per site, colour gradient)
    - ATP levels bar chart
    - Spike raster (last N steps)
    - LLM directive history
    - Learning curve summary
"""

from __future__ import annotations

import shutil
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .gotm_brain import GOTMBrain

# ANSI escape codes
_RESET = "\033[0m"
_BOLD = "\033[1m"
_DIM = "\033[2m"
_CYAN = "\033[36m"
_GREEN = "\033[32m"
_YELLOW = "\033[33m"
_RED = "\033[31m"
_MAGENTA = "\033[35m"
_BG_BLACK = "\033[40m"
_CLEAR_SCREEN = "\033[2J\033[H"

# Block characters for bar rendering (8 levels)
_BLOCKS = " ▁▂▃▄▅▆▇█"

# Colour gradient for entanglement heatmap (cold → hot)
_HEAT_COLOURS = [
    "\033[38;5;17m",  # deep blue (low)
    "\033[38;5;27m",
    "\033[38;5;33m",
    "\033[38;5;39m",  # cyan
    "\033[38;5;46m",  # green
    "\033[38;5;226m",  # yellow
    "\033[38;5;208m",  # orange
    "\033[38;5;196m",  # red (high)
]


def _heat_char(value: float, max_val: float) -> str:
    """Map a value to a coloured block character."""
    if max_val <= 0:
        return f"{_DIM}▁{_RESET}"
    frac = min(value / max_val, 1.0)
    block_idx = int(frac * (len(_BLOCKS) - 1))
    colour_idx = int(frac * (len(_HEAT_COLOURS) - 1))
    return f"{_HEAT_COLOURS[colour_idx]}{_BLOCKS[block_idx]}{_RESET}"


def _bar(value: float, max_val: float, width: int = 20) -> str:
    """Render a simple bar with fill fraction."""
    if max_val <= 0:
        return "░" * width
    frac = min(value / max_val, 1.0)
    filled = int(frac * width)
    if frac > 0.7:
        colour = _GREEN
    elif frac > 0.3:
        colour = _YELLOW
    else:
        colour = _RED
    return f"{colour}{'█' * filled}{'░' * (width - filled)}{_RESET}"


def _directive_colour(directive: str) -> str:
    """Return ANSI colour for a directive."""
    return {
        "FOCUS": _CYAN,
        "EXPLORE": _MAGENTA,
        "STABILIZE": _GREEN,
    }.get(directive, _DIM)


class TerminalDashboard:
    """ANSI terminal dashboard for quantum cognition monitoring.

    Parameters
    ----------
    max_raster_steps : int
        Number of recent steps to show in the spike raster.
    clear_screen : bool
        Whether to clear terminal before drawing.
    """

    def __init__(
        self,
        max_raster_steps: int = 40,
        clear_screen: bool = True,
    ) -> None:
        self.max_raster_steps = max_raster_steps
        self.clear_screen = clear_screen

    def draw(self, brain: "GOTMBrain") -> None:
        """Render a single dashboard frame.

        Parameters
        ----------
        brain : GOTMBrain
            The brain instance to visualise.
        """
        try:
            cols, rows = shutil.get_terminal_size((80, 24))
        except (ValueError, OSError):
            cols, rows = 80, 24

        lines: list[str] = []

        if self.clear_screen:
            lines.append(_CLEAR_SCREEN)

        # Header
        lines.append(f"{_BOLD}{_CYAN}╔══ GOTM Quantum Cognition Brain ══╗{_RESET}")
        state = brain.get_learning_state()
        lines.append(
            f"  Neurons: {state['n_neurons']}  │  "
            f"Steps: {state['total_steps']}  │  "
            f"Spikes: {state['total_spikes']}  │  "
            f"Backend: {state['bridge_backend']}"
        )
        lines.append(
            f"  LLM: {'✓' if state['has_llm'] else '✗'}  │  "
            f"Metabolic fails: {state['total_metabolic_failures']}"
        )
        lines.append("")

        # Entanglement heatmap
        emap = brain.pool.entanglement_map
        max_e = float(np.max(emap)) if len(emap) > 0 else 1.0
        heat = "".join(_heat_char(float(e), max_e) for e in emap)
        lines.append(f"  {_BOLD}Entanglement Map{_RESET}  (avg={state['avg_entanglement']:.6f})")
        lines.append(f"  {heat}")
        lines.append("")

        # ATP levels
        lines.append(f"  {_BOLD}ATP Levels{_RESET}  (avg={state['avg_atp']:.4f})")
        n_show = min(len(brain.neurons), cols // 3)
        atp_line = "  "
        for i in range(n_show):
            atp = brain.neurons[i].atp_level
            atp_line += _bar(atp, 1.0, width=2) + " "
        if len(brain.neurons) > n_show:
            atp_line += f"  {_DIM}(+{len(brain.neurons) - n_show} more){_RESET}"
        lines.append(atp_line)
        lines.append("")

        # Spike raster (last N steps)
        history = brain.get_history()
        recent = history[-self.max_raster_steps :]
        if recent:
            lines.append(f"  {_BOLD}Spike Raster{_RESET}  (last {len(recent)} steps)")
            raster_line = "  "
            for h in recent:
                n_spikes = h.get("n_spikes", 0)
                if n_spikes == 0:
                    raster_line += f"{_DIM}·{_RESET}"
                elif n_spikes <= 3:
                    raster_line += f"{_GREEN}│{_RESET}"
                elif n_spikes <= 8:
                    raster_line += f"{_YELLOW}║{_RESET}"
                else:
                    raster_line += f"{_RED}█{_RESET}"
            lines.append(raster_line)
            lines.append("")

        # Directive history (last 10)
        if recent:
            lines.append(f"  {_BOLD}Directive History{_RESET}")
            dir_line = "  "
            for h in recent[-10:]:
                d = h.get("directive", "?")
                c = _directive_colour(d)
                dir_line += f"{c}{d[0]}{_RESET} "
            lines.append(dir_line)
            lines.append(f"  {_DIM}F=FOCUS  E=EXPLORE  S=STABILIZE{_RESET}")
            lines.append("")

        # Footer
        lines.append(f"{_BOLD}{_CYAN}╚══════════════════════════════════╝{_RESET}")

        print("\n".join(lines))

    def __repr__(self) -> str:
        """Return a concise representation of the dashboard window size."""
        return f"TerminalDashboard(max_raster={self.max_raster_steps})"


__all__ = ["TerminalDashboard"]
