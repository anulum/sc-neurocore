# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Simple CLI Dashboard for monitoring SC simulation

"""Text dashboard for quick terminal monitoring of SC simulation rates."""


class SCDashboard:
    """Simple CLI dashboard for monitoring SC simulation rates."""

    def __init__(self, n_neurons: int) -> None:
        """Create a dashboard with one rolling history per neuron."""
        self.n_neurons = n_neurons
        self.history: list[list[float]] = [[] for _ in range(n_neurons)]

    def update(self, firing_rates: list[float], step: int) -> None:
        """Append one frame of firing rates and render the dashboard."""
        # Update history
        for i, rate in enumerate(firing_rates):
            self.history[i].append(rate)
            if len(self.history[i]) > 20:  # Keep last 20
                self.history[i].pop(0)

        self._render(step)

    def _render(self, step: int) -> None:
        # ANSI Escape codes for clearing screen/cursor might not work well in all notebook/CLI envs
        # We will just print a frame separator.

        print(f"\n--- SC DASHBOARD | Step {step} ---")
        print(f"{'Neuron':<8} | {'Rate':<8} | {'Trend (Last 5)'}")
        print("-" * 40)

        for i in range(self.n_neurons):
            rate = self.history[i][-1]

            # Simple sparkline-like trend
            trend = ""
            if len(self.history[i]) >= 2:
                diff = rate - self.history[i][-2]
                if diff > 0.01:
                    trend = "/ UP"
                elif diff < -0.01:
                    trend = "\\ DWN"
                else:
                    trend = "- STY"

            # Bar chart
            bar_len = int(rate * 20)
            bar = "|" * bar_len

            print(f"#{i:<7} | {rate:.3f}    | {trend} {bar}")

        print("-" * 40)
        # In a real terminal, we would use 'curses' to overwrite.
        # Here we just append.
