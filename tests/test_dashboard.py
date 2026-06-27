# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for CLI dashboard

"""Tests for the legacy flat SCDashboard test surface."""

import pytest

from sc_neurocore.dashboard.text_dashboard import SCDashboard


class TestSCDashboard:
    """Tests for dashboard construction and rendering."""

    def test_construction(self) -> None:
        """Constructor allocates one history list per neuron."""
        d = SCDashboard(n_neurons=5)
        assert d.n_neurons == 5
        assert len(d.history) == 5

    def test_update_records_history(self) -> None:
        """Update appends one firing-rate frame per neuron."""
        d = SCDashboard(n_neurons=3)
        d.update([0.1, 0.5, 0.9], step=0)
        assert d.history[0] == [0.1]
        assert d.history[1] == [0.5]
        assert d.history[2] == [0.9]

    def test_history_truncated_at_20(self) -> None:
        """History retains only the last twenty frames."""
        d = SCDashboard(n_neurons=1)
        for i in range(30):
            d.update([float(i)], step=i)
        assert len(d.history[0]) == 20

    def test_multiple_updates(self) -> None:
        """Multiple updates keep chronological history."""
        d = SCDashboard(n_neurons=2)
        d.update([0.1, 0.2], step=0)
        d.update([0.3, 0.4], step=1)
        assert len(d.history[0]) == 2
        assert d.history[0][-1] == 0.3

    def test_render_does_not_crash(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Render prints a dashboard frame after updates."""
        d = SCDashboard(n_neurons=2)
        d.update([0.5, 0.8], step=0)
        d.update([0.6, 0.3], step=1)
        captured = capsys.readouterr()
        assert "SC DASHBOARD" in captured.out
