# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestScopeRenderer from former test_sc_scope.py

"""Focused suite: TestScopeRenderer from former test_sc_scope.py."""

from __future__ import annotations

from sc_neurocore.debug.sc_scope import (
    LiveAnalyzer,
    ScopeRenderer,
    ScopeSession,
    TransportBackend,
    TransportConfig,
    TransportType,
)


class TestScopeRenderer:
    """Rendering the scope as text."""

    def test_density_bar(self) -> None:
        """A half-density bar shows filled and empty cells."""
        bar = ScopeRenderer.render_density_bar(0.5)
        assert "█" in bar
        assert "░" in bar
        assert "0.500" in bar

    def test_density_bar_empty(self) -> None:
        """A zero-density bar is all empty cells."""
        bar = ScopeRenderer.render_density_bar(0.0)
        assert "░" in bar

    def test_density_bar_full(self) -> None:
        """A full-density bar shows filled cells."""
        bar = ScopeRenderer.render_density_bar(1.0)
        assert "█" in bar

    def test_layer_summary(self) -> None:
        """A layer line shows the layer and its effective bits."""
        stats = {"mean_density": 0.5, "mean_effective_bits": 128.0, "sample_count": 10}
        line = ScopeRenderer.render_layer_summary(0, stats)
        assert "L0" in line
        assert "eff=" in line

    def test_render_session(self) -> None:
        """A live session renders its title and layers."""
        cfg = TransportConfig(TransportType.SIMULATED)
        tb = TransportBackend(cfg)
        la = LiveAnalyzer(num_layers=2)
        session = ScopeSession(transport=tb, analyzer=la)
        session.start()
        for _ in range(5):
            session.capture_sweep(num_layers=2)
        text = ScopeRenderer.render_session(session)
        assert "SC Bitstream Scope" in text
        assert "LIVE" in text
        assert "L0" in text
        session.stop()
