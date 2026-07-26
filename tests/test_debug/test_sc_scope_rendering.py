# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — SC scope rendering edge tests

"""Contracts for SC scope summary and density-bar rendering edges."""

from __future__ import annotations

from sc_neurocore.debug.sc_scope import ScopeRenderer
from tests.test_debug.sc_scope_edges_support import _session


def test_render_layer_summary_reports_no_data_for_empty_stats() -> None:
    """An empty stats mapping renders an explicit no-data row."""
    assert ScopeRenderer.render_layer_summary(3, {}) == "  L3: (no data)"


def test_render_session_includes_error_budget_section() -> None:
    """A session carrying an error budget renders the error-budget section."""
    session = _session()
    session.add_error_budget(layer_id=0, expected_density=0.4)

    rendered = ScopeRenderer.render_session(session)

    assert "Error Budgets" in rendered
    assert "L0:" in rendered


def test_render_density_bar_fills_proportionally() -> None:
    """The density bar fills a proportional number of cells and prints the value."""
    bar: str = ScopeRenderer.render_density_bar(0.5, width=10)

    assert bar.count("█") == 5
    assert "0.500" in bar
