"""Tests for SCDashboard text rendering and history handling."""

import numpy as np
import pytest

from sc_neurocore.dashboard.text_dashboard import SCDashboard


def test_dashboard_initial_history_size():
    """History should initialize with correct length."""
    dash = SCDashboard(n_neurons=3)
    assert len(dash.history) == 3


def test_dashboard_update_appends_history():
    """Update should append latest firing rates."""
    dash = SCDashboard(n_neurons=2)
    dash.update([0.1, 0.2], step=1)
    assert dash.history[0][-1] == 0.1
    assert dash.history[1][-1] == 0.2


def test_dashboard_history_trims_to_20():
    """History should keep only last 20 entries."""
    dash = SCDashboard(n_neurons=1)
    for i in range(25):
        dash.update([i / 100.0], step=i)
    assert len(dash.history[0]) == 20


def test_dashboard_render_outputs_header(capsys):
    """Render output should include dashboard header."""
    dash = SCDashboard(n_neurons=1)
    dash.update([0.1], step=5)
    output = capsys.readouterr().out
    assert "SC DASHBOARD" in output
    assert "Step 5" in output


def test_dashboard_render_includes_neuron_ids(capsys):
    """Render output should include neuron identifiers."""
    dash = SCDashboard(n_neurons=2)
    dash.update([0.1, 0.2], step=1)
    output = capsys.readouterr().out
    assert "#0" in output
    assert "#1" in output


def test_dashboard_trend_up(capsys):
    """Increasing rates should show UP trend."""
    dash = SCDashboard(n_neurons=1)
    dash.update([0.10], step=1)
    _ = capsys.readouterr()
    dash.update([0.20], step=2)
    output = capsys.readouterr().out
    assert "/ UP" in output


def test_dashboard_trend_down(capsys):
    """Decreasing rates should show DWN trend."""
    dash = SCDashboard(n_neurons=1)
    dash.update([0.30], step=1)
    _ = capsys.readouterr()
    dash.update([0.10], step=2)
    output = capsys.readouterr().out
    assert "\\ DWN" in output


def test_dashboard_trend_stable(capsys):
    """Small changes should show STY trend."""
    dash = SCDashboard(n_neurons=1)
    dash.update([0.10], step=1)
    _ = capsys.readouterr()
    dash.update([0.105], step=2)
    output = capsys.readouterr().out
    assert "- STY" in output


def test_dashboard_bar_length(capsys):
    """Bar length should scale with rate."""
    dash = SCDashboard(n_neurons=1)
    dash.update([0.5], step=1)
    output = capsys.readouterr().out
    assert "||||||||||" in output


def test_dashboard_update_length_mismatch_raises():
    """Fewer rates than neurons should raise IndexError in render."""
    dash = SCDashboard(n_neurons=2)
    with pytest.raises(IndexError):
        dash.update([0.1], step=1)
