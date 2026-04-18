# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for SVG vector export

from __future__ import annotations

import pytest

from sc_neurocore.studio.svg_export import traces_to_svg


@pytest.fixture()
def sample_trace():
    time = [i * 0.1 for i in range(100)]
    states = {"v": [float(-65 + 30 * (i % 20 > 15)) for i in range(100)]}
    spikes = [i for i in range(100) if i % 20 == 16]
    return time, states, spikes


class TestTracesToSVG:
    def test_returns_valid_svg(self, sample_trace):
        time, states, spikes = sample_trace
        svg = traces_to_svg(time, states, spikes, model_name="LIFNeuron")
        assert svg.startswith("<svg")
        assert svg.endswith("</svg>")
        assert 'xmlns="http://www.w3.org/2000/svg"' in svg

    def test_contains_polyline(self, sample_trace):
        time, states, spikes = sample_trace
        svg = traces_to_svg(time, states, spikes)
        assert "<polyline" in svg
        assert 'stroke="#4fc3f7"' in svg

    def test_contains_axes(self, sample_trace):
        time, states, spikes = sample_trace
        svg = traces_to_svg(time, states, spikes)
        assert "time (ms)" in svg
        assert "mV" in svg

    def test_contains_spike_markers(self, sample_trace):
        time, states, spikes = sample_trace
        svg = traces_to_svg(time, states, spikes)
        assert 'stroke="#ff5252"' in svg

    def test_contains_legend(self, sample_trace):
        time, states, spikes = sample_trace
        svg = traces_to_svg(time, states, spikes)
        assert ">v</text>" in svg

    def test_model_name_watermark(self, sample_trace):
        time, states, spikes = sample_trace
        svg = traces_to_svg(time, states, spikes, model_name="AdExNeuron")
        assert "AdExNeuron" in svg

    def test_no_model_name_omits_watermark(self, sample_trace):
        time, states, spikes = sample_trace
        svg = traces_to_svg(time, states, spikes, model_name="")
        assert "font-family" in svg  # still has axis labels
        # no model name watermark line at end
        lines = svg.strip().split("\n")
        assert 'text-anchor="end"' not in lines[-2]

    def test_multiple_state_variables(self):
        time = [i * 0.1 for i in range(50)]
        states = {
            "v": [float(-65 + i) for i in range(50)],
            "w": [float(0.1 * i) for i in range(50)],
        }
        svg = traces_to_svg(time, states)
        assert svg.count("<polyline") == 2
        assert ">v</text>" in svg
        assert ">w</text>" in svg

    def test_empty_data_returns_placeholder(self):
        svg = traces_to_svg([], {})
        assert "No data" in svg
        assert "<svg" in svg

    def test_custom_dimensions(self, sample_trace):
        time, states, spikes = sample_trace
        svg = traces_to_svg(time, states, width=1200, height=600)
        assert 'width="1200"' in svg
        assert 'height="600"' in svg

    def test_no_spikes_omits_markers(self):
        time = [i * 0.1 for i in range(50)]
        states = {"v": [float(-65) for _ in range(50)]}
        svg = traces_to_svg(time, states, spikes=[])
        assert "#ff5252" not in svg

    def test_downsampling_for_large_traces(self):
        n = 10000
        time = [i * 0.01 for i in range(n)]
        states = {"v": [float(-65 + (i % 100)) for i in range(n)]}
        svg = traces_to_svg(time, states)
        # Should have <=2000 points in the polyline
        points_str = svg.split('points="')[1].split('"')[0]
        point_count = len(points_str.split(" "))
        assert point_count <= 2001

    def test_grid_lines(self, sample_trace):
        time, states, spikes = sample_trace
        svg = traces_to_svg(time, states)
        assert svg.count('stroke="#1a1f2a"') == 5

    def test_y_axis_ticks(self, sample_trace):
        time, states, spikes = sample_trace
        svg = traces_to_svg(time, states)
        assert svg.count('font-family="monospace"') == 5
