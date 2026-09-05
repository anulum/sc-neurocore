# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Studio display projection and raw custody block

"""Bounded display projection and raw block of ``sc_neurocore.studio.trace_projection``.

The projection must never exceed its point budget, must keep the first and the
last sample and every bucket extremum of every series, and every display point
must map to a raw step. The raw block must state its budget verdict instead of
shortening anything.
"""

from __future__ import annotations

import numpy as np
import pytest

from sc_neurocore.studio.state_layout import (
    DeclaredState,
    ObservedState,
    StateLayout,
)
from sc_neurocore.studio.trace_projection import (
    DISPLAY_SCHEMA_VERSION,
    MAX_PLOT_POINTS,
    RAW_SCHEMA_VERSION,
    custody_payload,
    display_sample_indices,
    full_state_trace,
    full_state_traces,
    raw_block,
    sample_times,
)


def _layout(*names: str) -> StateLayout:
    variables = tuple(
        ObservedState(
            DeclaredState(name, "unassigned", "", "", None), "scalar", (), True, "", "per-step"
        )
        for name in names
    )
    return StateLayout(source="descriptor", schema_profile="", variables=variables)


class TestDisplaySampleIndices:
    def test_identity_below_the_budget(self) -> None:
        projection = display_sample_indices(MAX_PLOT_POINTS, [np.zeros(MAX_PLOT_POINTS)])
        assert projection.method == "identity"
        assert projection.point_count == MAX_PLOT_POINTS
        assert projection.sample_index.tolist() == list(range(MAX_PLOT_POINTS))

    @pytest.mark.parametrize("n_steps", [MAX_PLOT_POINTS + 1, 9_999, 100_000])
    @pytest.mark.parametrize("n_series", [1, 4, 14])
    def test_bound_first_last_and_extrema(self, n_steps: int, n_series: int) -> None:
        rng = np.random.default_rng(n_steps + n_series)
        series = [rng.standard_normal(n_steps) for _ in range(n_series)]
        projection = display_sample_indices(n_steps, series)
        index = projection.sample_index
        assert projection.method == "bucket-extrema"
        assert projection.point_count <= MAX_PLOT_POINTS
        assert index[0] == 0 and index[-1] == n_steps - 1
        assert np.all(np.diff(index) > 0)
        for values in series:
            assert float(values[index].max()) == float(values.max())
            assert float(values[index].min()) == float(values.min())

    def test_isolated_spike_survives_reduction(self) -> None:
        n_steps = 40_000
        v = np.full(n_steps, -65.0)
        v[31_337] = 30.0
        v[31_338] = -75.0
        projection = display_sample_indices(n_steps, [v])
        shown = set(projection.sample_index.tolist())
        assert {31_337, 31_338} <= shown

    def test_unused_budget_is_spent_on_even_samples(self) -> None:
        n_steps = 20_000
        projection = display_sample_indices(n_steps, [np.zeros(n_steps)])
        index = projection.sample_index
        assert projection.point_count <= MAX_PLOT_POINTS
        assert projection.point_count > projection.bucket_count
        assert int(np.max(np.diff(index))) <= 2 * (n_steps // MAX_PLOT_POINTS) + 1

    def test_invalid_inputs_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="n_steps"):
            display_sample_indices(0, [])
        with pytest.raises(ValueError, match="max_points"):
            display_sample_indices(10, [np.zeros(10)], max_points=1)
        with pytest.raises(ValueError, match="does not match"):
            display_sample_indices(10, [np.zeros(9)])

    def test_budget_cannot_silently_drop_required_extrema(self) -> None:
        values = np.array([0.0, 2.0, -2.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        with pytest.raises(ValueError, match="preserve endpoints"):
            display_sample_indices(8, [values], max_points=2)
        projection = display_sample_indices(8, [values], max_points=4)
        assert projection.sample_index.tolist() == [0, 1, 2, 7]

    @pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -float("inf")])
    def test_nonfinite_samples_are_not_displayed_as_valid_extrema(self, invalid: float) -> None:
        with pytest.raises(ValueError, match="finite"):
            display_sample_indices(3, [np.array([1.0, invalid, 2.0])])


class TestClockAndRawBlock:
    def test_sample_times_are_post_step(self) -> None:
        assert sample_times(3, 0.5).tolist() == [0.5, 1.0, 1.5]

    def test_raw_block_within_budget_carries_every_step(self) -> None:
        block = raw_block(
            dt=0.5,
            n_steps=3,
            scalar_traces={"v": np.array([1.0, 2.0, 3.0])},
            vector_traces={"u": np.zeros((3, 2))},
            vector_omitted=["big"],
            drive=np.array([0.0, 1.0, 0.0]),
            spikes=[1],
        )
        assert block["schema_version"] == RAW_SCHEMA_VERSION
        assert block["included"] is True
        assert block["element_count"] == 3 * 2 + 6
        assert block["states"] == {"v": [1.0, 2.0, 3.0]}
        assert block["vector_states"] == {"u": [[0.0, 0.0]] * 3}
        assert block["drive"] == [0.0, 1.0, 0.0]
        assert block["spike_indices"] == [1]
        assert block["spike_times_ms"] == [1.0]
        assert block["vector_snapshots_only"] == ["big"]

    def test_raw_block_over_budget_states_the_reason_instead_of_shortening(self) -> None:
        block = raw_block(
            dt=0.1,
            n_steps=10,
            scalar_traces={"v": np.zeros(10)},
            vector_traces={},
            vector_omitted=[],
            drive=np.zeros(10),
            spikes=[],
            element_budget=5,
        )
        assert block["included"] is False
        assert "states" not in block and "drive" not in block
        assert "above the budget" in str(block["reason"])
        assert block["spike_indices"] == []


class TestCustodyPayload:
    def test_display_points_are_exact_raw_samples(self) -> None:
        n_steps = 12_345
        dt = 0.1
        rng = np.random.default_rng(7)
        v = rng.standard_normal(n_steps).cumsum()
        drive = np.sin(np.arange(n_steps) * 0.01)
        payload = custody_payload(
            dt=dt,
            n_steps=n_steps,
            layout=_layout("v"),
            initial_state={"v": 0.0},
            final_state={"v": float(v[-1])},
            scalar_traces={"v": v},
            vector_traces={},
            vector_omitted=(),
            drive=drive,
            spikes=[5, 77],
            stats={"rate_hz": 0.0},
        )
        display = payload["display"]
        index = np.array(display["sample_index"])
        assert display["schema_version"] == DISPLAY_SCHEMA_VERSION
        assert display["point_count"] == len(payload["time"]) == len(payload["states"]["v"])
        assert display["point_count"] <= MAX_PLOT_POINTS
        assert payload["states"]["v"] == v[index].tolist()
        assert payload["current_trace"] == drive[index].tolist()
        assert payload["time"] == ((index + 1) * dt).tolist()
        assert payload["raw"]["states"]["v"] == v.tolist()
        assert payload["raw"]["drive"] == drive.tolist()
        assert payload["spikes"] == [5, 77]
        assert payload["observation"]["clock"] == "post-step"
        assert payload["initial_state"] == {"v": 0.0}
        assert payload["final_state"] == {"v": float(v[-1])}
        assert payload["state_layout"]["complete"] is True


class TestFullStateTraces:
    def test_prefers_included_raw_traces(self) -> None:
        result = {
            "states": {"v": [1.0, 3.0]},
            "raw": {"included": True, "states": {"v": [1.0, 2.0, 3.0]}},
        }
        assert full_state_traces(result) == {"v": [1.0, 2.0, 3.0]}
        assert full_state_trace(result, "v") == [1.0, 2.0, 3.0]

    def test_falls_back_to_states_for_legacy_or_raw_less_results(self) -> None:
        assert full_state_traces({"states": {"v": [1.0]}}) == {"v": [1.0]}
        assert full_state_traces({"states": {"v": [1.0]}, "raw": {"included": False}}) == {
            "v": [1.0]
        }
        assert full_state_traces({}) == {}
        assert full_state_trace({"states": {}}, "v") == []
