# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestEdgeDetectionRunner from former test_edge_crossing_detection.py

"""Focused suite: TestEdgeDetectionRunner from former test_edge_crossing_detection.py."""

from __future__ import annotations

from tests.edge_crossing_detection_support import *  # noqa: F403


class TestEdgeDetectionRunner:
    """The Python runner's ``crossing`` vs ``level`` spike-decision semantics."""

    def test_crossing_matches_hand_oscillator_over_a_sequence(self) -> None:
        """A no-reset crossing schema reproduces the McKean hand model's edge decision."""
        hand = McKeanNeuron(dt=0.1, **_MCKEAN_INIT, **_MCKEAN_PARAMS)
        schema = UniversalNeuron.from_dict(_mckean_schema("crossing"))

        for current in (0.0, 0.2, 0.3, 0.2, 0.1, 0.2):
            for _ in range(200):
                assert int(bool(schema.step(I=current))) == hand.step(current)
                assert schema.state["v"] == hand.v
                assert schema.state["w"] == hand.w

    def test_level_over_counts_a_non_resetting_oscillator(self) -> None:
        """``level`` fires every step above threshold; ``crossing`` fires once per crossing.

        During a spike the membrane stays above ``v_peak`` for several steps, so a
        non-resetting oscillator run under ``level`` detection reports many more spikes
        than the true number of upward crossings — exactly the over-count that edge
        detection exists to prevent.
        """
        edge = UniversalNeuron.from_dict(_mckean_schema("crossing"))
        level = UniversalNeuron.from_dict(_mckean_schema("level"))
        edge_spikes = sum(1 for _ in range(1000) if edge.step(I=0.2))
        level_spikes = sum(1 for _ in range(1000) if level.step(I=0.2))

        assert edge_spikes > 0
        assert level_spikes > edge_spikes

    def test_edge_detection_flag_requires_crossing_and_no_reset(self) -> None:
        """``_edge_detection`` is set only for a crossing model that declares no reset."""
        crossing_no_reset = UniversalNeuron.from_dict(_fhn_schema("crossing")).to_equation_neuron()
        assert crossing_no_reset._edge_detection is True

        level_no_reset = UniversalNeuron.from_dict(_fhn_schema("level")).to_equation_neuron()
        assert level_no_reset._edge_detection is False

        # A crossing model WITH a reset stays on the level path (reset clears the condition).
        crossing_with_reset = EquationNeuron(
            equations={"v": "-(v - E_L) / tau_m + I"},
            parameters={"E_L": -65.0, "tau_m": 10.0},
            state={"v": -65.0},
            threshold="v >= -50.0",
            reset={"v": "-65.0"},
            detection="crossing",
        )
        assert crossing_with_reset._edge_detection is False

    def test_reset_model_identical_under_level_and_crossing(self) -> None:
        """A reset-based integrate-and-fire model spikes identically either way."""

        def _lif(detection: str) -> EquationNeuron:
            return EquationNeuron(
                equations={"v": "-(v - E_L) / tau_m + I"},
                parameters={"E_L": -65.0, "tau_m": 10.0},
                state={"v": -65.0},
                threshold="v >= -50.0",
                reset={"v": "-65.0"},
                dt=1.0,
                detection=detection,
            )

        level_neuron = _lif("level")
        crossing_neuron = _lif("crossing")
        level_spikes = sum(level_neuron.step(I=5.0) for _ in range(200))
        crossing_spikes = sum(crossing_neuron.step(I=5.0) for _ in range(200))
        assert level_spikes > 0
        assert level_spikes == crossing_spikes

    def test_initial_threshold_active_seeds_edge_history(self) -> None:
        """The edge history is seeded from the initial state, not assumed ``False``."""
        below = UniversalNeuron.from_dict(_fhn_schema("crossing")).to_equation_neuron()
        assert below.initial_threshold_active() is False
        assert below._prev_threshold_active is False

        # A no-reset crossing neuron that starts already above threshold seeds ``True`` so
        # it does not emit a spurious first-step spike before a genuine re-crossing.
        above = EquationNeuron(
            equations={"v": "-v + I", "w": "0.0 * w"},
            state={"v": 5.0, "w": 0.0},
            threshold="v >= 1.0",
            detection="crossing",
        )
        assert above.initial_threshold_active() is True
        assert above._prev_threshold_active is True
        assert above.step(I=0.0) == 0  # above threshold but no rising edge -> no spike

    def test_reset_reseeds_edge_history(self) -> None:
        """``reset()`` restores the seeded edge history so a re-run is reproducible."""
        neuron = UniversalNeuron.from_dict(_mckean_schema("crossing")).to_equation_neuron()
        for _ in range(300):
            neuron.step(I=0.2)
        neuron.reset()
        assert neuron._prev_threshold_active is neuron.initial_threshold_active()
        assert neuron.state == neuron.initial_state

    def test_no_threshold_neuron_never_edge_active(self) -> None:
        """A neuron without a threshold reports no initial activity and never edge-fires."""
        neuron = EquationNeuron(equations={"v": "I"}, state={"v": 0.0}, detection="crossing")
        assert neuron.initial_threshold_active() is False
        assert neuron._edge_detection is False

    def test_invalid_detection_rejected(self) -> None:
        """An unknown detection mode fails closed rather than silently defaulting."""
        with pytest.raises(ValueError, match="detection must be one of"):
            EquationNeuron(equations={"v": "I"}, state={"v": 0.0}, detection="edge")

    def test_previous_state_alias_exposes_unwrapped_candidate_crossing(self) -> None:
        """A backward phase wrap at negative current must not look like a spike."""
        schema = UniversalNeuron.from_dict(_wrapped_phase_schema(theta=0.01))

        assert schema.step(I=-0.5) == 0
        assert 0.0 <= schema.state["theta"] < 2.0 * 3.141592653589793
        assert schema.state["theta"] > 3.141592653589793

    def test_previous_state_alias_names_are_reserved(self) -> None:
        """A user parameter cannot shadow the generated macro-boundary alias."""
        with pytest.raises(ValueError, match="previous-state aliases"):
            EquationNeuron(
                equations={"v": "v + I"},
                parameters={"v_prev": 0.0},
                state={"v": 0.0},
                dt=1.0,
                method="map",
            )

    @pytest.mark.parametrize("detection", ["poisson", "escape_rate"])
    def test_stochastic_detection_markers_accepted_without_edge(self, detection: str) -> None:
        """Zero-probability stochastic trials advance RNG without level spikes."""
        probability_expression = "0.0" if detection == "poisson" else None
        rate_expression = "0.0" if detection == "escape_rate" else None
        neuron = EquationNeuron(
            equations={"v": "I"},
            state={"v": 0.0},
            threshold="stochastic",
            detection=detection,
            probability_expression=probability_expression,
            rate_expression=rate_expression,
        )
        initial_rng = neuron.stochastic_rng_state

        assert [neuron.step(I=10.0) for _ in range(4)] == [0, 0, 0, 0]
        assert neuron.state["v"] == pytest.approx(4.0)
        assert neuron.stochastic_rng_state != initial_rng
