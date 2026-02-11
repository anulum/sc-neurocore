# CopyRight: (c) 1998-2026 Miroslav Sotek. All rights reserved.
# Contact us: www.anulum.li  protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AFFERO GENERAL PUBLIC LICENSE v3
# Commercial Licensing: Available

"""Tests for PetriNetEngine — SCPN Fusion Engine via stochastic DenseLayer."""

import numpy as np
import pytest
from sc_neurocore_engine import PetriNetEngine


class TestPetriNetEngine:
    @staticmethod
    def _traffic_light_artifacts():
        """Simple 3-place, 3-transition traffic light Petri net.

        Places:  [Red, Yellow, Green]
        Transitions: [R->G, G->Y, Y->R]
        """
        # W_in: which places feed each transition (transitions x places)
        w_in = np.array([
            [0.9, 0.0, 0.0],  # T0 (R->G): consumes Red
            [0.0, 0.0, 0.9],  # T1 (G->Y): consumes Green
            [0.0, 0.9, 0.0],  # T2 (Y->R): consumes Yellow
        ], dtype=np.float64)

        # W_out: which places receive from each transition (places x transitions)
        w_out = np.array([
            [0.0, 0.0, 0.9],  # Red produced by T2
            [0.0, 0.9, 0.0],  # Yellow produced by T1
            [0.9, 0.0, 0.0],  # Green produced by T0
        ], dtype=np.float64)

        thresholds = np.array([0.3, 0.3, 0.3])
        marking = np.array([1.0, 0.0, 0.0])  # Start with Red

        return {
            "w_in": w_in,
            "w_out": w_out,
            "thresholds": thresholds,
            "marking": marking,
        }

    def test_init(self):
        arts = self._traffic_light_artifacts()
        engine = PetriNetEngine(arts)
        assert engine.n_places == 3
        assert engine.n_transitions == 3
        np.testing.assert_array_equal(engine.marking, [1.0, 0.0, 0.0])

    def test_step_changes_marking(self):
        arts = self._traffic_light_artifacts()
        engine = PetriNetEngine(arts, length=2048, seed=42)
        initial = engine.marking.copy()
        engine.step(seed=100)
        after = engine.marking
        # Marking should change after a step (Red consumed, Green produced)
        assert not np.array_equal(initial, after), "Marking should change after step"

    def test_step_count_increments(self):
        arts = self._traffic_light_artifacts()
        engine = PetriNetEngine(arts)
        assert engine.step_count == 0
        engine.step()
        assert engine.step_count == 1
        engine.step()
        assert engine.step_count == 2

    def test_run_returns_history(self):
        arts = self._traffic_light_artifacts()
        engine = PetriNetEngine(arts, length=1024)
        history = engine.run(5, seed=42)
        assert len(history) == 5
        for m in history:
            assert m.shape == (3,)
            assert np.all(m >= 0)

    def test_reset(self):
        arts = self._traffic_light_artifacts()
        engine = PetriNetEngine(arts)
        engine.step()
        engine.step()
        engine.reset(np.array([0.0, 1.0, 0.0]))
        np.testing.assert_array_equal(engine.marking, [0.0, 1.0, 0.0])
        assert engine.step_count == 0

    def test_reset_to_zero(self):
        arts = self._traffic_light_artifacts()
        engine = PetriNetEngine(arts)
        engine.step()
        engine.reset()
        np.testing.assert_array_equal(engine.marking, [0.0, 0.0, 0.0])

    def test_marking_non_negative(self):
        """Marking should never go negative."""
        arts = self._traffic_light_artifacts()
        engine = PetriNetEngine(arts, length=1024)
        for i in range(20):
            engine.step(seed=i)
            assert np.all(engine.marking >= 0), f"Negative marking at step {i}"

    def test_shape_mismatch_raises(self):
        with pytest.raises(AssertionError):
            PetriNetEngine({
                "w_in": np.eye(3),
                "w_out": np.eye(2),  # wrong shape
                "thresholds": np.zeros(3),
            })

    def test_token_conservation_qualitative(self):
        """Total tokens should not explode unboundedly."""
        arts = self._traffic_light_artifacts()
        engine = PetriNetEngine(arts, length=2048, seed=99)
        for i in range(50):
            engine.step(seed=i)
        total = engine.marking.sum()
        # Should stay bounded (not go to infinity)
        assert total < 100, f"Total marking {total} seems unbounded"
