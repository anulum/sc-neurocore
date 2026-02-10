"""Tests for GeneticRegulatoryLayer protein dynamics."""

import os
import time

import numpy as np
import pytest

from sc_neurocore.bio.grn import GeneticRegulatoryLayer


def _perf_enabled() -> bool:
    return os.environ.get("SC_NEUROCORE_PERF") == "1"


def test_grn_init_protein_levels():
    """Protein levels should initialize to zeros."""
    layer = GeneticRegulatoryLayer(n_neurons=4)
    assert np.allclose(layer.protein_levels, 0.0)


def test_grn_step_increases_with_spikes():
    """Spikes should increase protein levels."""
    layer = GeneticRegulatoryLayer(n_neurons=3, production_rate=0.1, decay_rate=0.0)
    layer.step(np.ones(3))
    assert np.all(layer.protein_levels > 0.0)


def test_grn_step_decays_without_spikes():
    """No spikes should decay protein levels."""
    layer = GeneticRegulatoryLayer(n_neurons=2, production_rate=0.0, decay_rate=0.1)
    layer.protein_levels[:] = 1.0
    layer.step(np.zeros(2))
    assert np.all(layer.protein_levels < 1.0)


def test_grn_clips_protein_levels():
    """Protein levels should be clipped between 0 and 10."""
    layer = GeneticRegulatoryLayer(n_neurons=2, production_rate=10.0, decay_rate=0.0)
    layer.step(np.ones(2))
    assert np.all(layer.protein_levels <= 10.0)


def test_grn_threshold_modulators_returns_protein():
    """Threshold modulators should equal protein levels."""
    layer = GeneticRegulatoryLayer(n_neurons=3)
    layer.protein_levels[:] = 2.5
    assert np.array_equal(layer.get_threshold_modulators(), layer.protein_levels)


def test_grn_negative_spikes_safe():
    """Negative spikes should not drive protein levels below zero."""
    layer = GeneticRegulatoryLayer(n_neurons=2, production_rate=0.1, decay_rate=0.0)
    layer.step(np.array([-1.0, -0.5]))
    assert np.all(layer.protein_levels >= 0.0)


def test_grn_shape_preserved():
    """Protein levels length should match n_neurons."""
    layer = GeneticRegulatoryLayer(n_neurons=5)
    layer.step(np.ones(5))
    assert layer.protein_levels.shape == (5,)


def test_grn_rate_parameters_effect():
    """Production and decay rates should influence change."""
    layer = GeneticRegulatoryLayer(n_neurons=1, production_rate=0.2, decay_rate=0.1)
    layer.step(np.array([1.0]))
    assert layer.protein_levels[0] > 0.0


def test_grn_dtype_float():
    """Protein levels should be float dtype."""
    layer = GeneticRegulatoryLayer(n_neurons=2)
    layer.step(np.ones(2))
    assert np.issubdtype(layer.protein_levels.dtype, np.floating)


@pytest.mark.skipif(not _perf_enabled(), reason="Set SC_NEUROCORE_PERF=1 to enable perf checks.")
def test_grn_perf_small():
    """Benchmark a small GRN update."""
    layer = GeneticRegulatoryLayer(n_neurons=128)
    spikes = np.random.randint(0, 2, size=128)
    start = time.perf_counter()
    layer.step(spikes)
    elapsed = time.perf_counter() - start
    assert elapsed < 2.0
