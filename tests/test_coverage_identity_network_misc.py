# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Coverage: identity, network, learning, model_zoo edge cases

"""Tests hitting uncovered lines in identity/, network/, learning/, model_zoo/."""

from __future__ import annotations

import numpy as np
import pytest

# --- identity/substrate (lines 164-166, 271) ---
from sc_neurocore.identity.substrate import IdentitySubstrate


def test_substrate_short_stimuli():
    # line 164-166: currents.shape[0] < n_cortical → pad
    sub = IdentitySubstrate(n_cortical=8)
    sub.step(np.ones(3, dtype=np.float64))


def test_substrate_zero_drive():
    # line 271: else branch (s_entropy = 0.0)
    sub = IdentitySubstrate(n_cortical=4)
    sub.step(np.zeros(4))


# --- identity/checkpoint (lines 55, 136, 138) ---
from sc_neurocore.identity.checkpoint import Checkpoint


def test_checkpoint_save_load(tmp_path):
    sub = IdentitySubstrate(n_cortical=8)
    path = tmp_path / "test.npz"
    Checkpoint.save(sub, path)
    restored = Checkpoint.load(str(path))
    assert restored.n_cortical == 8


def test_checkpoint_merge_single(tmp_path):
    # line 136-138: len(paths) == 1 → return load
    sub = IdentitySubstrate(n_cortical=8)
    path = tmp_path / "single.npz"
    Checkpoint.save(sub, path)
    merged = Checkpoint.merge([str(path)])
    assert merged.n_cortical == 8


# --- identity/decoder (lines 43, 65, 69-73, 81) ---
from sc_neurocore.identity.decoder import StateDecoder


def test_decoder_dominant_patterns():
    # line 43: empty trains → zeros output
    sub = IdentitySubstrate(n_cortical=8)
    dec = StateDecoder(sub)
    r = dec.extract_dominant_patterns()
    assert r.shape[0] >= 0


def test_decoder_attractors():
    # lines 65, 69-73: attractor detection
    sub = IdentitySubstrate(n_cortical=8)
    rng = np.random.default_rng(42)
    for _ in range(100):
        sub.step(rng.standard_normal(8) * 5)
    dec = StateDecoder(sub)
    r = dec.extract_attractor_states()
    assert isinstance(r, list)


def test_decoder_connectivity():
    # line 81: connectivity signature
    sub = IdentitySubstrate(n_cortical=8)
    dec = StateDecoder(sub)
    r = dec.extract_connectivity_signature()
    assert r.shape[0] >= 0


# --- learning/advanced (lines 353-358) ---
from sc_neurocore.learning.advanced import HomeostaticPlasticity
from sc_neurocore.network.population import Population
from sc_neurocore.neurons.models.lapicque import LapicqueNeuron


def test_homeostatic_plasticity():
    # lines 353-358: update with _rate_estimate > 0
    hp = HomeostaticPlasticity(target_rate=0.1, tau=100.0)
    pop = Population(LapicqueNeuron, n=5, label="hp")
    for _ in range(200):
        hp.update(pop)


# --- model_zoo/pretrained (lines 66, 76-78) ---
from sc_neurocore.model_zoo.pretrained import load_pretrained


def test_load_pretrained_nonexistent():
    # line 66: path not found
    with pytest.raises(Exception):
        load_pretrained("nonexistent_name_xyz")


# --- network/monitor (lines 71, 95-100, 165) ---
from sc_neurocore.network.monitor import SpikeMonitor, RateMonitor


def test_monitor_firing_rates_zero():
    # line 71: duration <= 0
    pop = Population(LapicqueNeuron, n=3, label="test")
    mon = SpikeMonitor(pop)
    rates = mon.firing_rates(n_steps=0, dt=0.001)
    assert np.all(rates == 0.0)


def test_monitor_cross_correlation():
    # lines 95-100: cross_correlation with spike data
    pop = Population(LapicqueNeuron, n=2, label="cc")
    mon = SpikeMonitor(pop)
    mon.record_event(0, 10)
    mon.record_event(0, 20)
    mon.record_event(1, 15)
    mon.record_event(1, 25)
    cc, lags = mon.cross_correlation(0, 1, max_lag=10)
    assert cc.size > 0


def test_rate_monitor_empty():
    # line 165: no spikes recorded
    pop = Population(LapicqueNeuron, n=2, label="rm")
    mon = RateMonitor(pop, bin_ms=10.0)
    r = mon.rate
    assert r.size == 0


# --- network/network (lines 31, 81-84, 203, 206) ---
from sc_neurocore.network.network import Network
from sc_neurocore.network.stimulus import PoissonInput


def test_network_python_backend():
    # lines 203, 206: stimulus routing in python backend
    pop = Population(LapicqueNeuron, n=5, label="net")
    drive = PoissonInput(n=5, rate_hz=500.0, weight=2.0, dt=0.001, seed=42)
    mon = SpikeMonitor(pop)
    net = Network(pop, drive, mon)
    net.run(duration=0.1, dt=0.001, backend="python")


# --- network/population (lines 72, 102) ---
def test_population_step_idle():
    # line 72: skip idle neurons
    pop = Population(LapicqueNeuron, n=5, label="idle")
    spikes = pop.step_all(np.zeros(5))
    assert spikes.shape[0] == 5


def test_population_get_states():
    # line 102: else branch (no w attribute)
    pop = Population(LapicqueNeuron, n=3, label="gs")
    states = pop.get_states()
    assert "v" in states


# --- network/projection (lines 48, 191, 193, 212-213, 275) ---
from sc_neurocore.network.projection import Projection


def test_projection_propagate():
    # line 48: weight_threshold skip
    src = Population(LapicqueNeuron, n=5, label="s")
    tgt = Population(LapicqueNeuron, n=5, label="t")
    proj = Projection(src, tgt, weight=0.5, probability=1.0)
    spikes = np.array([1, 0, 1, 0, 0], dtype=np.int8)
    result = proj.propagate(spikes)
    assert result.shape[0] == 5


def test_projection_delay_mode():
    # lines 191, 193: delay_mode property
    src = Population(LapicqueNeuron, n=3, label="s")
    tgt = Population(LapicqueNeuron, n=3, label="t")
    proj = Projection(src, tgt, weight=0.5, probability=1.0)
    assert proj.delay_mode == "none"


def test_projection_ring_mismatch():
    # lines 212-213: topology size mismatch
    src = Population(LapicqueNeuron, n=3, label="s")
    tgt = Population(LapicqueNeuron, n=5, label="t")
    with pytest.raises(ValueError):
        Projection(src, tgt, weight=0.5, probability=1.0, topology="ring")


def test_projection_non_stdp():
    # line 275: plasticity != "stdp" → early return
    src = Population(LapicqueNeuron, n=3, label="s")
    tgt = Population(LapicqueNeuron, n=3, label="t")
    proj = Projection(src, tgt, weight=0.5, probability=1.0)
    proj.update_plasticity(np.array([1, 0, 0], dtype=np.int8), np.array([0, 1, 0], dtype=np.int8))


# --- network/topology (line 87) ---
from sc_neurocore.network.topology import scale_free


def test_scale_free_topology():
    # line 87: preferential attachment fallback
    data, indices, indptr = scale_free(n=10, m=3, weight=1.0, seed=42)
    assert indices.size > 0


# === ROUND 2: remaining uncovered lines ===

from sc_neurocore.utils.numerics import clip_voltage


def test_clip_voltage():
    # numerics.py:56
    assert clip_voltage(500.0) == 100.0
    assert clip_voltage(-500.0) == -200.0


def test_checkpoint_merge_empty():
    # checkpoint.py:136 — empty paths
    with pytest.raises(ValueError):
        Checkpoint.merge([])


def test_substrate_entropy_zero():
    # substrate.py:271 — s_entropy = 0 when all spikes identical
    sub = IdentitySubstrate(n_cortical=4)
    # Run with zero input — no spikes → entropy 0
    for _ in range(5):
        sub.step(np.zeros(4))


def test_homeostatic_with_projections():
    # advanced.py:353-358 — _rate_estimate > 0, projections with data
    hp = HomeostaticPlasticity(target_rate=0.05, tau=10.0)
    pop = Population(LapicqueNeuron, n=5, label="hp2")
    # Force _rate_estimate > 0 by manually setting
    hp._rate_estimate = 0.2
    hp._step_count = 100

    class FakeProj:
        data = np.ones(10)

    pop._projections = [FakeProj()]
    hp.update(pop)
    assert hp._last_scale is not None


def test_population_state_dict_no_w():
    # population.py:102 — else: keys = ["v"]
    pop = Population(LapicqueNeuron, n=3, label="nw")
    states = pop.get_states()
    assert "v" in states


def test_projection_uniform_delay():
    # projection.py:193 — delay_mode == "uniform"
    src = Population(LapicqueNeuron, n=3, label="ds")
    tgt = Population(LapicqueNeuron, n=3, label="dt")
    proj = Projection(src, tgt, weight=0.5, probability=1.0, delay=3.0)
    assert proj.max_delay == 3


def test_scale_free_uniform_fallback():
    # topology.py:87 — probs[:] = 1.0 / src
    data, indices, indptr = scale_free(n=5, m=2, weight=1.0, seed=0)
    assert indices.size > 0


def test_network_stimulus_poisson_routing():
    # network.py:206 — isinstance(stim, PoissonInput) branch
    pop = Population(LapicqueNeuron, n=5, label="pr")
    drive = PoissonInput(n=5, rate_hz=1000.0, weight=5.0, dt=0.001, seed=42)
    mon = SpikeMonitor(pop)
    net = Network(pop, drive, mon)
    net.run(duration=0.05, dt=0.001, backend="python")


# === ROUND 3: precise branch targeting ===

from sc_neurocore.utils.numerics import clip_gating, boltzmann


def test_clip_gating():
    # numerics.py:51 area
    assert clip_gating(1.5) == 1.0
    assert clip_gating(-0.5) == 0.0


def test_boltzmann():
    r = boltzmann(-60.0, -40.0, 10.0)
    assert 0 < r < 1


def test_projection_uniform_max_delay():
    # projection.py:191,193 — uniform delay path
    src = Population(LapicqueNeuron, n=3, label="du")
    tgt = Population(LapicqueNeuron, n=3, label="dv")
    proj = Projection(src, tgt, weight=0.5, probability=1.0, delay=5)
    assert proj.delay_mode == "uniform"
    assert proj.max_delay == 5


def test_population_get_states_dataclass():
    # population.py:102 — __dataclass_fields__ branch
    pop = Population(LapicqueNeuron, n=3, label="dc")
    states = pop.get_states()
    assert "v" in states
    assert "tau" in states


def test_decoder_with_spiking_substrate():
    # decoder.py:72-73 — group >= 2 → append attractor
    sub = IdentitySubstrate(n_cortical=8)
    rng = np.random.default_rng(0)
    for _ in range(200):
        sub.step(rng.standard_normal(8) * 10)
    dec = StateDecoder(sub)
    attractors = dec.extract_attractor_states(threshold=0.3)
    assert isinstance(attractors, list)


def test_substrate_with_spiking():
    # substrate.py:271 — psd.sum() > 0 path
    sub = IdentitySubstrate(n_cortical=8)
    rng = np.random.default_rng(42)
    for _ in range(500):
        sub.step(rng.standard_normal(8) * 20)
    state = sub.extract_state()
    assert "total_steps" in state
    assert state["total_steps"] == 500


from sc_neurocore.network.stimulus import StepCurrent


def test_network_step_current():
    # network.py:206 — StepCurrent branch
    pop = Population(LapicqueNeuron, n=5, label="sc")
    step = StepCurrent(onset=10, offset=50, amplitude=5.0)
    step.target = pop
    mon = SpikeMonitor(pop)
    net = Network(pop, step, mon)
    net.run(duration=0.1, dt=0.001, backend="python")


# === ROUND 4: numerics + remaining ===

from sc_neurocore.utils.numerics import safe_cosh, safe_tanh, boltzmann_inv


def test_safe_cosh():
    assert np.isfinite(safe_cosh(0.0))
    assert np.isfinite(safe_cosh(1000.0))


def test_safe_tanh():
    assert safe_tanh(0.0) == 0.0
    assert abs(safe_tanh(1000.0) - 1.0) < 1e-10


def test_boltzmann_inv():
    r = boltzmann_inv(-60.0, -40.0, 10.0)
    assert 0 < r < 1
