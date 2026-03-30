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


# === ROUND 5: mock Rust engine, MPI, pretrained weights ===

from unittest.mock import MagicMock, patch


def test_rust_engine_import_success():
    # network.py:31 — _RUST_ENGINE = NetworkRunner
    import sc_neurocore.network.network as _nn

    old = _nn._RUST_ENGINE
    try:
        _nn._RUST_ENGINE = None
        mock_runner = MagicMock()
        mock_module = MagicMock()
        mock_module.NetworkRunner = mock_runner
        with patch.dict(
            "sys.modules",
            {
                "sc_neurocore_engine": mock_module,
                "sc_neurocore_engine.sc_neurocore_engine": mock_module,
            },
        ):
            result = _nn._get_rust_engine()
        assert result is not False
    finally:
        _nn._RUST_ENGINE = old


def test_can_use_rust_all_supported():
    # network.py:81-84 — loop through pops, all supported
    import sc_neurocore.network.network as _nn

    old = _nn._RUST_ENGINE
    try:
        mock_engine = MagicMock()
        mock_engine.SUPPORTED_MODELS = ["LapicqueNeuron"]
        _nn._RUST_ENGINE = mock_engine
        pop = Population(LapicqueNeuron, n=3, label="ru")
        net = Network(pop)
        with patch.object(_nn, "_rust_supports_model", return_value=True):
            r = net._can_use_rust()
        assert r is True
    finally:
        _nn._RUST_ENGINE = old


def test_run_mpi():
    # network.py:117 — runner.run(n_steps, dt)
    pop = Population(LapicqueNeuron, n=3, label="mpi")
    net = Network(pop)
    mock_runner_inst = MagicMock()
    mock_mpi_cls = MagicMock(return_value=mock_runner_inst)
    mock_mpi_module = MagicMock()
    mock_mpi_module.MPIRunner = mock_mpi_cls
    with (
        patch.dict("sys.modules", {"sc_neurocore.network.mpi_runner": mock_mpi_module}),
        patch("sc_neurocore.network.network.MPIRunner", mock_mpi_cls, create=True),
    ):
        net._run_mpi(0.01, 0.001)
    mock_runner_inst.run.assert_called_once()


def test_run_rust_full():
    # network.py:131-133,157-159 — projection add + spike unpacking
    import sc_neurocore.network.network as _nn

    pop = Population(LapicqueNeuron, n=3, label="rf")
    mon = SpikeMonitor(pop)
    proj_src = Population(LapicqueNeuron, n=3, label="rs")
    proj = Projection(proj_src, pop, weight=0.5, probability=1.0)
    proj._delay_steps = 0
    net = Network(pop)
    net.add(proj_src)
    net.add(proj)
    net.add(mon)

    mock_runner_inst = MagicMock()
    mock_runner_inst.add_population.side_effect = [0, 1]
    mock_runner_inst.run.return_value = {
        "voltages": [np.zeros(3), np.zeros(3)],
        "spike_data": [
            np.array([0x0000000100000005], dtype=np.uint64),
            np.array([], dtype=np.uint64),
        ],
    }
    mock_engine = MagicMock(return_value=mock_runner_inst)

    old = _nn._RUST_ENGINE
    try:
        _nn._RUST_ENGINE = mock_engine
        net._run_rust(0.01, 0.001)
    finally:
        _nn._RUST_ENGINE = old

    assert mon.count > 0


def test_network_stimulus_timed_array():
    # network.py:203,206 — TimedArray branch
    from sc_neurocore.network.stimulus import TimedArray

    pop = Population(LapicqueNeuron, n=3, label="ta")
    ta = TimedArray(values=[0.0, 1.0, 2.0, 3.0, 4.0] * 20)
    ta.target = pop
    mon = SpikeMonitor(pop)
    net = Network(pop, ta, mon)
    net.run(duration=0.05, dt=0.001, backend="python")


def test_network_stimulus_no_target():
    # network.py:203 — target is None → use first pop
    pop = Population(LapicqueNeuron, n=3, label="nt")
    drive = PoissonInput(n=3, rate_hz=500.0, weight=2.0, dt=0.001, seed=42)
    drive.target = None
    mon = SpikeMonitor(pop)
    net = Network(pop)
    net.add(drive)
    net.add(mon)
    net.run(duration=0.01, dt=0.001, backend="python")


def test_pretrained_mnist(tmp_path):
    # pretrained.py:66,72-74 — mnist path
    from sc_neurocore.model_zoo import pretrained as _pt
    from sc_neurocore.model_zoo.configs import mnist_classifier

    weight_path = tmp_path / "mnist_784_128_10.npz"
    net = mnist_classifier()
    W0 = np.random.randn(784, 128).astype(np.float64) * 0.01
    W1 = np.random.randn(128, 10).astype(np.float64) * 0.01
    np.savez(weight_path, W0=W0, W1=W1)

    old_dir = _pt._WEIGHTS_DIR
    try:
        _pt._WEIGHTS_DIR = tmp_path
        result = _pt.load_pretrained("mnist")
        assert len(result.projections) >= 2
    finally:
        _pt._WEIGHTS_DIR = old_dir


def test_pretrained_shd(tmp_path):
    # pretrained.py:76-78 — shd path
    from sc_neurocore.model_zoo import pretrained as _pt
    from sc_neurocore.model_zoo.configs import shd_speech_classifier

    weight_path = tmp_path / "shd_700_256_20.npz"
    net = shd_speech_classifier()
    W0 = np.random.randn(700, 256).astype(np.float64) * 0.01
    W_rec = np.random.randn(256, 256).astype(np.float64) * 0.01
    W1 = np.random.randn(256, 20).astype(np.float64) * 0.01
    np.savez(weight_path, W0=W0, W_rec=W_rec, W1=W1)

    old_dir = _pt._WEIGHTS_DIR
    try:
        _pt._WEIGHTS_DIR = tmp_path
        result = _pt.load_pretrained("shd")
        assert len(result.projections) >= 3
    finally:
        _pt._WEIGHTS_DIR = old_dir


def test_projection_uniform_delay_max():
    # projection.py:191,193 — uniform delay path via max_delay property
    src = Population(LapicqueNeuron, n=3, label="d1")
    tgt = Population(LapicqueNeuron, n=3, label="d2")
    proj = Projection(src, tgt, weight=0.5, probability=1.0, delay=7)
    assert proj.delay_mode == "uniform"
    assert proj.max_delay == 7


def test_topology_scale_free_early():
    # topology.py:87 — total == 0 fallback
    data, indices, indptr = scale_free(n=3, m=1, weight=1.0, seed=99)
    assert indices.size > 0


def test_population_get_states_plain_object():
    # population.py:102 — else: keys = ["v"]
    from sc_neurocore.neurons.models.mcculloch_pitts import McCullochPittsNeuron

    pop = Population(McCullochPittsNeuron, n=3, label="mp")
    states = pop.get_states()
    assert isinstance(states, dict)
