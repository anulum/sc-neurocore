# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Source/config provenance header

"""Focused parity and custody tests for the complete Mojo network lane."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from sc_neurocore.accel.mojo.sc_compte_wm_network import (
    LIBRARY_PATH,
    SOURCE_PATH,
    SCCompteWMMojoNetwork,
    _HAS_MOJO_SC_COMPTE_WM_NETWORK,
)
from sc_neurocore.network import SCCompteWMNetworkSpec, SCCompteWMStimulus
from sc_neurocore.network.sc_compte_wm_network import SCCompteWMNetworkState
from sc_neurocore.neurons.models.compte_wm import CompteWMNeuron

REPOSITORY = Path(__file__).resolve().parents[1]
FACADE = SOURCE_PATH.with_name("__init__.py")
BENCHMARK = REPOSITORY / "benchmarks/bench_sc_compte_wm_network_mojo.py"
RESULT = REPOSITORY / "benchmarks/results/bench_sc_compte_wm_network_mojo.json"
PYTHON_RESULT = REPOSITORY / "benchmarks/results/bench_sc_compte_wm_network.json"


def _zero_events() -> tuple[np.ndarray, np.ndarray]:
    return np.zeros(2048, dtype=np.int64), np.zeros(512, dtype=np.int64)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_mojo_library_and_documented_complete_abi_exist() -> None:
    assert _HAS_MOJO_SC_COMPTE_WM_NETWORK
    assert LIBRARY_PATH.is_file() and LIBRARY_PATH.stat().st_size > 0
    source = SOURCE_PATH.read_text(encoding="utf-8")
    assert "all 2,048" in source and "excitatory and 512 inhibitory cells" in source
    for symbol in (
        "sc_compte_wm_network_kernel_spectrum_c",
        "sc_compte_wm_network_counter_poisson_c",
        "sc_compte_wm_network_step_c",
    ):
        assert f"fn {symbol}(" in source
        assert "@export" in source[max(0, source.index(symbol) - 80) : source.index(symbol)]
    assert "--target-cpu x86-64-v3" in source


def test_mojo_counter_fixture_and_preserved_scalar_cell_parity() -> None:
    network = SCCompteWMMojoNetwork()
    counts, receipt = network._counter_events(64, 0, 0)
    assert np.flatnonzero(counts).tolist() == [49, 61]
    assert receipt.total_events == 2
    exc, inh = _zero_events()
    exc[17] = 1
    step = network.step(external_exc_events=exc, external_inh_events=inh)
    original = CompteWMNeuron()
    assert original.step(external_spike=True) == 0
    state = network.state()
    assert state.v_exc_mv[17] == pytest.approx(original.v, rel=0.0, abs=2e-14)
    assert state.external_ampa_exc[17] == pytest.approx(original.s_ampa, abs=2e-14)
    assert step.excitatory_input.total_events == 1


def test_mojo_recurrent_fft_matches_cross_runtime_dense_anchor() -> None:
    zeros_exc = np.zeros(2048, dtype=np.float64)
    zeros_inh = np.zeros(512, dtype=np.float64)
    state = SCCompteWMNetworkState(
        step_index=0,
        v_exc_mv=np.full(2048, -60.0, dtype=np.float64),
        v_inh_mv=np.full(512, -70.0, dtype=np.float64),
        refractory_exc_ms=zeros_exc.copy(),
        refractory_inh_ms=zeros_inh.copy(),
        external_ampa_exc=zeros_exc.copy(),
        external_ampa_inh=zeros_inh.copy(),
        recurrent_nmda=zeros_exc.copy(),
        recurrent_nmda_rise=zeros_exc.copy(),
        recurrent_gabaa=zeros_inh.copy(),
    )
    state.recurrent_nmda[[0, 37, 1024, 1901]] = [0.2, 0.4, 0.1, 0.3]
    network = SCCompteWMMojoNetwork(state=state)
    exc, inh = _zero_events()
    network.step(external_exc_events=exc, external_inh_events=inh)
    result = network.state()
    assert result.v_exc_mv[113] == pytest.approx(-60.0099068230443, abs=3e-13)
    assert result.recurrent_nmda[37] == 0.39992000800000005


def test_mojo_seed_determinism_and_full_population_refractory() -> None:
    first = SCCompteWMMojoNetwork().run(0.1, statistics_window_ms=0.1)
    second = SCCompteWMMojoNetwork().run(0.1, statistics_window_ms=0.1)
    third = SCCompteWMMojoNetwork(SCCompteWMNetworkSpec(seed=43)).run(0.1, statistics_window_ms=0.1)
    assert first == second
    assert first.input_sha256 != third.input_sha256
    assert first.final_state_sha256 != third.final_state_sha256
    stimulus = SCCompteWMStimulus(0.0, 0.02, 600_000.0, kind="global_current", center_deg=None)
    stimulated = SCCompteWMMojoNetwork()
    receipt = stimulated.run(0.02, stimuli=(stimulus,), statistics_window_ms=0.02)
    assert receipt.excitatory_spikes == 2048
    assert np.all(stimulated.state().v_exc_mv == -60.0)
    assert np.all(stimulated.state().refractory_exc_ms == 2.0)
    exc, inh = _zero_events()
    stimulated.step(external_exc_events=exc, external_inh_events=inh)
    assert np.all(stimulated.state().v_exc_mv == -60.0)


def test_mojo_native_rejection_is_atomic_and_fixed_spec_fails_closed() -> None:
    state = SCCompteWMMojoNetwork().state()
    state.external_ampa_exc[0] = 1.0e6
    network = SCCompteWMMojoNetwork(state=state)
    before = network.state().sha256()
    exc, inh = _zero_events()
    exc[0] = 1
    with pytest.raises(ValueError, match="status 3"):
        network.step(external_exc_events=exc, external_inh_events=inh)
    assert network.state().sha256() == before
    with pytest.raises(ValueError, match="fixes"):
        SCCompteWMMojoNetwork(SCCompteWMNetworkSpec(dt_ms=0.01))


def test_mojo_benchmark_is_source_bound_and_matches_python_events() -> None:
    payload = json.loads(RESULT.read_text(encoding="utf-8"))
    python = json.loads(PYTHON_RESULT.read_text(encoding="utf-8"))
    assert payload["model"] == "SC-COMPTE-WM-NETWORK"
    assert payload["execution_path"] == "mojo-midpoint-rk2-radix2-fft-x86-64-v3"
    assert payload["configuration"]["cells"] == 2560
    assert payload["configuration"]["steps"] == 1000
    assert payload["configuration"]["repeats"] == 3
    assert payload["configuration"]["target_cpu"] == "x86-64-v3"
    assert payload["repeat_receipts_exact"] is payload["passed"] is True
    assert payload["persistent_bump_claimed"] is False
    assert payload["distractor_resistance_claimed"] is False
    assert payload["input_sha256"] == python["input_sha256"]
    assert payload["spike_sha256"] == python["spike_sha256"]
    assert payload["spike_counts"] == python["spike_counts"]
    for path in (SOURCE_PATH, FACADE, BENCHMARK):
        relative = path.relative_to(REPOSITORY).as_posix()
        assert payload["source_sha256"][relative] == _sha256(path)
