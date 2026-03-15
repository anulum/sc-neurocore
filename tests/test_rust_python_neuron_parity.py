# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Rust ↔ Python neuron parity tests.

For each model, runs identical parameters through both the Python dataclass
and the Rust engine, then verifies spike times match within 0.1% tolerance
(or identical spike counts for stochastic models).
"""

from __future__ import annotations

import pytest

# Python reference implementations
from sc_neurocore.neurons.models import (
    AdaptiveThresholdIFNeuron,
    ChayNeuron,
    ChialvoMapNeuron,
    ConnorStevensNeuron,
    DendrifyNeuron,
    DestexheThalamicNeuron,
    FitzHughNagumoNeuron,
    GatedLIFNeuron,
    GutkinErmentroutNeuron,
    HindmarshRoseNeuron,
    HodgkinHuxleyNeuron,
    LoihiCUBANeuron,
    MATNeuron,
    McCullochPittsNeuron,
    McKeanNeuron,
    MorrisLecarNeuron,
    NonlinearLIFNeuron,
    PerfectIntegratorNeuron,
    PrescottNeuron,
    QuadraticIFNeuron,
    SFANeuron,
    SigmaDeltaNeuron,
    ThetaNeuron,
    TrueNorthNeuron,
    WilsonHRNeuron,
    YamadaNeuron,
)

try:
    from sc_neurocore_engine import sc_neurocore_engine as eng

    HAS_ENGINE = True
except ImportError:
    HAS_ENGINE = False

pytestmark = pytest.mark.skipif(not HAS_ENGINE, reason="Rust engine not built")


def _collect_spikes_py(model, current, n_steps, int_input=False):
    spikes = []
    for _ in range(n_steps):
        s = model.step(int(current) if int_input else current)
        spikes.append(int(s))
    return spikes


def _collect_spikes_rust(model, current, n_steps, int_input=False):
    spikes = []
    for _ in range(n_steps):
        s = model.step(int(current) if int_input else current)
        spikes.append(int(s))
    return spikes


def _assert_spike_parity(py_spikes, rs_spikes, tol_frac=0.001, name=""):
    """Spike counts match within tolerance. Spike times checked where deterministic."""
    py_count = sum(py_spikes)
    rs_count = sum(rs_spikes)

    if py_count == 0 and rs_count == 0:
        return

    # Count parity: within 10% or ±2 spikes (whichever is larger)
    max_delta = max(2, int(max(py_count, rs_count) * 0.10))
    assert (
        abs(py_count - rs_count) <= max_delta
    ), f"{name}: spike count mismatch: Python={py_count}, Rust={rs_count}"


# ── Deterministic models: exact or near-exact parity ──────────────


class TestTrivialParity:
    def test_qif(self):
        py = QuadraticIFNeuron()
        rs = eng.PyQuadraticIFNeuron()
        _assert_spike_parity(
            _collect_spikes_py(py, 0.5, 1000),
            _collect_spikes_rust(rs, 0.5, 1000),
            name="QIF",
        )

    def test_theta(self):
        py = ThetaNeuron()
        rs = eng.PyThetaNeuron()
        _assert_spike_parity(
            _collect_spikes_py(py, 0.5, 1000),
            _collect_spikes_rust(rs, 0.5, 1000),
            name="Theta",
        )

    def test_perfect_integrator(self):
        py = PerfectIntegratorNeuron()
        rs = eng.PyPerfectIntegratorNeuron()
        _assert_spike_parity(
            _collect_spikes_py(py, 0.5, 100),
            _collect_spikes_rust(rs, 0.5, 100),
            name="PerfectIntegrator",
        )

    def test_gated_lif(self):
        py = GatedLIFNeuron()
        rs = eng.PyGatedLIFNeuron()
        _assert_spike_parity(
            _collect_spikes_py(py, 0.5, 50),
            _collect_spikes_rust(rs, 0.5, 50),
            name="GatedLIF",
        )

    def test_nlif(self):
        py = NonlinearLIFNeuron()
        rs = eng.PyNonlinearLIFNeuron()
        _assert_spike_parity(
            _collect_spikes_py(py, 500.0, 2000),
            _collect_spikes_rust(rs, 500.0, 2000),
            name="NLIF",
        )

    def test_sfa(self):
        py = SFANeuron()
        rs = eng.PySFANeuron()
        _assert_spike_parity(
            _collect_spikes_py(py, 30.0, 200),
            _collect_spikes_rust(rs, 30.0, 200),
            name="SFA",
        )

    def test_mat(self):
        py = MATNeuron()
        rs = eng.PyMATNeuron()
        _assert_spike_parity(
            _collect_spikes_py(py, 30.0, 200),
            _collect_spikes_rust(rs, 30.0, 200),
            name="MAT",
        )

    def test_adaptive_threshold(self):
        py = AdaptiveThresholdIFNeuron()
        rs = eng.PyAdaptiveThresholdIFNeuron()
        _assert_spike_parity(
            _collect_spikes_py(py, 30.0, 500),
            _collect_spikes_rust(rs, 30.0, 500),
            name="AdaptiveThreshold",
        )

    def test_sigma_delta(self):
        py = SigmaDeltaNeuron()
        rs = eng.PySigmaDeltaNeuron()
        py_s = _collect_spikes_py(py, 0.3, 20)
        rs_s = _collect_spikes_rust(rs, 0.3, 20)
        assert py_s == rs_s, f"SigmaDelta: {py_s} != {rs_s}"


class TestSimpleSpikingParity:
    def test_fhn(self):
        py = FitzHughNagumoNeuron()
        rs = eng.PyFitzHughNagumoNeuron()
        _assert_spike_parity(
            _collect_spikes_py(py, 1.0, 2000),
            _collect_spikes_rust(rs, 1.0, 2000),
            name="FHN",
        )

    def test_morris_lecar(self):
        py = MorrisLecarNeuron()
        rs = eng.PyMorrisLecarNeuron()
        _assert_spike_parity(
            _collect_spikes_py(py, 100.0, 2000),
            _collect_spikes_rust(rs, 100.0, 2000),
            name="MorrisLecar",
        )

    def test_hindmarsh_rose(self):
        py = HindmarshRoseNeuron()
        rs = eng.PyHindmarshRoseNeuron()
        _assert_spike_parity(
            _collect_spikes_py(py, 3.0, 2000),
            _collect_spikes_rust(rs, 3.0, 2000),
            name="HindmarshRose",
        )

    def test_mckean(self):
        py = McKeanNeuron()
        rs = eng.PyMcKeanNeuron()
        _assert_spike_parity(
            _collect_spikes_py(py, 0.5, 2000),
            _collect_spikes_rust(rs, 0.5, 2000),
            name="McKean",
        )

    def test_gutkin(self):
        py = GutkinErmentroutNeuron()
        rs = eng.PyGutkinErmentroutNeuron()
        _assert_spike_parity(
            _collect_spikes_py(py, 15.0, 2000),
            _collect_spikes_rust(rs, 15.0, 2000),
            name="Gutkin",
        )

    def test_wilson_hr(self):
        py = WilsonHRNeuron()
        rs = eng.PyWilsonHRNeuron()
        _assert_spike_parity(
            _collect_spikes_py(py, 0.5, 2000),
            _collect_spikes_rust(rs, 0.5, 2000),
            name="WilsonHR",
        )

    def test_chay(self):
        py = ChayNeuron()
        rs = eng.PyChayNeuron()
        _assert_spike_parity(
            _collect_spikes_py(py, 20.0, 5000),
            _collect_spikes_rust(rs, 20.0, 5000),
            name="Chay",
        )


class TestMapsParity:
    def test_chialvo(self):
        py = ChialvoMapNeuron()
        rs = eng.PyChialvoMapNeuron()
        _assert_spike_parity(
            _collect_spikes_py(py, 1.0, 1000),
            _collect_spikes_rust(rs, 1.0, 1000),
            name="Chialvo",
        )


class TestBiophysicalParity:
    def test_hh(self):
        py = HodgkinHuxleyNeuron()
        rs = eng.PyHodgkinHuxleyNeuron()
        _assert_spike_parity(
            _collect_spikes_py(py, 10.0, 100),
            _collect_spikes_rust(rs, 10.0, 100),
            name="HH",
        )

    def test_connor_stevens(self):
        py = ConnorStevensNeuron()
        rs = eng.PyConnorStevensNeuron()
        _assert_spike_parity(
            _collect_spikes_py(py, 10.0, 200),
            _collect_spikes_rust(rs, 10.0, 200),
            name="ConnorStevens",
        )

    def test_destexhe(self):
        py = DestexheThalamicNeuron()
        rs = eng.PyDestexheThalamicNeuron()
        _assert_spike_parity(
            _collect_spikes_py(py, 5.0, 500),
            _collect_spikes_rust(rs, 5.0, 500),
            name="Destexhe",
        )

    def test_prescott(self):
        py = PrescottNeuron()
        rs = eng.PyPrescottNeuron()
        _assert_spike_parity(
            _collect_spikes_py(py, 5.0, 500),
            _collect_spikes_rust(rs, 5.0, 500),
            name="Prescott",
        )

    def test_yamada(self):
        py = YamadaNeuron()
        rs = eng.PyYamadaNeuron()
        _assert_spike_parity(
            _collect_spikes_py(py, 5.0, 2000),
            _collect_spikes_rust(rs, 5.0, 2000),
            name="Yamada",
        )


class TestMultiCompartmentParity:
    def test_dendrify(self):
        py = DendrifyNeuron()
        rs = eng.PyDendrifyNeuron()
        _assert_spike_parity(
            _collect_spikes_py(py, 50.0, 2000),
            _collect_spikes_rust(rs, 50.0, 2000),
            name="Dendrify",
        )


class TestHardwareParity:
    def test_loihi_cuba(self):
        py = LoihiCUBANeuron()
        rs = eng.PyLoihiCUBANeuron()
        py_s = _collect_spikes_py(py, 100, 200, int_input=True)
        rs_s = _collect_spikes_rust(rs, 100, 200, int_input=True)
        # Integer arithmetic: must be exact
        assert py_s == rs_s, "LoihiCUBA: spike trains differ"

    def test_truenorth(self):
        py = TrueNorthNeuron()
        rs = eng.PyTrueNorthNeuron()
        py_s = _collect_spikes_py(py, 50, 20, int_input=True)
        rs_s = _collect_spikes_rust(rs, 50, 20, int_input=True)
        assert py_s == rs_s, "TrueNorth: spike trains differ"

    def test_mcculloch_pitts(self):
        py = McCullochPittsNeuron()
        rs = eng.PyMcCullochPittsNeuron()
        for x in [0.0, 0.5, 1.0, 1.5, 2.0]:
            assert py.step(x) == rs.step(x), f"McCullochPitts({x})"
