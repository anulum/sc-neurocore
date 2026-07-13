# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Hodgkin-Huxley Mojo C-ABI parity tests

"""Real-surface parity tests for the compiled Hodgkin-Huxley Mojo kernel."""

from __future__ import annotations

import math
import os
import ctypes

import numpy as np
import pytest

from sc_neurocore.neurons.models import hodgkin_huxley as hodgkin
from sc_neurocore.neurons.models.hodgkin_huxley import HodgkinHuxleyNeuron

_MOJO_AVAILABLE = hodgkin._ensure_mojo_loaded()
_ENROLLED_TRACE_ATOL = 2.0e-9


@pytest.mark.skipif(not _MOJO_AVAILABLE, reason="compiled Hodgkin-Huxley Mojo kernel unavailable")
class TestHodgkinHuxleyMojoParity:
    """Exercise the public Python-to-Mojo simulation boundary."""

    @pytest.mark.parametrize(
        ("current", "expected_spikes"),
        [(0.0, 0), (10.0, 6), (20.0, 9)],
    )
    def test_enrolled_golden_trace_and_events(self, current: float, expected_spikes: int) -> None:
        """Match the established 100-macro-step baseline-Euler envelope."""
        reference = HodgkinHuxleyNeuron()
        accelerated = HodgkinHuxleyNeuron()
        reference_trace, reference_spikes = reference.simulate(100, current, backend="python")
        mojo_trace, mojo_spikes = accelerated.simulate(100, current, backend="mojo")

        assert reference_spikes == expected_spikes
        assert mojo_spikes == reference_spikes
        np.testing.assert_allclose(
            mojo_trace,
            reference_trace,
            atol=_ENROLLED_TRACE_ATOL,
            rtol=0.0,
        )
        np.testing.assert_allclose(
            [accelerated.v, accelerated.m, accelerated.h, accelerated.n],
            [reference.v, reference.m, reference.h, reference.n],
            atol=_ENROLLED_TRACE_ATOL,
            rtol=0.0,
        )

    def test_non_default_state_and_parameters_cross_the_c_abi(self) -> None:
        """Prove the dispatcher carries every maintained numeric field."""
        configuration = {
            "v": -62.0,
            "m": 0.08,
            "h": 0.72,
            "n": 0.27,
            "c_m": 1.1,
            "g_na": 115.0,
            "g_k": 34.0,
            "g_l": 0.28,
            "e_na": 52.0,
            "e_k": -75.0,
            "e_l": -53.0,
            "dt": 0.02,
            "v_threshold": -2.0,
        }

        def configured() -> HodgkinHuxleyNeuron:
            return HodgkinHuxleyNeuron(
                v=configuration["v"],
                m=configuration["m"],
                h=configuration["h"],
                n=configuration["n"],
                c_m=configuration["c_m"],
                g_na=configuration["g_na"],
                g_k=configuration["g_k"],
                g_l=configuration["g_l"],
                e_na=configuration["e_na"],
                e_k=configuration["e_k"],
                e_l=configuration["e_l"],
                dt=configuration["dt"],
                v_threshold=configuration["v_threshold"],
            )

        reference = configured()
        accelerated = configured()
        reference_trace, reference_spikes = reference.simulate(20, 8.5, backend="python")
        mojo_trace, mojo_spikes = accelerated.simulate(20, 8.5, backend="mojo")

        assert mojo_spikes == reference_spikes
        np.testing.assert_allclose(mojo_trace, reference_trace, atol=5.0e-10, rtol=0.0)
        np.testing.assert_allclose(
            [accelerated.v, accelerated.m, accelerated.h, accelerated.n],
            [reference.v, reference.m, reference.h, reference.n],
            atol=5.0e-10,
            rtol=0.0,
        )

    def test_half_even_substep_rounding_matches_python(self) -> None:
        """Keep Python's round(2.5)=2 schedule for non-default dt."""
        reference = HodgkinHuxleyNeuron(dt=0.4)
        accelerated = HodgkinHuxleyNeuron(dt=0.4)
        reference_trace, reference_spikes = reference.simulate(1, 0.0, backend="python")
        mojo_trace, mojo_spikes = accelerated.simulate(1, 0.0, backend="mojo")

        assert round(1.0 / reference.dt) == 2
        assert mojo_spikes == reference_spikes
        np.testing.assert_allclose(mojo_trace, reference_trace, atol=5.0e-12, rtol=0.0)

    @pytest.mark.parametrize("voltage", [-40.0, -55.0])
    def test_singular_opening_rate_limits_match_python(self, voltage: float) -> None:
        """Exercise both analytic alpha-rate limits through the compiled ABI."""
        reference = HodgkinHuxleyNeuron(v=voltage)
        accelerated = HodgkinHuxleyNeuron(v=voltage)
        reference_trace, reference_spikes = reference.simulate(1, 0.0, backend="python")
        mojo_trace, mojo_spikes = accelerated.simulate(1, 0.0, backend="mojo")

        assert mojo_spikes == reference_spikes
        np.testing.assert_allclose(mojo_trace, reference_trace, atol=5.0e-11, rtol=0.0)

    def test_empty_run_preserves_the_complete_state(self) -> None:
        """Return an empty trace without discarding any state component."""
        neuron = HodgkinHuxleyNeuron(v=-64.0, m=0.06, h=0.58, n=0.33)
        before = (neuron.v, neuron.m, neuron.h, neuron.n)
        trace, spikes = neuron.simulate(0, 4.0, backend="mojo")

        assert trace.shape == (0,)
        assert spikes == 0
        assert (neuron.v, neuron.m, neuron.h, neuron.n) == before

    def test_kernel_rejects_non_finite_input_at_the_c_boundary(self) -> None:
        """Reject invalid input inside Mojo even when the Python guard is bypassed."""
        assert hodgkin._mojo_lib is not None
        neuron = HodgkinHuxleyNeuron()
        output = np.full(5, -999.0, dtype=np.float64)
        result = hodgkin._mojo_lib.hodgkin_huxley_simulate_c(
            neuron.v,
            neuron.m,
            neuron.h,
            neuron.n,
            neuron.c_m,
            neuron.g_na,
            neuron.g_k,
            neuron.g_l,
            neuron.e_na,
            neuron.e_k,
            neuron.e_l,
            neuron.dt,
            neuron.v_threshold,
            1,
            math.nan,
            output.ctypes.data,
        )

        assert result == -1
        np.testing.assert_array_equal(output, np.full(5, -999.0, dtype=np.float64))

    def test_kernel_rejection_does_not_commit_instance_state(self) -> None:
        """Translate a C-ABI rejection into a typed, mutation-free failure."""
        neuron = HodgkinHuxleyNeuron()
        neuron.m = 1.2
        before = (neuron.v, neuron.m, neuron.h, neuron.n)

        with pytest.raises(FloatingPointError, match="kernel rejected"):
            neuron.simulate(1, 5.0, backend="mojo")

        assert (neuron.v, neuron.m, neuron.h, neuron.n) == before


def test_requested_mojo_backend_rejects_non_baseline_integrator() -> None:
    """Do not silently run baseline Euler for an RK4-configured instance."""
    neuron = HodgkinHuxleyNeuron(integrator="rk4")
    before = (neuron.v, neuron.m, neuron.h, neuron.n)

    with pytest.raises(RuntimeError, match="baseline_euler"):
        neuron.simulate(1, 0.0, backend="mojo")

    assert (neuron.v, neuron.m, neuron.h, neuron.n) == before


def test_requested_mojo_backend_fails_closed_when_library_is_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Report an actionable build command instead of falling back to Python."""
    monkeypatch.setattr(hodgkin, "_mojo_lib", None)
    monkeypatch.setattr(hodgkin, "_HAS_MOJO", False)
    monkeypatch.setattr(os.path, "isfile", lambda _path: False)

    with pytest.raises(RuntimeError, match="libhodgkin_huxley.so is not built"):
        HodgkinHuxleyNeuron().simulate(1, 0.0, backend="mojo")


@pytest.mark.parametrize("failure", ["load", "symbol"])
def test_mojo_loader_rejects_invalid_library_boundaries(
    failure: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep an unloadable library or missing ABI symbol unavailable."""
    monkeypatch.setattr(hodgkin, "_mojo_lib", None)
    monkeypatch.setattr(hodgkin, "_HAS_MOJO", False)
    monkeypatch.setattr(os.path, "isfile", lambda _path: True)
    if failure == "load":

        def reject_load(_path: str) -> object:
            raise OSError("invalid shared library")

        monkeypatch.setattr(ctypes, "CDLL", reject_load)
    else:
        monkeypatch.setattr(ctypes, "CDLL", lambda _path: object())

    assert hodgkin._ensure_mojo_loaded() is False
    assert hodgkin._mojo_lib is None
    assert hodgkin._HAS_MOJO is False


def test_unknown_backend_is_rejected() -> None:
    """Keep explicit backend selection fail-closed."""
    with pytest.raises(ValueError, match="auto/python/rust/mojo"):
        HodgkinHuxleyNeuron().simulate(1, 0.0, backend="gpu")
