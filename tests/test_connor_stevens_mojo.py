# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Connor-Stevens Mojo C-ABI parity tests

"""Real-surface parity tests for the compiled Connor-Stevens Mojo kernel."""

from __future__ import annotations

import math
import os

import numpy as np
import pytest

from sc_neurocore.neurons.models import connor_stevens as connor
from sc_neurocore.neurons.models.connor_stevens import ConnorStevensNeuron

_MOJO_AVAILABLE = connor._ensure_mojo_loaded()
_ENROLLED_TRACE_ATOL = 2.0e-6


@pytest.mark.skipif(not _MOJO_AVAILABLE, reason="compiled Connor-Stevens Mojo kernel unavailable")
class TestConnorStevensMojoParity:
    """Exercise the public Python-to-Mojo simulation boundary."""

    @pytest.mark.parametrize(
        ("current", "expected_spikes"),
        [(0.0, 0), (10.0, 2), (20.0, 9)],
    )
    def test_enrolled_golden_trace_and_events(self, current: float, expected_spikes: int) -> None:
        """Match the established 100-macro-step Python golden envelope."""
        reference = ConnorStevensNeuron()
        accelerated = ConnorStevensNeuron()
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
            [
                accelerated.v,
                accelerated.m,
                accelerated.h,
                accelerated.n,
                accelerated.a,
                accelerated.b,
            ],
            [reference.v, reference.m, reference.h, reference.n, reference.a, reference.b],
            atol=_ENROLLED_TRACE_ATOL,
            rtol=0.0,
        )

    def test_non_default_state_and_parameters_cross_the_c_abi(self) -> None:
        """Prove the Mojo dispatcher carries the maintained parameter surface."""
        configuration = {
            "v": -62.0,
            "m": 0.05,
            "h": 0.84,
            "n": 0.22,
            "a": 0.41,
            "b": 0.27,
            "g_a": 40.0,
            "dt": 0.02,
        }
        reference = ConnorStevensNeuron(**configuration)
        accelerated = ConnorStevensNeuron(**configuration)
        reference_trace, reference_spikes = reference.simulate(20, 8.5, backend="python")
        mojo_trace, mojo_spikes = accelerated.simulate(20, 8.5, backend="mojo")

        assert mojo_spikes == reference_spikes
        np.testing.assert_allclose(mojo_trace, reference_trace, atol=1.0e-6, rtol=0.0)
        np.testing.assert_allclose(
            [
                accelerated.v,
                accelerated.m,
                accelerated.h,
                accelerated.n,
                accelerated.a,
                accelerated.b,
            ],
            [reference.v, reference.m, reference.h, reference.n, reference.a, reference.b],
            atol=1.0e-6,
            rtol=0.0,
        )

    def test_empty_run_preserves_the_complete_state(self) -> None:
        """Return an empty trace without discarding any state component."""
        neuron = ConnorStevensNeuron(v=-67.0, m=0.02, h=0.97, n=0.12, a=0.45, b=0.13)
        before = (neuron.v, neuron.m, neuron.h, neuron.n, neuron.a, neuron.b)
        trace, spikes = neuron.simulate(0, 4.0, backend="mojo")

        assert trace.shape == (0,)
        assert spikes == 0
        assert (neuron.v, neuron.m, neuron.h, neuron.n, neuron.a, neuron.b) == before

    def test_kernel_rejects_non_finite_input_at_the_c_boundary(self) -> None:
        """Reject invalid input inside Mojo even when the Python guard is bypassed."""
        assert connor._mojo_lib is not None
        neuron = ConnorStevensNeuron()
        output = np.full(7, -999.0, dtype=np.float64)
        result = connor._mojo_lib.connor_stevens_simulate_c(
            neuron.v,
            neuron.m,
            neuron.h,
            neuron.n,
            neuron.a,
            neuron.b,
            neuron.g_na,
            neuron.g_k,
            neuron.g_a,
            neuron.g_l,
            neuron.e_na,
            neuron.e_k,
            neuron.e_a,
            neuron.e_l,
            neuron.c_m,
            neuron.dt,
            neuron.v_threshold,
            1,
            math.nan,
            output.ctypes.data,
        )

        assert result == -1
        np.testing.assert_array_equal(output, np.full(7, -999.0, dtype=np.float64))

    def test_kernel_rejection_does_not_commit_instance_state(self) -> None:
        """Translate a C-ABI rejection into a typed, mutation-free failure."""
        neuron = ConnorStevensNeuron()
        neuron.a = 1.6
        before = (neuron.v, neuron.m, neuron.h, neuron.n, neuron.a, neuron.b)

        with pytest.raises(FloatingPointError, match="kernel rejected"):
            neuron.simulate(1, 5.0, backend="mojo")

        assert (neuron.v, neuron.m, neuron.h, neuron.n, neuron.a, neuron.b) == before


def test_requested_mojo_backend_fails_closed_when_library_is_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Report an actionable build command instead of falling back to Python."""
    monkeypatch.setattr(connor, "_mojo_lib", None)
    monkeypatch.setattr(connor, "_HAS_MOJO", False)
    monkeypatch.setattr(os.path, "isfile", lambda _path: False)

    with pytest.raises(RuntimeError, match="libconnor_stevens.so is not built"):
        ConnorStevensNeuron().simulate(1, 0.0, backend="mojo")


def test_unknown_backend_is_rejected() -> None:
    """Keep explicit backend selection fail-closed."""
    with pytest.raises(ValueError, match="auto/python/rust/mojo"):
        ConnorStevensNeuron().simulate(1, 0.0, backend="gpu")
