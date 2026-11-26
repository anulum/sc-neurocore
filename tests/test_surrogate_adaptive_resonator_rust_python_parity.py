# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Rust/Python C libm parity witnesses for surrogate and adaptive neurons

"""Verify bit-exact C libm and Rust engine parity for surrogate and adaptive neurons.

Covers ``KLIFNeuron``, ``SuperSpikeNeuron``, and ``LiquidTimeConstantNeuron`` at
inputs where scalar NumPy SIMD routines differ by 1 ULP from C ``libm``.
"""

from __future__ import annotations

import math
from typing import Any

import pytest

from sc_neurocore.neurons.models.klif import KLIFNeuron
from sc_neurocore.neurons.models.ltc import LiquidTimeConstantNeuron
from sc_neurocore.neurons.models.superspike_neuron import SuperSpikeNeuron
from tests.engine_requirement import require_engine

ENGINE: Any = require_engine()


class TestLibmRustParityWitness:
    """Bit-for-bit C libm and Rust engine parity at arguments where NumPy differs."""

    def test_klif_and_superspike_decay_match_libm_and_rust_engine_bit_for_bit(self) -> None:
        """``KLIFNeuron`` and ``SuperSpikeNeuron`` match Rust ``(-dt / tau).exp()`` bit-for-bit.

        ``py_klif.v`` is compared against the compiled engine ``SuperSpikeNeuron``
        state ``v`` at ``dt = 1.786387917040303e-05`` because the engine ``KLIFNeuron``
        binding exposes only default parameters (checked separately below); at
        ``tau = tau_m = 1.0``, ``k = 1.0``, ``v_reset = 0.0``, and ``v_threshold = 1.0``
        both share the exact sub-threshold decay ``alpha == math.exp(-dt)``.
        """
        dt = 1.786387917040303e-05
        expected_alpha = math.exp(-dt)

        py_klif = KLIFNeuron(tau=1.0, k=1.0, dt=dt)
        py_ss = SuperSpikeNeuron(tau_m=1.0, tau_e=1.0, dt=dt)
        rust_ss = ENGINE.SuperSpikeNeuron(tau_m=1.0, tau_e=1.0, dt=dt)

        assert py_klif.alpha == expected_alpha
        assert py_ss.alpha_m == expected_alpha
        assert py_ss.alpha_e == expected_alpha

        for _ in range(2):
            assert py_klif.step(0.5) == 0
            assert py_ss.step(0.5) == rust_ss.step(0.5)

        rust_state = rust_ss.get_state()
        assert py_klif.v == rust_state["v"]
        assert py_ss.v == rust_state["v"]
        assert py_ss.trace == rust_state["trace"]

        py_klif_default = KLIFNeuron()
        rust_klif_default = ENGINE.KLIFNeuron()
        for _ in range(2):
            assert py_klif_default.step(0.5) == rust_klif_default.step(0.5)
        assert py_klif_default.v == rust_klif_default.get_state()["v"]

    @pytest.mark.parametrize("current", [-3.572775834080606e-05, 0.7])
    def test_ltc_step_matches_rust_engine_bit_for_bit(self, current: float) -> None:
        """``LiquidTimeConstantNeuron`` matches Rust ``exp`` and ``tanh`` bit-for-bit."""
        py_ltc = LiquidTimeConstantNeuron()
        rust_ltc = ENGINE.LiquidTimeConstantNeuron()

        assert py_ltc.step(current) == rust_ltc.step(current)
        assert py_ltc.x == rust_ltc.get_state()["x"]
