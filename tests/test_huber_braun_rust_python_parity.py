# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Rust/Python C libm parity witness for HuberBraunNeuron

"""Verify bit-exact C libm and Rust engine parity for ``HuberBraunNeuron``.

Checks that deterministic ``HuberBraunNeuron(eta=0.0)`` matches the compiled
Rust engine bit-for-bit over 500 steps where scalar NumPy SIMD ``exp`` differed
from C ``libm``.
"""

from __future__ import annotations

from typing import Any

from sc_neurocore.neurons.models.huber_braun import HuberBraunNeuron
from tests.engine_requirement import require_engine

ENGINE: Any = require_engine()


class TestHuberBraunLibmRustParityWitness:
    """Bit-for-bit C libm and Rust engine parity for ``HuberBraunNeuron``."""

    def test_huber_braun_deterministic_matches_rust_engine_bit_for_bit(self) -> None:
        """``HuberBraunNeuron(eta=0.0)`` matches the Rust engine bit-for-bit over 500 steps."""
        py_hb = HuberBraunNeuron(eta=0.0)
        rust_hb = ENGINE.HuberBraunNeuron()

        for _ in range(500):
            assert py_hb.step(5.0) == rust_hb.step(5.0)
            rust_state = rust_hb.get_state()
            assert py_hb.v == rust_state["v"]
            assert py_hb.a_sd == rust_state["a_sd"]
            assert py_hb.a_sr == rust_state["a_sr"]
