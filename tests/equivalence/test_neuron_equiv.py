# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Strict blueprint semantics tests for FixedPointLIFNeuron

"""Strict blueprint semantics tests for FixedPointLIFNeuron."""

from dataclasses import dataclass

import pytest

pytest.importorskip("sc_neurocore_engine", reason="Rust engine not built", exc_type=ImportError)

from sc_neurocore_engine import FixedPointLIFNeuron as V3Lif


def _mask(value: int, width: int) -> int:
    """Apply signed two's-complement wrapping at the independent reference width."""
    m = (1 << width) - 1
    v = value & m
    if v >= (1 << (width - 1)):
        v -= 1 << width
    return v


@dataclass
class _BlueprintLif:
    """Compute native LIF transitions independently with explicit integer masking."""

    data_width: int = 16
    fraction: int = 8
    v_rest: int = 0
    v_reset: int = 0
    v_threshold: int = 256
    refractory_period: int = 2

    def __post_init__(self) -> None:
        """Initialise reference voltage and its refractory counter."""
        self.v = self.v_rest
        self.refractory_counter = 0

    def step(self, leak_k: int, gain_k: int, I_t: int, noise_in: int = 0) -> tuple[int, int]:
        """Advance the integer reference in the native leak, input and threshold order."""
        W = self.data_width
        if self.refractory_counter > 0:
            self.refractory_counter -= 1
            self.v = self.v_rest
            return 0, _mask(self.v, W)

        diff = _mask(self.v_rest - self.v, 2 * W)
        dv_leak = _mask(diff * leak_k >> self.fraction, W)
        dv_in = _mask(I_t * gain_k >> self.fraction, W)
        v_next = _mask(self.v + dv_leak + dv_in + noise_in, W)

        if v_next >= self.v_threshold:
            self.v = self.v_reset
            self.refractory_counter = self.refractory_period
            return 1, _mask(self.v_reset, W)
        self.v = v_next
        return 0, _mask(v_next, W)


class TestLIFBlueprintSemantics:
    """Compare actual native samples and state with independent integer expectations."""

    def test_100_steps_constant_input(self) -> None:
        """Match every emitted spike and voltage across one hundred actual native steps."""
        ref = _BlueprintLif()
        v3 = V3Lif()
        for t in range(100):
            exp_spike, exp_v = ref.step(leak_k=20, gain_k=256, I_t=128, noise_in=0)
            got_spike, got_v = v3.step(leak_k=20, gain_k=256, I_t=128, noise_in=0)
            assert got_spike == exp_spike, f"Spike mismatch at step {t}"
            assert got_v == exp_v, f"Voltage mismatch at step {t}: expected={exp_v}, got={got_v}"

    def test_refractory_override_order(self) -> None:
        """Retain refractory update ordering across repeated native spikes."""
        ref = _BlueprintLif(refractory_period=5)
        v3 = V3Lif(refractory_period=5)
        for t in range(200):
            exp_spike, exp_v = ref.step(20, 256, 200, 0)
            got_spike, got_v = v3.step(20, 256, 200, 0)
            assert got_spike == exp_spike, f"Spike mismatch at step {t}"
            assert got_v == exp_v, f"Voltage mismatch at step {t}"

    def test_state_dictionary_tracks_spikes_and_both_reset_aliases(self) -> None:
        """Observe actual native dictionary state through spiking, reset and reset_state."""
        lif = V3Lif(v_rest=-32, v_reset=-16, v_threshold=0, refractory_period=3)
        initial = {"v": -32, "refractory_counter": 0}
        assert lif.get_state() == initial
        assert lif.step(0, 256, 64) == (1, -16)
        assert lif.get_state() == {"v": -16, "refractory_counter": 3}
        lif.reset()
        assert lif.get_state() == initial
        assert lif.step(0, 256, 64) == (1, -16)
        lif.reset_state()
        assert lif.get_state() == initial
