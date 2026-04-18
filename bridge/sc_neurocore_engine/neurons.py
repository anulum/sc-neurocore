# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Drop-in replacement for

"""Drop-in replacement for sc_neurocore.neurons.FixedPointLIFNeuron."""

from sc_neurocore_engine.sc_neurocore_engine import FixedPointLif as _RustLif


class FixedPointLIFNeuron:
    """
    Fixed-point LIF neuron using Rust backend.
    API-compatible with sc_neurocore.neurons.FixedPointLIFNeuron.
    """

    def __init__(
        self,
        data_width=16,
        fraction=8,
        v_rest=0,
        v_reset=0,
        v_threshold=256,
        refractory_period=2,
    ):
        self._engine = _RustLif(
            data_width,
            fraction,
            v_rest,
            v_reset,
            v_threshold,
            refractory_period,
        )

    def step(self, leak_k, gain_k, I_t, noise_in=0):
        """Return (spike: int, v_out: int)."""
        return self._engine.step(leak_k, gain_k, I_t, noise_in)

    def reset(self):
        self._engine.reset()

    def reset_state(self):
        self.reset()

    def get_state(self):
        return self._engine.get_state()
