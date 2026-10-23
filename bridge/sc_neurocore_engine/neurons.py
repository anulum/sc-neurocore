# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Drop-in replacement for

"""Drop-in replacement for sc_neurocore.neurons.FixedPointLIFNeuron."""

from __future__ import annotations

from typing import TYPE_CHECKING as _TYPE_CHECKING

from sc_neurocore_engine.sc_neurocore_engine import FixedPointLif as _RustLif

if _TYPE_CHECKING:
    from ._binding_types import FixedPointLif as _LifType


class FixedPointLIFNeuron:
    """Fixed-point LIF facade for the native signed-word neuron."""

    def __init__(
        self,
        data_width: int = 16,
        fraction: int = 8,
        v_rest: int = 0,
        v_reset: int = 0,
        v_threshold: int = 256,
        refractory_period: int = 2,
    ) -> None:
        """Construct the native neuron with its fixed-point and refractory configuration.

        Parameters
        ----------
        data_width, fraction : int, optional
            Native word width and fractional-bit count.
        v_rest, v_reset, v_threshold : int, optional
            Raw signed voltage words.
        refractory_period : int, optional
            Number of refractory timesteps after a spike.
        """
        self._engine: _LifType = _RustLif(
            data_width,
            fraction,
            v_rest,
            v_reset,
            v_threshold,
            refractory_period,
        )

    def step(self, leak_k: int, gain_k: int, I_t: int, noise_in: int = 0) -> tuple[int, int]:
        """Advance one native timestep and return its spike and voltage.

        Parameters
        ----------
        leak_k : int
            Fixed-point membrane leak coefficient.
        gain_k : int
            Fixed-point input-current gain.
        I_t : int
            Signed current word for this timestep.
        noise_in : int, optional
            Signed additive noise word, default zero.

        Returns
        -------
        tuple of int
            Spike indicator and signed output voltage, in that order.
        """
        return self._engine.step(leak_k, gain_k, I_t, noise_in)

    def reset(self) -> None:
        """Restore the configured resting voltage and clear the refractory counter."""
        self._engine.reset()

    def reset_state(self) -> None:
        """Restore voltage and refractory state through the native reset operation."""
        self.reset()

    def get_state(self) -> dict[str, int]:
        """Return the native named voltage and refractory state.

        Returns
        -------
        dict of str to int
            Current v and refractory_counter fields as Python integers.
        """
        return self._engine.get_state()
