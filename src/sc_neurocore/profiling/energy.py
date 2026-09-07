# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Operation-count energy estimate for a layer call

"""Estimate the energy of a layer call from the operations it performs.

Nothing here measures power. The profiler counts logic operations and memory
bits, multiplies each count by a fixed per-operation energy for a 45 nm CMOS
equivalent, and sums them. The result is an arithmetic consequence of the
counts and those constants, and it moves only when a count or a constant moves
— never with voltage, frequency, temperature, process corner, or the machine
the code runs on.

A measured figure needs an instrument. Where one is required, a power receipt
naming the instrument and its calibration is the evidence; this module is not
that, and its output must not be reported as though it were.

See Also
--------
Horowitz (2014), "Computing's Energy Problem (and what we can do about it)",
ISSCC keynote, cited by ``docs/tutorials/41_energy_estimation.md`` as the
reference for the project's energy estimation. The specific constants below are
of that family and order of magnitude; **this repository records no source for
their exact values**, which is stated here rather than papered over with a
citation they may not come from.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any
from dataclasses import dataclass


@dataclass
class EnergyMetrics:
    """Running operation counts and the per-operation energies applied to them.

    Attributes
    ----------
    E_AND, E_XOR, E_ADD : float
        Energy in joules attributed to one gate operation at a 45 nm CMOS
        equivalent. Assumed values; see the module docstring on provenance.
    E_MEM : float
        Energy in joules attributed to reading one bit of memory, same basis.
    total_ops_and, total_ops_xor, total_bits_mem : int
        Counts accumulated since the last :meth:`reset`.
    """

    # 45nm CMOS technology estimates (joules). Assumed, not measured.
    E_AND: float = 0.1e-15  # 0.1 fJ
    E_XOR: float = 0.15e-15  # 0.15 fJ
    E_ADD: float = 0.5e-15  # 0.5 fJ (1-bit)
    E_MEM: float = 5.0e-15  # 5 fJ per bit read

    total_ops_and: int = 0
    total_ops_xor: int = 0
    total_bits_mem: int = 0

    def reset(self) -> None:
        """Zero every accumulated count, leaving the energy constants alone."""
        self.total_ops_and = 0
        self.total_ops_xor = 0
        self.total_bits_mem = 0

    def estimate_energy(self) -> float:
        """Return the energy the accumulated counts imply, in joules.

        Returns
        -------
        float
            ``counts x per-operation constants``. An estimate, not a
            measurement: no power was observed to produce it.
        """
        e_logic = (self.total_ops_and * self.E_AND) + (self.total_ops_xor * self.E_XOR)
        e_mem = self.total_bits_mem * self.E_MEM
        return e_logic + e_mem

    def co2_emission_g(self, carbon_intensity_g_per_kwh: float = 475) -> float:
        """Return the CO2 the estimated energy implies, in grams.

        Two assumptions compound here: the per-operation energies behind
        :meth:`estimate_energy`, and the grid carbon intensity below. The result
        is an estimate over an estimate and carries the uncertainty of both.

        Parameters
        ----------
        carbon_intensity_g_per_kwh : float
            Grams of CO2 per kilowatt-hour attributed to the supplying grid.
            The default is a global-average figure and is not sourced in this
            repository; a regional or measured value should be passed when one
            is known. ``docs/guides/carbon_sustainability.md`` tabulates
            regional intensities for the separate lifecycle estimator.

        Returns
        -------
        float
            Grams of CO2 implied by the estimate and the intensity given.
        """
        # Energy in joules -> kWh -> grams CO2. 1 J = 2.7778e-7 kWh.
        kwh = self.estimate_energy() * 2.7778e-7
        return kwh * carbon_intensity_g_per_kwh


# Global Profiler Instance
profiler = EnergyMetrics()


def track_energy(func: Callable[..., Any]) -> Callable[..., Any]:
    """Wrap a layer call so its operation counts reach the global profiler.

    Parameters
    ----------
    func : Callable
        The layer call to wrap. Its dimensions determine the counts added.

    Returns
    -------
    Callable
        The same call, adding to :data:`profiler` on each invocation. It
        accumulates counts, not measurements.
    """

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        res = func(*args, **kwargs)

        # Determine 'self' object
        # 1. If func is a bound method, it has __self__
        layer_obj = getattr(func, "__self__", None)

        # 2. If used on class def, args[0] is self
        if layer_obj is None and len(args) > 0:
            # Check if args[0] looks like a layer
            if hasattr(args[0], "n_neurons"):
                layer_obj = args[0]

        if (
            layer_obj
            and hasattr(layer_obj, "n_neurons")
            and hasattr(layer_obj, "n_inputs")
            and hasattr(layer_obj, "length")
        ):
            # Dense Layer Ops:
            ops = layer_obj.n_inputs * layer_obj.n_neurons * layer_obj.length
            profiler.total_ops_and += ops

            # Memory Read
            mem = (layer_obj.n_neurons * layer_obj.n_inputs * layer_obj.length) + (
                layer_obj.n_inputs * layer_obj.length
            )
            profiler.total_bits_mem += mem

        return res

    return wrapper
