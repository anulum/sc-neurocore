# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Registry adapter classes for importer modules

"""Registry adapter classes for SC-NeuroCore importer modules.

The historical importer APIs are plain functions.  These small classes make the
same functionality discoverable through the class-oriented
``ComponentRegistry`` and Python packaging entry points without changing the
function-level APIs used by existing callers.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from sc_neurocore.adapters.neuroml import ImportedCell
    from sc_neurocore.adapters.sonata import SONATANetwork


class NeuroMLImporter:
    """Class-oriented registry surface for NeuroML 2 cell imports."""

    @staticmethod
    def import_cells(path: str | Path) -> list[ImportedCell]:
        """Parse a NeuroML 2 XML file into imported cell definitions.

        Parameters
        ----------
        path : str or Path
            Local ``.nml`` or ``.xml`` file containing NeuroML 2 cell elements.

        Returns
        -------
        list of ImportedCell
            Parsed cell definitions ready for SC-NeuroCore neuron creation.
        """
        from sc_neurocore.adapters.neuroml import import_neuroml

        return import_neuroml(path)

    @staticmethod
    def create_neuron(cell: ImportedCell) -> Any:
        """Instantiate a neuron from a parsed NeuroML cell definition.

        Parameters
        ----------
        cell : ImportedCell
            Parsed NeuroML cell definition returned by :meth:`import_cells`.

        Returns
        -------
        Any
            Concrete SC-NeuroCore neuron instance selected by the cell type.
        """
        from sc_neurocore.adapters.neuroml import create_neuron

        return create_neuron(cell)


class SONATAImporter:
    """Class-oriented registry surface for SONATA network imports."""

    @staticmethod
    def import_network(
        nodes_path: str | Path,
        edges_path: str | Path | None = None,
    ) -> SONATANetwork:
        """Import a SONATA network from nodes and optional edges files.

        Parameters
        ----------
        nodes_path : str or Path
            Path to a SONATA ``nodes.h5`` file.
        edges_path : str or Path, optional
            Optional path to a SONATA ``edges.h5`` file.

        Returns
        -------
        SONATANetwork
            Parsed nodes, edges, population metadata, and connectivity helpers.
        """
        from sc_neurocore.adapters.sonata import import_sonata

        return import_sonata(nodes_path, edges_path)


class SpikeInterfaceImporter:
    """Class-oriented registry surface for spike-train conversion imports."""

    @staticmethod
    def to_bitstreams(
        spike_times: dict[int, np.ndarray[Any, Any]],
        duration_ms: float,
        dt: float = 1.0,
    ) -> np.ndarray[Any, Any]:
        """Convert spike times to a binary bitstream matrix.

        Parameters
        ----------
        spike_times : dict[int, np.ndarray]
            Mapping from unit identifier to spike times in milliseconds.
        duration_ms : float
            Recording duration in milliseconds.
        dt : float, default=1.0
            Time-bin width in milliseconds.

        Returns
        -------
        np.ndarray
            Binary matrix shaped ``(n_units, n_bins)`` with dtype ``uint8``.
        """
        from sc_neurocore.adapters.spikeinterface import spike_trains_to_bitstreams

        return spike_trains_to_bitstreams(spike_times, duration_ms, dt)

    @staticmethod
    def to_population_input(
        spike_times: dict[int, np.ndarray[Any, Any]],
        duration_ms: float,
        dt: float = 1.0,
    ) -> np.ndarray[Any, Any]:
        """Convert spike times to ``Population.step_all`` input.

        Parameters
        ----------
        spike_times : dict[int, np.ndarray]
            Mapping from unit identifier to spike times in milliseconds.
        duration_ms : float
            Recording duration in milliseconds.
        dt : float, default=1.0
            Time-bin width in milliseconds.

        Returns
        -------
        np.ndarray
            Floating-point array shaped ``(n_timesteps, n_units)``.
        """
        from sc_neurocore.adapters.spikeinterface import spike_trains_to_population_input

        return spike_trains_to_population_input(spike_times, duration_ms, dt)

    @staticmethod
    def to_probabilities(
        spike_times: dict[int, np.ndarray[Any, Any]],
        duration_ms: float,
        max_rate_hz: float = 100.0,
    ) -> np.ndarray[Any, Any]:
        """Convert spike trains into bounded stochastic-computing probabilities.

        Parameters
        ----------
        spike_times : dict[int, np.ndarray]
            Mapping from unit identifier to spike times in milliseconds.
        duration_ms : float
            Recording duration in milliseconds.
        max_rate_hz : float, default=100.0
            Firing rate that maps to probability ``1.0``.

        Returns
        -------
        np.ndarray
            Probability vector shaped ``(n_units,)`` and bounded in ``[0, 1]``.
        """
        from sc_neurocore.adapters.spikeinterface import firing_rates_to_sc_probs

        return firing_rates_to_sc_probs(spike_times, duration_ms, max_rate_hz)


__all__ = [
    "NeuroMLImporter",
    "SONATAImporter",
    "SpikeInterfaceImporter",
]
