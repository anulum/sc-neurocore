# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Wetware MEA mapper

"""Map neuron populations to wetware Multi-Electrode Array (MEA) sites."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MEAMapping:
    """Mapping of neuron populations to MEA electrodes.

    Attributes
    ----------
    electrode_count : int
    stimulation_freq_hz : float
    voltage_amplitude_mv : float
    spatial_density : str
    """

    electrode_count: int
    stimulation_freq_hz: float
    voltage_amplitude_mv: float
    spatial_density: str


def map_wetware_mea(populations: int, connectivity: float) -> MEAMapping:
    """Map neuron populations to wetware Multi-Electrode Array (MEA) sites."""
    electrodes = min(1024, populations * int(1.0 / max(0.01, connectivity)))

    if connectivity > 0.5:
        freq = 40.0
        amp = 150.0
    else:
        freq = 8.0
        amp = 200.0

    density = "High" if electrodes > 512 else "Standard"

    return MEAMapping(
        electrode_count=electrodes,
        stimulation_freq_hz=freq,
        voltage_amplitude_mv=amp,
        spatial_density=density,
    )
