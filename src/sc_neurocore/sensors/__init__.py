# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Sensor interfaces for neuromorphic processing

"""Sensor interfaces: event cameras (DVS), audio, and other neuromorphic sensors."""

from .adc_to_spike_kernel import (
    ADCSpikeWindowConfig,
    ADCSpikeWindowResult,
    adc_to_spike_windows,
    adc_to_spike_windows_q,
    available_backends,
    quantise_adc,
)
from .dvs import DVSLoader, events_to_spike_trains, events_to_frames

__all__ = [
    "ADCSpikeWindowConfig",
    "ADCSpikeWindowResult",
    "DVSLoader",
    "adc_to_spike_windows",
    "adc_to_spike_windows_q",
    "available_backends",
    "events_to_frames",
    "events_to_spike_trains",
    "quantise_adc",
]
