# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Sensor interfaces for neuromorphic processing

"""Sensor interfaces: event cameras (DVS), audio, and other neuromorphic sensors."""

from .dvs import DVSLoader, events_to_spike_trains, events_to_frames

__all__ = ["DVSLoader", "events_to_spike_trains", "events_to_frames"]
