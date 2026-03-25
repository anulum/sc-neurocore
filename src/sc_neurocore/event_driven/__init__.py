# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Event-driven asynchronous SNN simulation

"""Event-driven simulation: only update neurons with pending events.
10,000x speedup for sparse networks vs clock-driven simulation."""

from .simulator import EventDrivenSimulator, SpikeEvent, EventStats

__all__ = ["EventDrivenSimulator", "SpikeEvent", "EventStats"]
