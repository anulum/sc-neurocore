# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Multi-timescale SNN architecture

"""Multi-timescale SNN: per-synapse time constants + multi-clock scheduling."""

from .multi_clock import MultiClockSNN, ClockDomain, HetSynLayer

__all__ = ["MultiClockSNN", "ClockDomain", "HetSynLayer"]
