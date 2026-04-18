# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Neuromorphic symbolic reasoning engine

"""Spike-based logic gates, registers, ALU. Turing-complete spike computation."""

from .spike_logic import SpikeGate, SpikeRegister, SpikeALU, spike_sort

__all__ = ["SpikeGate", "SpikeRegister", "SpikeALU", "spike_sort"]
