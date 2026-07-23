# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_event_driven.py

from __future__ import annotations

import numpy as np
from sc_neurocore.event_driven import EventDrivenSimulator, SpikeEvent, EventStats
def _simple_chain(n=5):
    """Linear chain: 0→1→2→3→4."""
    conns = [(i, i + 1, 0.6, 1.0) for i in range(n - 1)]
    return EventDrivenSimulator(n_neurons=n, connectivity=conns, threshold=1.0, tau_mem=20.0)

__all__ = ['np', 'EventDrivenSimulator', 'SpikeEvent', 'EventStats', '_simple_chain']
