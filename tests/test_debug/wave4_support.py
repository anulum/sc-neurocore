# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_wave4.py

from __future__ import annotations

"""Comprehensive tests mirroring Rust and Go test suites."""
import pytest
from sc_neurocore.debug.sc_doctor import ScDoctor
from sc_neurocore.debug.hil_client import (
    SpikeEvent,
    SpikeRingBuffer,
    LayerAggregator,
    ErrorBudget,
    CorrelationWindow,
    PrecisionTracker,
    EventFilter,
    filter_events,
    TriggerCondition,
    TriggerLog,
    RateLimiter,
    check_health,
    export_csv,
    export_json,
)
from sc_neurocore.bridges.aer_router import (
    SpikePacket,
    AERRouter,
    PACKET_SIZE,
)

__all__ = ['pytest', 'ScDoctor', 'SpikeEvent', 'SpikeRingBuffer', 'LayerAggregator', 'ErrorBudget', 'CorrelationWindow', 'PrecisionTracker', 'EventFilter', 'filter_events', 'TriggerCondition', 'TriggerLog', 'RateLimiter', 'check_health', 'export_csv', 'export_json', 'SpikePacket', 'AERRouter', 'PACKET_SIZE']
