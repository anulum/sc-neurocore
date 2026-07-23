# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_comm_aer.py

from __future__ import annotations

import struct
import numpy as np
from sc_neurocore.comm.aer_udp import (
    AEREvent,
    AERSender,
    AERReceiver,
    MAGIC,
    HEADER_FMT,
    EVENT_FMT,
    HEADER_SIZE,
    EVENT_SIZE,
    MAX_EVENTS_PER_PACKET,
)

__all__ = ['struct', 'np', 'AEREvent', 'AERSender', 'AERReceiver', 'MAGIC', 'HEADER_FMT', 'EVENT_FMT', 'HEADER_SIZE', 'EVENT_SIZE', 'MAX_EVENTS_PER_PACKET']
