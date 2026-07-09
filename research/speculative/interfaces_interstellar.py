# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Source/config provenance header


import logging
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass
class Packet:
    id: int
    data: np.ndarray
    ttl: int = 1000  # Time To Live (Years)


@dataclass
class InterstellarDTN:
    """
    Delay-Tolerant Networking (DTN) for Interstellar Communication.
    Uses 'Store-and-Forward' architecture.
    """

    node_id: str
    buffer: List[Packet] = field(default_factory=list)
    link_availability: float = 0.1  # Probability of link up

    def receive(self, packet: Packet):
        """Store packet in non-volatile memory."""
        self.buffer.append(packet)
        logger.debug("DTN Node %s: Packet %d buffered.", self.node_id, packet.id)

    def step(self) -> Optional[Packet]:
        """
        Attempt to forward a packet.
        """
        if not self.buffer:
            return None

        # Check link
        if np.random.random() < self.link_availability:
            # Link UP
            packet = self.buffer.pop(0)  # FIFO
            packet.ttl -= 1  # Cost of hop
            return packet

        return None
