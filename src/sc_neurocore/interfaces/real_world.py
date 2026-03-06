# SPDX-License-Identifier: AGPL-3.0-or-later
from typing import Any
import logging
import numpy as np

logger = logging.getLogger(__name__)


class LSLBridge:
    """
    Lab Streaming Layer (LSL) Bridge.
    Connects EEG/Physiological streams to sc-neurocore.
    (Mock implementation for standalone use).
    """

    def __init__(self, stream_name="NeuromorphicIn"):  # type: ignore
        self.stream_name = stream_name
        logger.info("LSL: Listening for stream '%s'...", stream_name)

    def receive_chunk(self, max_samples=32) -> np.ndarray[Any, Any]:  # type: ignore
        """
        Simulates receiving a chunk of samples.
        In real version: calls inlet.pull_chunk().
        """
        # Mock EEG data: 8 channels, random signals
        return np.random.normal(0, 50e-6, (8, max_samples))


class ROS2Node:
    """
    ROS 2 Interface Node.
    Publishes motor commands from sc-neurocore to robots.
    """

    def __init__(self, node_name="neuro_controller"):  # type: ignore
        self.node_name = node_name
        logger.info("ROS2: Node '%s' initialized.", node_name)

    def publish_cmd_vel(self, linear_x: float, angular_z: float):  # type: ignore
        """
        Simulates publishing to /cmd_vel.
        """
        msg = {"linear": linear_x, "angular": angular_z}
        # print(f"ROS2: Publishing to /cmd_vel: {json.dumps(msg)}")
        # In real version: self.publisher.publish(msg)
        return True
