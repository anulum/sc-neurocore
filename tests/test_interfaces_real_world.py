# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for the mock LSL and ROS2 real-world interface bridges

"""Contracts for the mock Lab-Streaming-Layer and ROS2 real-world bridges."""

from __future__ import annotations

from sc_neurocore.interfaces.real_world import LSLBridge, ROS2Node


def test_lsl_bridge_receive_chunk_returns_eight_channel_window() -> None:
    """LSLBridge.receive_chunk returns an 8-channel chunk of the requested width."""
    bridge = LSLBridge(stream_name="eeg")

    chunk = bridge.receive_chunk(max_samples=16)

    assert chunk.shape == (8, 16)


def test_ros2_node_publish_cmd_vel_reports_success() -> None:
    """ROS2Node.publish_cmd_vel builds a velocity message and reports success."""
    node = ROS2Node(node_name="bot")

    assert node.publish_cmd_vel(linear_x=0.5, angular_z=-0.2) is True
