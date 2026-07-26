# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Physical-twin emulation tests

"""Emulated output type, noise band, and disconnected fallback contracts."""

import numpy as np

from sc_neurocore.drivers.physical_twin import PhysicalTwinBridge


def test_sync_step_returns_float():  # type: ignore[no-untyped-def] # Preserved legacy test AST
    bridge = PhysicalTwinBridge()
    hw_v = bridge.sync_step(sw_v_mem=-65.0, sw_spike=0)
    assert isinstance(hw_v, (float, np.floating))


def test_sync_step_close_to_input():  # type: ignore[no-untyped-def] # Preserved legacy test AST
    bridge = PhysicalTwinBridge()
    sw_v = -65.0
    results = [bridge.sync_step(sw_v, 0) for _ in range(100)]
    mean_diff = np.mean(np.abs(np.array(results) - sw_v))
    # Mock adds N(0, 0.01) noise, so mean abs diff << 0.1
    assert mean_diff < 0.05


def test_sync_step_disconnected_returns_input():  # type: ignore[no-untyped-def] # Preserved legacy test AST
    bridge = PhysicalTwinBridge()
    bridge.connected = False
    sw_v = -70.0
    assert bridge.sync_step(sw_v, 1) == sw_v
