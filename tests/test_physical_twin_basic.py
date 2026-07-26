# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Physical-twin construction tests

"""Import, default, custom, and public-attribute construction contracts."""

from sc_neurocore.drivers.physical_twin import PhysicalTwinBridge


def test_import():  # type: ignore[no-untyped-def] # Preserved legacy test AST
    assert PhysicalTwinBridge is not None


def test_instantiation_default():  # type: ignore[no-untyped-def] # Preserved legacy test AST
    bridge = PhysicalTwinBridge()
    assert bridge.connected is True
    assert bridge.ip == "192.168.2.99"
    assert bridge.port == 5000


def test_instantiation_custom():  # type: ignore[no-untyped-def] # Preserved legacy test AST
    bridge = PhysicalTwinBridge(ip="10.0.0.1", port=9999)
    assert bridge.ip == "10.0.0.1"
    assert bridge.port == 9999


def test_has_expected_attributes():  # type: ignore[no-untyped-def] # Preserved legacy test AST
    bridge = PhysicalTwinBridge()
    assert hasattr(bridge, "sync_step")
    assert hasattr(bridge, "connected")
    assert hasattr(bridge, "ip")
    assert hasattr(bridge, "port")
