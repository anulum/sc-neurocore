# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Physical-twin validation tests

"""Mode, timeout, noise, and divergence-threshold validation contracts."""

import pytest

from sc_neurocore.drivers.physical_twin import PhysicalTwinBridge


def test_rejects_unknown_mode():  # type: ignore[no-untyped-def] # Preserved legacy test AST
    with pytest.raises(ValueError, match="'EMULATION' or 'TCP'"):
        PhysicalTwinBridge(mode="BOGUS")


def test_rejects_non_positive_timeout():  # type: ignore[no-untyped-def] # Preserved legacy test AST
    with pytest.raises(ValueError, match="timeout_s must be positive"):
        PhysicalTwinBridge(timeout_s=0.0)


def test_rejects_negative_noise_sigma():  # type: ignore[no-untyped-def] # Preserved legacy test AST
    with pytest.raises(ValueError, match="noise_sigma must be non-negative"):
        PhysicalTwinBridge(noise_sigma=-0.1)


def test_rejects_non_positive_divergence_threshold():  # type: ignore[no-untyped-def] # Preserved legacy test AST
    with pytest.raises(ValueError, match="divergence_threshold must be positive"):
        PhysicalTwinBridge(divergence_threshold=0.0)
