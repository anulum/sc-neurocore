# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestPackageImport from former test_quantum_cognition.py

"""Focused suite: TestPackageImport from former test_quantum_cognition.py."""

from __future__ import annotations

from tests.quantum_cognition_support import *  # noqa: F403

class TestPackageImport:
    """Verify the package-level imports work correctly."""

    def test_import_all(self) -> None:
        from sc_neurocore.quantum_cognition import (
            SpinPoolMPS,
            HybridFisherPosnerLIF,
            FisherPosnerQuantumBridge,
            QuantumStudioHook,
        )

        assert SpinPoolMPS is not None
        assert HybridFisherPosnerLIF is not None
        assert FisherPosnerQuantumBridge is not None
        assert QuantumStudioHook is not None

    def test_tier_label(self) -> None:
        from sc_neurocore import quantum_cognition

        assert quantum_cognition.__tier__ == "experimental"
