# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestDNAMutation from former test_jax_adapter_fallback_contracts.py

"""Focused suite: TestDNAMutation from former test_jax_adapter_fallback_contracts.py."""

from __future__ import annotations

from tests.jax_adapter_fallback_contracts_support import *  # noqa: F403

class TestDNAMutation:
    def test_decode_triggers_mutation(self):
        from sc_neurocore.adapters.holonomic.dna_storage import DNAEncoder

        enc = DNAEncoder(mutation_rate=1.0)
        result = enc.decode("ACGT")
        assert len(result) == 8
