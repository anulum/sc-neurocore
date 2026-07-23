# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestProvenanceChain from former test_intelligence_reporting.py

"""Focused suite: TestProvenanceChain from former test_intelligence_reporting.py."""

from __future__ import annotations

from tests.intelligence_reporting_support import *  # noqa: F403

class TestProvenanceChain:
    """Cryptographic audit trail."""

    def test_chain_length(self):
        from sc_neurocore.compiler.intelligence import (
            generate_provenance_chain,
        )

        chain = generate_provenance_chain(
            "sc_lif",
            {"v": "a + b"},
        )
        assert len(chain) == 3
        assert chain[0].stage == "source_equations"
        assert chain[1].stage == "compilation_config"
        assert chain[2].stage == "verilog_generation"

    def test_hash_chain_linked(self):
        from sc_neurocore.compiler.intelligence import (
            generate_provenance_chain,
        )

        chain = generate_provenance_chain(
            "sc_lif",
            {"v": "a + b"},
        )
        assert chain[0].output_hash == chain[1].input_hash
        assert chain[1].output_hash == chain[2].input_hash

    def test_genesis(self):
        from sc_neurocore.compiler.intelligence import (
            generate_provenance_chain,
        )

        chain = generate_provenance_chain("sc_lif", {"v": "a"})
        assert chain[0].input_hash == "genesis"

    def test_json_format(self):
        from sc_neurocore.compiler.intelligence import (
            generate_provenance_chain,
            format_provenance_json,
        )

        chain = generate_provenance_chain("sc_lif", {"v": "a"})
        j = format_provenance_json(chain)
        data = json.loads(j)
        assert "sc_neurocore_provenance" in data
        assert len(data["sc_neurocore_provenance"]["chain"]) == 3

    def test_deterministic_hashes(self):
        from sc_neurocore.compiler.intelligence import (
            generate_provenance_chain,
        )

        c1 = generate_provenance_chain("sc_lif", {"v": "a + b"})
        c2 = generate_provenance_chain("sc_lif", {"v": "a + b"})
        assert c1[0].output_hash == c2[0].output_hash
