# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestModelChecksum from former test_intelligence_security_and_compliance.py

"""Focused suite: TestModelChecksum from former test_intelligence_security_and_compliance.py."""

from __future__ import annotations


class TestModelChecksum:
    """SHA-256 model checksum embedding."""

    def test_checksum_embedded(self) -> None:
        """The model hash is embedded in the module."""
        from sc_neurocore.compiler.intelligence import embed_model_checksum

        verilog = "// Test module\nmodule sc_lif (...);\nendmodule"
        result = embed_model_checksum(
            verilog,
            equations={"v": "a + b"},
            params={"data_width": 16},
        )
        assert "SHA-256:" in result
        assert "MODEL_HASH" in result
        assert "256'h" in result

    def test_checksum_deterministic(self) -> None:
        """The same inputs give the same hash."""
        from sc_neurocore.compiler.intelligence import embed_model_checksum

        v1 = embed_model_checksum(
            "module x; endmodule",
            equations={"v": "a * b"},
        )
        v2 = embed_model_checksum(
            "module x; endmodule",
            equations={"v": "a * b"},
        )
        # Same inputs → same hash
        import re

        m_h1 = re.search(r"SHA-256: ([0-9a-f]+)", v1)
        assert m_h1 is not None
        h1 = m_h1.group(1)
        m_h2 = re.search(r"SHA-256: ([0-9a-f]+)", v2)
        assert m_h2 is not None
        h2 = m_h2.group(1)
        assert h1 == h2

    def test_different_equations_different_hash(self) -> None:
        """Different equations give different hashes."""
        from sc_neurocore.compiler.intelligence import embed_model_checksum
        import re

        v1 = embed_model_checksum("module x; endmodule", equations={"v": "a+b"})
        v2 = embed_model_checksum("module x; endmodule", equations={"v": "a*b"})
        m_h1 = re.search(r"SHA-256: ([0-9a-f]+)", v1)
        assert m_h1 is not None
        h1 = m_h1.group(1)
        m_h2 = re.search(r"SHA-256: ([0-9a-f]+)", v2)
        assert m_h2 is not None
        h2 = m_h2.group(1)
        assert h1 != h2

    def test_no_equations_still_works(self) -> None:
        """A module without equations still gets a hash."""
        from sc_neurocore.compiler.intelligence import embed_model_checksum

        result = embed_model_checksum("module y; endmodule")
        assert "MODEL_HASH" in result
