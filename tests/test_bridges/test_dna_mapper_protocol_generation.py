# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestProtocolGeneration from former test_dna_mapper.py

"""Focused suite: TestProtocolGeneration from former test_dna_mapper.py."""

from __future__ import annotations

from dna_mapper_support import *  # noqa: F403


class TestProtocolGeneration:
    """Wet-lab protocol generation."""

    def test_protocol_is_markdown(self, simple_and_circuit: DNACircuitDesign) -> None:
        protocol = generate_protocol(simple_and_circuit)
        assert protocol.startswith("# Wet-Lab Protocol")
        assert "## Materials" in protocol
        assert "## Procedure" in protocol

    def test_protocol_contains_strands(self, simple_and_circuit: DNACircuitDesign) -> None:
        protocol = generate_protocol(simple_and_circuit)
        assert "translator" in protocol or "signal" in protocol

    def test_protocol_custom_volume(self, simple_and_circuit: DNACircuitDesign) -> None:
        protocol = generate_protocol(simple_and_circuit, volume_uL=100.0)
        assert "100.0" in protocol
