# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestTypesCompatible from former test_ir_type_checker.py

"""Focused suite: TestTypesCompatible from former test_ir_type_checker.py."""

from __future__ import annotations

from sc_neurocore.compiler.ir_type_checker import (
    SignalType,
    types_compatible,
)


class TestTypesCompatible:
    """Coverage for direct and wildcard signal-type compatibility."""

    def test_same_type_compatible(self) -> None:
        """A type is compatible with itself."""
        assert types_compatible(SignalType.BITSTREAM, SignalType.BITSTREAM)
        assert types_compatible(SignalType.RATE, SignalType.RATE)
        assert types_compatible(SignalType.SPIKE, SignalType.SPIKE)

    def test_rate_to_bitstream_incompatible(self) -> None:
        """A rate is not a bitstream."""
        assert not types_compatible(SignalType.RATE, SignalType.BITSTREAM)

    def test_bitstream_to_rate_incompatible(self) -> None:
        """A bitstream is not a rate."""
        assert not types_compatible(SignalType.BITSTREAM, SignalType.RATE)

    def test_spike_to_bitstream_compatible(self) -> None:
        """A spike train is a bitstream."""
        assert types_compatible(SignalType.SPIKE, SignalType.BITSTREAM)

    def test_any_matches_everything(self) -> None:
        """The any type matches every type."""
        assert types_compatible(SignalType.ANY, SignalType.BITSTREAM)
        assert types_compatible(SignalType.RATE, SignalType.ANY)
