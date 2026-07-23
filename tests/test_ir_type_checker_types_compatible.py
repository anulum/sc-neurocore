# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestTypesCompatible from former test_ir_type_checker.py

"""Focused suite: TestTypesCompatible from former test_ir_type_checker.py."""

from __future__ import annotations

from tests.ir_type_checker_support import *  # noqa: F403

class TestTypesCompatible:
    """Coverage for direct and wildcard signal-type compatibility."""

    def test_same_type_compatible(self) -> None:
        assert types_compatible(SignalType.BITSTREAM, SignalType.BITSTREAM)
        assert types_compatible(SignalType.RATE, SignalType.RATE)
        assert types_compatible(SignalType.SPIKE, SignalType.SPIKE)

    def test_rate_to_bitstream_incompatible(self) -> None:
        assert not types_compatible(SignalType.RATE, SignalType.BITSTREAM)

    def test_bitstream_to_rate_incompatible(self) -> None:
        assert not types_compatible(SignalType.BITSTREAM, SignalType.RATE)

    def test_spike_to_bitstream_compatible(self) -> None:
        assert types_compatible(SignalType.SPIKE, SignalType.BITSTREAM)

    def test_any_matches_everything(self) -> None:
        assert types_compatible(SignalType.ANY, SignalType.BITSTREAM)
        assert types_compatible(SignalType.RATE, SignalType.ANY)
