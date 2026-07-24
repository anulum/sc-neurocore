# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestTentGate from former test_dcls_tent_kernel.py

"""Focused suite: TestTentGate from former test_dcls_tent_kernel.py."""

from __future__ import annotations

from tests.dcls_tent_kernel_support import *  # noqa: F403


class TestTentGate:
    """Triangular gate evaluation in Q8.8."""

    def test_peak_at_centre(self) -> None:
        # delay(tap 1) == centre 256 -> distance 0 -> full gate 256 (= 1.0).
        assert tent_gate_q88(1, 256, 512) == 256

    def test_linear_falloff(self) -> None:
        # delay(tap 0)=0, centre 256, sigma 512 -> (512-256)*256//512 = 128.
        assert tent_gate_q88(0, 256, 512) == 128

    def test_zero_outside_support(self) -> None:
        # delay(tap 3)=768, distance 512 >= sigma 512 -> gate clipped to 0.
        assert tent_gate_q88(3, 256, 512) == 0

    def test_non_positive_sigma_rejected(self) -> None:
        with pytest.raises(ValueError, match="sigma must be positive"):
            tent_gate_q88(0, 0, 0)

    def test_negative_tap_rejected(self) -> None:
        with pytest.raises(ValueError, match="tap index must be non-negative"):
            tent_gate_q88(-1, 0, 256)

    def test_gate_never_exceeds_unity(self) -> None:
        # Sweep a dense tent; the peak gate equals exactly Q88_ONE and nothing
        # exceeds it.
        gates = [tent_gate_q88(tap, 512, 600) for tap in range(8)]
        assert max(gates) == kernel.Q88_ONE
        assert min(gates) >= 0
