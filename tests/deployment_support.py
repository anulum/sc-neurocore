# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_deployment.py

from __future__ import annotations

"""Tests for resource estimation, constraint gen, driver gen, Cocotb gen."""
from collections.abc import Callable
from pathlib import Path
import shutil
import subprocess
from typing import Protocol, cast
import pytest
from sc_neurocore.compiler.deployment import (
    estimate_resources,
    generate_cocotb_testbench,
    generate_constraints,
    generate_host_driver,
)
from sc_neurocore.compiler.live_control import MMIOUpdateSpec, ParameterBankSpec

STUB_VERILOG = """
module sc_lif (input wire clk, input wire rst, input wire en,
               input wire signed [15:0] I_t, output wire spike_out);
    reg signed [15:0] v_reg;
    reg signed [15:0] v_rest;
    wire signed [31:0] _mul0 = v_reg * I_t;
    wire signed [31:0] _mul1 = v_rest * v_reg;
    wire signed [15:0] _t0 = (_mul0 >>> 8);
    wire signed [16:0] v_raw = v_reg + _t0 - v_rest;
    wire signed [15:0] v_next =
        (v_raw > 17'sd32767) ? 16'sd32767 :
        (v_raw < (-17'sd32768)) ? (-16'sd32768) :
        v_raw[15:0];
    assign spike_out = (v_next > 16'sd7680);
endmodule
"""
LIF_PARAMS = {"P_V_REST": 16, "P_V_THRESH": 16, "P_TAU_M": 16}


class _GeneratedUnsafeHostDriver(Protocol):
    """Protocol for the dynamically generated unsafe-name test driver."""

    def set_tau_m(self, value: float) -> None:
        """Set the generated tau parameter register."""

    def set_v_thresh(self, value: float) -> None:
        """Set the generated voltage-threshold parameter register."""


class _GeneratedUnsafeHostDriverFactory(Protocol):
    """Constructor protocol for the dynamically executed test driver class."""

    def __call__(
        self,
        read_fn: Callable[[int], int],
        write_fn: Callable[[int, int], None],
    ) -> _GeneratedUnsafeHostDriver:
        """Create a generated driver instance."""


class _GeneratedLiveHostDriver(Protocol):
    """Protocol for generated live-control host driver methods under test."""

    def verify_live_weights_w0_encoded(self, encoded_word: int) -> bool:
        """Update and verify the generated live-control weight slot."""

    def update_live_weights_w0_encoded(self, encoded_word: int) -> None:
        """Update the generated live-control weight slot."""

    def read_live_status(self) -> int:
        """Read the generated live-control status register."""

    def read_live_trap_status(self) -> int:
        """Read the generated live-control trap status register."""

    def rollback_live_shadow(self) -> None:
        """Rollback the generated live-control shadow bank."""

    def clear_selected_live_traps(self, trap_mask: int | bool) -> None:
        """Clear selected generated live-control traps."""

    def clear_live_traps(self) -> None:
        """Clear all generated live-control traps."""


class _GeneratedLiveHostDriverFactory(Protocol):
    """Constructor protocol for the dynamically executed live-control class."""

    def __call__(
        self,
        read_fn: Callable[[int], int],
        write_fn: Callable[[int, int], None],
    ) -> _GeneratedLiveHostDriver:
        """Create a generated live-control driver instance."""


__all__ = [
    "Callable",
    "Path",
    "shutil",
    "subprocess",
    "Protocol",
    "cast",
    "pytest",
    "estimate_resources",
    "generate_cocotb_testbench",
    "generate_constraints",
    "generate_host_driver",
    "MMIOUpdateSpec",
    "ParameterBankSpec",
    "STUB_VERILOG",
    "LIF_PARAMS",
    "_GeneratedUnsafeHostDriver",
    "_GeneratedUnsafeHostDriverFactory",
    "_GeneratedLiveHostDriver",
    "_GeneratedLiveHostDriverFactory",
]
