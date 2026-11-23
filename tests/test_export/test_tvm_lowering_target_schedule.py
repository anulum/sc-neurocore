# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestTargetSchedule from former test_tvm_lowering.py

"""Focused suite: TestTargetSchedule from former test_tvm_lowering.py."""

from __future__ import annotations

import unittest
from sc_neurocore.export.tvm_lowering import (
    TargetDevice,
    TargetSchedule,
)


class TestTargetSchedule(unittest.TestCase):
    """Device schedules for TVM lowering."""

    def test_cpu_defaults(self) -> None:
        """The CPU schedule targets the CPU at optimisation level three."""
        s = TargetSchedule.for_cpu()
        self.assertEqual(s.device, TargetDevice.CPU)
        self.assertEqual(s.opt_level, 3)

    def test_gpu_schedule(self) -> None:
        """The GPU schedule targets CUDA with warp-level popcount."""
        s = TargetSchedule.for_gpu()
        self.assertEqual(s.device, TargetDevice.CUDA)
        self.assertIn("warp_level_popcount", s.sc_specific)

    def test_fpga_xilinx(self) -> None:
        """The Xilinx schedule targets its FPGA with bitstream packing."""
        s = TargetSchedule.for_fpga("xilinx")
        self.assertEqual(s.device, TargetDevice.FPGA_XILINX)
        self.assertTrue(s.sc_specific["bitstream_packing"])

    def test_fpga_intel(self) -> None:
        """The Intel schedule targets its FPGA."""
        s = TargetSchedule.for_fpga("intel")
        self.assertEqual(s.device, TargetDevice.FPGA_INTEL)
