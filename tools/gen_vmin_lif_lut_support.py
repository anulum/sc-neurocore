# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_gen_vmin_lif_lut.py

from __future__ import annotations

import math
import sys
from pathlib import Path
import pytest
from tools.gen_vmin_lif_lut import (
    LUT_RANGE,
    LUT_SIZE,
    Q88_MAX,
    Q88_MIN,
    Q88_SCALE,
    VminLifConfig,
    decode_q88,
    encode_q88,
    emit_lut_verilog_header,
    gen_softplus_lut,
    lut_lookup,
    main,
    softplus_float,
    vmin_lif_step_float,
    vmin_lif_step_q88,
)

if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))

__all__ = [
    "math",
    "sys",
    "Path",
    "pytest",
    "LUT_RANGE",
    "LUT_SIZE",
    "Q88_MAX",
    "Q88_MIN",
    "Q88_SCALE",
    "VminLifConfig",
    "decode_q88",
    "encode_q88",
    "emit_lut_verilog_header",
    "gen_softplus_lut",
    "lut_lookup",
    "main",
    "softplus_float",
    "vmin_lif_step_float",
    "vmin_lif_step_q88",
]
