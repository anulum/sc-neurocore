# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_extract_shd_weights.py

from __future__ import annotations

import os
import sys
import hashlib
from pathlib import Path
from typing import cast
import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from extract_shd_weights import (  # noqa: E402
    SHD_LAYERS,
    extract,
    quantise_per_tensor_symmetric,
    to_csr,
    write_int8_hex,
    write_delays_hex,
)
from sc_neurocore.security.checkpoint_loading import CheckpointTrustError  # noqa: E402

REPO = "/media/anulum/724AA8E84AA8AA75/aaa_God_of_the_Math_Collection/03_CODE/SC-NEUROCORE"
CKPT = f"{REPO}/data/masquelier_shd/cloud_results/dcls_max/dcls_max/last.pth"


def _sha256(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))

__all__ = [
    "os",
    "sys",
    "hashlib",
    "Path",
    "cast",
    "pytest",
    "torch",
    "SHD_LAYERS",
    "extract",
    "quantise_per_tensor_symmetric",
    "to_csr",
    "write_int8_hex",
    "write_delays_hex",
    "CheckpointTrustError",
    "REPO",
    "CKPT",
    "_sha256",
]
