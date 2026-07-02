# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for GPU backend contracts

"""Contracts for GPU backend CPU fallback packing and vector operations."""

from __future__ import annotations

import importlib.machinery
import importlib.util
from pathlib import Path
import sys
from types import ModuleType
from typing import Any, cast

import numpy as np
import pytest


def _load_gpu_backend_without_cupy(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    source_path = (
        Path(__file__).resolve().parents[1] / "src" / "sc_neurocore" / "accel" / "gpu_backend.py"
    )
    monkeypatch.setitem(sys.modules, "cupy", None)
    spec = importlib.util.spec_from_file_location("gpu_backend_no_cupy_contract", source_path)
    assert spec is not None
    loader = spec.loader
    assert isinstance(loader, importlib.machinery.SourceFileLoader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


def test_gpu_backend_import_without_cupy_selects_numpy_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_gpu_backend_without_cupy(monkeypatch)
    backend_xp = cast(Any, module.__dict__["xp"])
    has_cupy = cast(bool, module.__dict__["HAS_CUPY"])

    assert has_cupy is False
    np.testing.assert_array_equal(backend_xp.arange(3), np.arange(3))


def test_gpu_pack_bitstream_pads_1d_and_2d_inputs() -> None:
    from sc_neurocore.accel.gpu_backend import gpu_pack_bitstream

    packed_1d = gpu_pack_bitstream(np.array([1, 0, 1], dtype=np.uint8))
    packed_2d = gpu_pack_bitstream(np.array([[1, 0, 1], [0, 1, 0]], dtype=np.uint8))

    assert packed_1d.shape == (1,)
    assert packed_2d.shape == (2, 1)


def test_gpu_pack_bitstream_rejects_3d_input() -> None:
    from sc_neurocore.accel.gpu_backend import gpu_pack_bitstream

    with pytest.raises(ValueError, match="Expected 1-D or 2-D"):
        gpu_pack_bitstream(np.zeros((2, 3, 4), dtype=np.uint8))


def test_gpu_backend_cpu_transfer_and_vector_ops_preserve_contracts() -> None:
    from sc_neurocore.accel.gpu_backend import gpu_popcount
    from sc_neurocore.accel.gpu_backend import gpu_vec_and
    from sc_neurocore.accel.gpu_backend import to_device
    from sc_neurocore.accel.gpu_backend import to_host

    values = np.array([0xF0F0, 0x00FF], dtype=np.uint64)
    device_values = to_device(values)

    np.testing.assert_array_equal(to_host(device_values), values)
    np.testing.assert_array_equal(gpu_vec_and(values, values), values)
    np.testing.assert_array_equal(gpu_popcount(values), np.array([8, 8], dtype=np.uint32))
