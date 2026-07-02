# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Property-based fuzz tests for model-zoo weight archives

"""Property-based fuzz tests for pretrained model-zoo NPZ inputs."""

from __future__ import annotations

import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from sc_neurocore.model_zoo import pretrained as _pt
from sc_neurocore.network import Network

_SAFE_EXTRA_KEY = st.from_regex(r"[A-Za-z][A-Za-z0-9_]{0,10}", fullmatch=True).filter(
    lambda key: key not in {"W0", "W1", "W_rec"}
)


def _write_archive(directory: Path, name: str, arrays: Mapping[str, np.ndarray[Any, Any]]) -> None:
    _builder, filename = _pt._REGISTRY[name]
    np.savez(directory / filename, **cast(Any, arrays))


def _load_with_weight_dir(name: str, directory: Path) -> Network:
    old_dir = _pt._WEIGHTS_DIR
    try:
        _pt._WEIGHTS_DIR = directory
        return _pt.load_pretrained(name)
    finally:
        _pt._WEIGHTS_DIR = old_dir


@given(name=st.sampled_from(["mnist", "shd", "dvs_gesture"]), extra_key=_SAFE_EXTRA_KEY)
@settings(max_examples=80, deadline=None)
def test_fuzz_load_pretrained_rejects_unexpected_npz_members(name: str, extra_key: str) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        weight_dir = Path(tmpdir)
        arrays = {
            key: np.asarray([[0.0]], dtype=np.float32) for key, _shape in _pt._WEIGHT_SPECS[name]
        }
        arrays[extra_key] = np.asarray([[1.0]], dtype=np.float32)
        _write_archive(weight_dir, name, arrays)

        with pytest.raises(ValueError, match="pretrained weight archive"):
            _load_with_weight_dir(name, weight_dir)


@given(name=st.sampled_from(["mnist", "shd", "dvs_gesture"]))
@settings(max_examples=20, deadline=None)
def test_fuzz_load_pretrained_rejects_missing_npz_members(name: str) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        weight_dir = Path(tmpdir)
        first_key = _pt._WEIGHT_SPECS[name][0][0]
        _write_archive(weight_dir, name, {first_key: np.asarray([[0.0]], dtype=np.float32)})

        with pytest.raises(ValueError, match="pretrained weight archive"):
            _load_with_weight_dir(name, weight_dir)


@given(nonfinite=st.sampled_from([np.nan, np.inf, -np.inf]))
@settings(max_examples=3, deadline=None)
def test_fuzz_load_pretrained_rejects_nonfinite_weights(nonfinite: float) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        weight_dir = Path(tmpdir)
        w0 = np.zeros((784, 128), dtype=np.float32)
        w1 = np.zeros((128, 10), dtype=np.float32)
        w0[0, 0] = nonfinite
        _write_archive(weight_dir, "mnist", {"W0": w0, "W1": w1})

        with pytest.raises(ValueError, match="non-finite"):
            _load_with_weight_dir("mnist", weight_dir)


@pytest.mark.parametrize(
    ("arrays", "message"),
    [
        ({"W0": np.asarray([1.0], dtype=np.float32), "W1": np.zeros((128, 10))}, "2-D"),
        ({"W0": np.zeros((1, 1), dtype=np.float32), "W1": np.zeros((128, 10))}, "shape"),
        ({"W0": np.zeros((784, 128), dtype=np.complex64), "W1": np.zeros((128, 10))}, "real"),
    ],
)
def test_load_pretrained_rejects_bad_weight_matrix_schema(
    arrays: Mapping[str, np.ndarray[Any, Any]], message: str
) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        weight_dir = Path(tmpdir)
        _write_archive(weight_dir, "mnist", arrays)

        with pytest.raises(ValueError, match=message):
            _load_with_weight_dir("mnist", weight_dir)


def test_load_pretrained_uses_non_pickle_npz_boundary() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        weight_dir = Path(tmpdir)
        w0 = np.empty((784, 128), dtype=object)
        w0.fill({"unsafe": "object"})
        _write_archive(weight_dir, "mnist", {"W0": w0, "W1": np.zeros((128, 10))})

        with pytest.raises(ValueError, match="cannot be loaded"):
            _load_with_weight_dir("mnist", weight_dir)
