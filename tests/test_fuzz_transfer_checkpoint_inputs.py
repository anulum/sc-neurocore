# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Property-based fuzz tests for transfer checkpoint inputs

"""Property-based fuzz tests for transfer checkpoint metadata and archives."""

from __future__ import annotations

import json
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from sc_neurocore.transfer import SNNCheckpoint, load_checkpoint, save_checkpoint

_JSON_SCALAR = (
    st.none()
    | st.booleans()
    | st.integers()
    | st.floats(allow_nan=False, allow_infinity=False)
    | st.text()
)
_JSON_VALUE = st.recursive(
    _JSON_SCALAR,
    lambda children: (
        st.lists(children, max_size=4) | st.dictionaries(st.text(max_size=12), children, max_size=4)
    ),
    max_leaves=24,
)


def _write_checkpoint_pair(
    base_path: Path,
    meta: object,
    arrays: Mapping[str, NDArray[Any]],
) -> None:
    (base_path.with_suffix(".json")).write_text(json.dumps(meta), encoding="utf-8")
    np.savez_compressed(base_path.with_suffix(".npz"), **cast(Any, arrays))


@given(meta=_JSON_VALUE)
@settings(max_examples=120, deadline=None)
def test_fuzz_load_checkpoint_rejects_malformed_metadata(meta: object) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        base = Path(tmpdir) / "model"
        _write_checkpoint_pair(base, meta, {"layer_0": np.asarray([1.0], dtype=np.float32)})

        try:
            loaded = load_checkpoint(base)
        except ValueError:
            return

        assert isinstance(loaded, SNNCheckpoint)
        assert loaded.n_layers == len(loaded.weights)


@given(extra_key=st.text(min_size=1, max_size=12).filter(lambda key: key != "layer_0"))
@settings(max_examples=80, deadline=None)
def test_fuzz_load_checkpoint_rejects_unexpected_npz_members(extra_key: str) -> None:
    meta = {
        "layer_names": ["hidden"],
        "layer_sizes": [[1, 1]],
        "neuron_types": ["LIF"],
        "frozen_layers": [],
        "n_layers": 1,
        "total_params": 1,
        "metadata": {},
    }
    with tempfile.TemporaryDirectory() as tmpdir:
        base = Path(tmpdir) / "model"
        _write_checkpoint_pair(
            base,
            meta,
            {
                "layer_0": np.asarray([1.0], dtype=np.float32),
                extra_key: np.asarray([2.0], dtype=np.float32),
            },
        )

        with pytest.raises(ValueError, match="weight archive"):
            load_checkpoint(base)


def test_load_checkpoint_uses_non_pickle_npz_boundary(tmp_path: Path) -> None:
    base = tmp_path / "model"
    meta = {
        "layer_names": ["hidden"],
        "layer_sizes": [[1, 1]],
        "neuron_types": ["LIF"],
        "frozen_layers": [],
        "n_layers": 1,
        "total_params": 1,
        "metadata": {},
    }
    _write_checkpoint_pair(
        base, meta, {"layer_0": np.asarray([{"unsafe": "object"}], dtype=object)}
    )

    with pytest.raises(ValueError):
        load_checkpoint(base)


def test_load_checkpoint_roundtrip_after_schema_validation(tmp_path: Path) -> None:
    checkpoint = SNNCheckpoint(
        weights=[np.asarray([[1.0, -1.0]], dtype=np.float32)],
        layer_names=["hidden"],
        layer_sizes=[(2, 1)],
        neuron_types=["LIF"],
        metadata={"task": "smoke"},
    )
    base = tmp_path / "model"
    save_checkpoint(checkpoint, base)

    loaded = load_checkpoint(base)

    assert loaded.layer_names == ["hidden"]
    assert loaded.layer_sizes == [(2, 1)]
    assert loaded.metadata == {"task": "smoke"}
    np.testing.assert_array_equal(loaded.weights[0], checkpoint.weights[0])
