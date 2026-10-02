# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — torchhd reference cross-check

"""Cross-check the binary spatter-code operators against ``torchhd``.

The pinned ``hdc-reference`` CI profile requires the reference to import.
Other environments skip only when TorchHD itself is absent; broken
installed references remain errors. The semantic contract page records
the deliberate differences from the external reference.
The adapter maps SC-NeuroCore's {0, 1} uint8 arrays onto torchhd BSC
tensors rather than weakening either side's semantics.
"""

from __future__ import annotations

import importlib
import importlib.util
import os
from typing import Any

import numpy as np
import pytest

from sc_neurocore.hdc import HDCEncoder

if (
    os.environ.get("SC_NEUROCORE_REQUIRE_TORCHHD") != "1"
    and importlib.util.find_spec("torchhd") is None
):
    pytest.skip("TorchHD reference extra is not installed", allow_module_level=True)

torchhd = importlib.import_module("torchhd")
torch = importlib.import_module("torch")

_SEEDS = (3, 41)
_DIMS = (256, 1024)


def _to_bsc(vector: np.ndarray[Any, Any]) -> Any:
    """Map the binary NumPy representation to the reference BSC tensor."""
    return torchhd.BSCTensor(torch.from_numpy(vector.astype(bool)))


def _from_bsc(tensor: Any) -> np.ndarray[Any, Any]:
    """Recover the canonical binary NumPy representation from BSC."""
    result: np.ndarray[Any, Any] = tensor.detach().cpu().numpy().astype(np.uint8)
    return result


@pytest.mark.parametrize("seed", _SEEDS)
@pytest.mark.parametrize("dim", _DIMS)
def test_bind_matches_torchhd_bsc(seed: int, dim: int) -> None:
    """Compare XOR binding with the pinned BSC operator on identical inputs."""
    enc = HDCEncoder(dim=dim, seed=seed)
    a = enc.generate_random_vector()
    b = enc.generate_random_vector()
    ours = enc.bind(a, b)
    reference = _from_bsc(_to_bsc(a).bind(_to_bsc(b)))
    assert np.array_equal(ours, reference)


@pytest.mark.parametrize("seed", _SEEDS)
@pytest.mark.parametrize("dim", _DIMS)
@pytest.mark.parametrize("count", (1, 3, 5))
def test_odd_bundle_matches_torchhd_bsc_majority(seed: int, dim: int, count: int) -> None:
    """Odd counts involve no tie policy, so both libraries must agree exactly."""
    enc = HDCEncoder(dim=dim, seed=seed)
    members = [enc.generate_random_vector() for _ in range(count)]
    ours = enc.bundle(members)
    stacked = torch.stack([_to_bsc(member) for member in members])
    reference = _from_bsc(torchhd.multibundle(stacked))
    assert np.array_equal(ours, reference)


@pytest.mark.parametrize("seed", _SEEDS)
@pytest.mark.parametrize("dim", _DIMS)
@pytest.mark.parametrize("shifts", (-3, 0, 3, 2051))
def test_permute_matches_torchhd(seed: int, dim: int, shifts: int) -> None:
    """Compare rotations, including inverse, identity and wrapped shifts."""
    enc = HDCEncoder(dim=dim, seed=seed)
    v = enc.generate_random_vector()
    ours = enc.permute(v, shifts)
    reference = _from_bsc(torchhd.permute(_to_bsc(v), shifts=shifts))
    assert np.array_equal(ours, reference)
