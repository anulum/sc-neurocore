# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Pre-trained weight loading for model zoo architectures

"""Load pre-initialised weights into model zoo networks.

Weights use Xavier/Glorot with spiking correction factor 0.5
(Zenke & Ganguli 2018). Stored as ``.npz`` in the ``weights/``
subdirectory.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse

from sc_neurocore.network import Network

from .configs import dvs_gesture_classifier, mnist_classifier, shd_speech_classifier

_WEIGHTS_DIR = Path(__file__).parent / "weights"

_PretrainedBuilder = Callable[[], Network]
_ProjectionSpec = tuple[str, tuple[int, int]]
_WeightSpec = tuple[_ProjectionSpec, ...]

_REGISTRY: dict[str, tuple[_PretrainedBuilder, str]] = {
    "mnist": (mnist_classifier, "mnist_784_128_10.npz"),
    "shd": (shd_speech_classifier, "shd_700_256_20.npz"),
    "dvs_gesture": (dvs_gesture_classifier, "dvs_256_256_11.npz"),
}

_WEIGHT_SPECS: dict[str, _WeightSpec] = {
    "mnist": (("W0", (784, 128)), ("W1", (128, 10))),
    "shd": (("W0", (700, 256)), ("W_rec", (256, 256)), ("W1", (256, 20))),
    "dvs_gesture": (("W0", (256, 256)), ("W1", (256, 11))),
}


def _dense_to_csr(
    dense: np.ndarray[Any, Any],
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """Convert dense weight matrix (fan_in, fan_out) to CSR arrays."""
    sp = sparse.csr_matrix(dense)
    return (
        sp.indptr.astype(np.int32),
        sp.indices.astype(np.int32),
        sp.data.astype(np.float64),
    )


def _apply_weights(proj: object, dense: np.ndarray[Any, Any]) -> None:
    """Overwrite a Projection's CSR data with a dense weight matrix."""
    indptr, indices, data = _dense_to_csr(dense)
    proj.indptr = indptr  # type: ignore[attr-defined]
    proj.indices = indices  # type: ignore[attr-defined]
    proj.data = data  # type: ignore[attr-defined]


def _weight_spec_for(name: str) -> _WeightSpec:
    """Return the archive schema for a registered pretrained model."""
    try:
        return _WEIGHT_SPECS[name]
    except KeyError as exc:
        raise ValueError(
            f"Invalid pretrained registry entry for '{name}': missing weight schema"
        ) from exc


def _validate_projection_layout(name: str, net: Network, specs: _WeightSpec) -> None:
    """Validate the built network before applying pretrained archive weights."""
    expected_count = len(specs)
    actual_count = len(net.projections)
    if actual_count != expected_count:
        raise ValueError(
            f"Invalid pretrained model wiring for '{name}': projection count "
            f"{actual_count} does not match archive schema count {expected_count}"
        )

    for index, (projection, (key, expected_shape)) in enumerate(
        zip(net.projections, specs, strict=True)
    ):
        actual_shape = (projection.source.n, projection.target.n)
        if actual_shape != expected_shape:
            raise ValueError(
                f"Invalid pretrained model wiring for '{name}': projection {index} "
                f"for archive member '{key}' has topology "
                f"{actual_shape[0]}->{actual_shape[1]}, expected "
                f"{expected_shape[0]}->{expected_shape[1]}"
            )


def _validate_archive_members(name: str, archive: np.lib.npyio.NpzFile, specs: _WeightSpec) -> None:
    """Validate archive keys against the registered pretrained schema."""
    expected = {key for key, _shape in specs}
    actual = set(archive.files)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing or extra:
        details = []
        if missing:
            details.append(f"missing={missing}")
        if extra:
            details.append(f"unexpected={extra}")
        raise ValueError(f"Invalid pretrained weight archive for '{name}': {', '.join(details)}")


def _load_weight_matrix(
    name: str,
    archive: np.lib.npyio.NpzFile,
    key: str,
    expected_shape: tuple[int, int],
) -> np.ndarray[Any, Any]:
    try:
        matrix = archive[key]
    except ValueError as exc:
        raise ValueError(
            f"Invalid pretrained weight archive for '{name}': '{key}' cannot be loaded"
        ) from exc

    if matrix.ndim != 2:
        raise ValueError(
            f"Invalid pretrained weight archive for '{name}': "
            f"'{key}' must be a 2-D matrix, got {matrix.ndim}-D"
        )
    if matrix.shape != expected_shape:
        raise ValueError(
            f"Invalid pretrained weight archive for '{name}': "
            f"'{key}' has shape {matrix.shape}, expected {expected_shape}"
        )
    if not np.issubdtype(matrix.dtype, np.number) or np.issubdtype(
        matrix.dtype, np.complexfloating
    ):
        raise ValueError(
            f"Invalid pretrained weight archive for '{name}': "
            f"'{key}' must contain real numeric weights"
        )

    weights = matrix.astype(np.float64, copy=False)
    if not np.all(np.isfinite(weights)):
        raise ValueError(
            f"Invalid pretrained weight archive for '{name}': '{key}' contains non-finite weights"
        )
    return weights


def load_pretrained(name: str) -> Network:
    """Build a model-zoo network and load its shipped pretrained weights.

    Parameters
    ----------
    name:
        Registered pretrained model name. Supported values are ``"mnist"``,
        ``"shd"``, and ``"dvs_gesture"``.

    Returns
    -------
    Network
        A freshly built network whose projection CSR arrays were replaced by
        validated arrays from the matching ``.npz`` archive.

    Raises
    ------
    ValueError
        If the name is unknown, the registry/schema wiring is inconsistent, the
        built network projection layout does not match the archive schema, or
        any archive member is malformed.
    FileNotFoundError
        If the expected pretrained archive is missing.
    """
    if name not in _REGISTRY:
        raise ValueError(f"Unknown pretrained model '{name}'. Available: {sorted(_REGISTRY)}")

    builder, weight_file = _REGISTRY[name]
    specs = _weight_spec_for(name)
    path = Path(_WEIGHTS_DIR) / weight_file
    if not path.exists():
        raise FileNotFoundError(
            f"Weights for pretrained model '{name}' not found: expected "
            f"{weight_file!r} under {Path(_WEIGHTS_DIR)}"
        )

    net = builder()
    _validate_projection_layout(name, net, specs)

    with np.load(path, allow_pickle=False) as data:
        _validate_archive_members(name, data, specs)
        for projection, (key, expected_shape) in zip(net.projections, specs, strict=True):
            weights = _load_weight_matrix(name, data, key, expected_shape)
            _apply_weights(projection, weights)

    return net
