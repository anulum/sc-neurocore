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

from pathlib import Path

import numpy as np
from scipy import sparse

from sc_neurocore.network import Network

from .configs import dvs_gesture_classifier, mnist_classifier, shd_speech_classifier

_WEIGHTS_DIR = Path(__file__).parent / "weights"

_REGISTRY: dict[str, tuple[object, str]] = {
    "mnist": (mnist_classifier, "mnist_784_128_10.npz"),
    "shd": (shd_speech_classifier, "shd_700_256_20.npz"),
    "dvs_gesture": (dvs_gesture_classifier, "dvs_256_256_11.npz"),
}


def _dense_to_csr(dense: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert dense weight matrix (fan_in, fan_out) to CSR arrays."""
    sp = sparse.csr_matrix(dense)
    return (
        sp.indptr.astype(np.int32),
        sp.indices.astype(np.int32),
        sp.data.astype(np.float64),
    )


def _apply_weights(proj: object, dense: np.ndarray) -> None:
    """Overwrite a Projection's CSR data with a dense weight matrix."""
    indptr, indices, data = _dense_to_csr(dense)
    proj.indptr = indptr  # type: ignore[attr-defined]
    proj.indices = indices  # type: ignore[attr-defined]
    proj.data = data  # type: ignore[attr-defined]


def load_pretrained(name: str) -> Network:
    """Load a network with pre-initialised weights.

    Supported names: ``'mnist'``, ``'shd'``, ``'dvs_gesture'``.

    Raises ``ValueError`` for unknown names.
    """
    if name not in _REGISTRY:
        raise ValueError(f"Unknown pretrained model '{name}'. Available: {sorted(_REGISTRY)}")

    builder, weight_file = _REGISTRY[name]
    path = _WEIGHTS_DIR / weight_file
    if not path.exists():
        raise FileNotFoundError(f"Weight file not found: {path}")

    net = builder()  # type: ignore[operator]
    data = np.load(path)

    projections = net.projections
    if name == "mnist":
        _apply_weights(projections[0], data["W0"])
        _apply_weights(projections[1], data["W1"])
    elif name == "shd":
        _apply_weights(projections[0], data["W0"])
        _apply_weights(projections[1], data["W_rec"])
        _apply_weights(projections[2], data["W1"])
    elif name == "dvs_gesture":
        _apply_weights(projections[0], data["W0"])
        _apply_weights(projections[1], data["W1"])

    return net
