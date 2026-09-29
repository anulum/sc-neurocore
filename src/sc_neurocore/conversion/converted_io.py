# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Portable file form of a converted network

"""Write a converted network to one ``.npz`` file and read it back unchanged.

The file holds every coefficient as float64 and a JSON header with the replay
semantics and the network's digest. Reading never unpickles, rebuilds the
network through its own validation and refuses a file whose rebuilt network
does not have the recorded digest.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .converted_snn import ConvertedSNN
from .loss_report import converted_sha256

CONVERTED_NETWORK_SCHEMA_VERSION = "sc-neurocore.converted-network.v1"


def save_converted_network(snn: ConvertedSNN, path: str | Path) -> str:
    """Write ``snn`` to ``path`` and return its digest.

    Parameters
    ----------
    snn : ConvertedSNN
        The network to write.
    path : str or Path
        Destination ``.npz`` file; an existing file is replaced.

    Returns
    -------
    str
        The network's ``converted_sha256``, also stored in the file.
    """
    digest = converted_sha256(snn)
    header = {
        "schema_version": CONVERTED_NETWORK_SCHEMA_VERSION,
        "converted_sha256": digest,
        "layers": snn.n_layers,
        "biases": [bias is not None for bias in snn.biases],
        "thresholds": [float(value) for value in snn.thresholds],
        "T": snn.T,
        "initial_membrane_fraction": float(snn.initial_membrane_fraction),
        "layer_membrane_fractions": snn.layer_membrane_fractions,
        "output_scale": float(snn.output_scale),
        "output_mode": snn.output_mode,
    }
    arrays: dict[str, Any] = {"header": np.array(json.dumps(header, sort_keys=True))}
    for index, (weight, bias) in enumerate(zip(snn.weights, snn.biases, strict=True)):
        arrays[f"weight_{index}"] = np.asarray(weight, dtype=np.float64)
        if bias is not None:
            arrays[f"bias_{index}"] = np.asarray(bias, dtype=np.float64)
    with Path(path).open("wb") as handle:
        np.savez(handle, **arrays)
    return digest


def load_converted_network(path: str | Path) -> ConvertedSNN:
    """Read a network written by :func:`save_converted_network`.

    Parameters
    ----------
    path : str or Path
        The ``.npz`` file.

    Returns
    -------
    ConvertedSNN
        The network, whose digest equals the one recorded in the file.

    Raises
    ------
    ValueError
        Another schema, missing or extra arrays, or a digest mismatch.
    """
    with np.load(Path(path), allow_pickle=False) as data:
        names = set(data.files)
        if "header" not in names:
            raise ValueError("not a converted network file: no header")
        header = json.loads(str(data["header"]))
        if header.get("schema_version") != CONVERTED_NETWORK_SCHEMA_VERSION:
            raise ValueError("converted network file uses another schema")
        layers = int(header["layers"])
        with_bias = list(header["biases"])
        expected = {"header"} | {f"weight_{i}" for i in range(layers)}
        expected |= {f"bias_{i}" for i in range(layers) if with_bias[i]}
        if names != expected or len(with_bias) != layers:
            raise ValueError("converted network file arrays do not match its header")
        snn = ConvertedSNN(
            [data[f"weight_{i}"] for i in range(layers)],
            [data[f"bias_{i}"] if with_bias[i] else None for i in range(layers)],
            header["thresholds"],
            int(header["T"]),
            header["initial_membrane_fraction"],
            header["output_scale"],
            header["output_mode"],
            layer_membrane_fractions=header["layer_membrane_fractions"],
        )
    if converted_sha256(snn) != header["converted_sha256"]:
        raise ValueError("converted network file digest does not match its contents")
    return snn


__all__ = [
    "CONVERTED_NETWORK_SCHEMA_VERSION",
    "load_converted_network",
    "save_converted_network",
]
