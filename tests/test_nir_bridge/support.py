# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Deterministic NIR bridge test graphs

"""Build deterministic public-surface graphs shared by NIR bridge tests."""

from __future__ import annotations

from typing import Any

import nir
import numpy as np


def make_lif_affine_graph(n_in: int = 3, n_out: int = 2) -> Any:
    """Build an input, affine, LIF, and output NIR graph.

    Parameters
    ----------
    n_in : int, optional
        Input vector width.
    n_out : int, optional
        Affine output and LIF population width.

    Returns
    -------
    Any
        A ``nir.NIRGraph`` from the optional untyped NIR dependency.
    """
    nodes = {
        "input": nir.Input(input_type={"input": np.array([n_in])}),
        "affine": nir.Affine(
            weight=np.random.RandomState(42).randn(n_out, n_in).astype(np.float32),
            bias=np.zeros(n_out, dtype=np.float32),
        ),
        "lif": nir.LIF(
            tau=np.full(n_out, 20.0),
            r=np.ones(n_out),
            v_leak=np.zeros(n_out),
            v_threshold=np.ones(n_out),
        ),
        "output": nir.Output(output_type={"output": np.array([n_out])}),
    }
    edges = [("input", "affine"), ("affine", "lif"), ("lif", "output")]
    return nir.NIRGraph(nodes=nodes, edges=edges)
