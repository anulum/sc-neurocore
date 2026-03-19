# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Export SC-NeuroCore networks to NIR format

from __future__ import annotations

from pathlib import Path

try:
    import nir
except ImportError as e:
    raise ImportError("pip install nir") from e


def to_nir(network, path: str | Path | None = None) -> nir.NIRGraph:
    """Export an SC-NeuroCore network to NIR format.

    Parameters
    ----------
    network : SCNetwork
        The network to export.
    path : str or Path, optional
        If provided, write the NIR graph to this file.

    Returns
    -------
    nir.NIRGraph

    .. note::
        Phase 3 — stub for now. Reverse mapping will be implemented
        after the import path is validated with real NIR models.
    """
    raise NotImplementedError(
        "to_nir() export is planned for Phase 3. "
        "Use from_nir() to import NIR graphs into SC-NeuroCore."
    )
