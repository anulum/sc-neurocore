# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — ANN-to-SNN conversion engine

"""ANN-to-SNN conversion: convert trained PyTorch ANNs to spiking networks.

Requires ``pip install sc-neurocore[torch]`` (PyTorch).
"""

from __future__ import annotations


def __getattr__(name: str) -> object:
    """Lazily resolve optional PyTorch conversion surfaces.

    Parameters
    ----------
    name : str
        Public conversion symbol requested from the package.

    Returns
    -------
    object
        The resolved conversion function or class.

    Raises
    ------
    AttributeError
        If ``name`` is not exported by this package.
    ImportError
        If the requested symbol requires PyTorch and PyTorch is unavailable.
    """
    if name in ("convert", "ConvertedSNN", "replace_relu_with_qcfs"):
        from .ann_to_snn import ConvertedSNN, convert, replace_relu_with_qcfs

        return {
            "convert": convert,
            "ConvertedSNN": ConvertedSNN,
            "replace_relu_with_qcfs": replace_relu_with_qcfs,
        }[name]
    if name == "QCFSActivation":
        try:
            from .qcfs import QCFSActivation
        except ImportError as exc:
            raise ImportError(
                "QCFSActivation requires PyTorch: pip install sc-neurocore[torch]"
            ) from exc
        return QCFSActivation
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["convert", "ConvertedSNN", "QCFSActivation", "replace_relu_with_qcfs"]
