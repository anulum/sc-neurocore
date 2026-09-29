# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — ANN-to-SNN conversion engine

"""ANN-to-SNN conversion: convert trained PyTorch ANNs to spiking networks.

Converting a network and ``QCFSActivation`` require ``pip install
sc-neurocore[torch]`` (PyTorch). ``ConvertedSNN`` replay and the QCFS
``qcfs_forward``/``qcfs_backward`` evaluation run on NumPy or a configured
native runtime without PyTorch. ``measure_conversion_loss`` runs a PyTorch
source beside its converted network and reports the measured difference.
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
    if name == "ConvertedSNN":
        from .converted_snn import ConvertedSNN

        return ConvertedSNN
    if name in ("convert", "replace_relu_with_qcfs"):
        from .ann_to_snn import convert, replace_relu_with_qcfs

        return {
            "convert": convert,
            "replace_relu_with_qcfs": replace_relu_with_qcfs,
        }[name]
    if name in ("qcfs_forward", "qcfs_backward"):
        from .qcfs_dispatch import qcfs_backward, qcfs_forward

        return {"qcfs_forward": qcfs_forward, "qcfs_backward": qcfs_backward}[name]
    if name in ("ConversionLossReport", "measure_conversion_loss"):
        from .loss_report import ConversionLossReport, measure_conversion_loss

        return {
            "ConversionLossReport": ConversionLossReport,
            "measure_conversion_loss": measure_conversion_loss,
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


__all__ = [
    "ConversionLossReport",
    "convert",
    "ConvertedSNN",
    "measure_conversion_loss",
    "QCFSActivation",
    "qcfs_backward",
    "qcfs_forward",
    "replace_relu_with_qcfs",
]
