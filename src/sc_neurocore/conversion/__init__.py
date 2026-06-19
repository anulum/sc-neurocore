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


def __getattr__(name):  # type: ignore[no-untyped-def]
    if name in ("convert", "ConvertedSNN"):
        from .ann_to_snn import ConvertedSNN, convert

        return {"convert": convert, "ConvertedSNN": ConvertedSNN}[name]
    if name == "QCFSActivation":
        try:
            from .qcfs import QCFSActivation
        except ImportError as exc:
            raise ImportError(
                "QCFSActivation requires PyTorch: pip install sc-neurocore[torch]"
            ) from exc
        return QCFSActivation
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["convert", "ConvertedSNN", "QCFSActivation"]
