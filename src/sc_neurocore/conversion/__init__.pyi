# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Types for lazy public conversion exports

"""Static signatures for the lazily imported optional PyTorch conversion API."""

from .ann_to_snn import (
    ConvertedSNN as ConvertedSNN,
    convert as convert,
    replace_relu_with_qcfs as replace_relu_with_qcfs,
)
from .loss_report import (
    ConversionLossReport as ConversionLossReport,
    measure_conversion_loss as measure_conversion_loss,
)
from .qcfs import QCFSActivation as QCFSActivation
from .qcfs_dispatch import qcfs_backward as qcfs_backward, qcfs_forward as qcfs_forward

__all__: list[str]

def __getattr__(name: str) -> object:
    """Resolve a conversion symbol lazily using the runtime package implementation."""
    ...
