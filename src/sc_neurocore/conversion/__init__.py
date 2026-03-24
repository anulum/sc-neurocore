# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — ANN-to-SNN conversion engine

"""ANN-to-SNN conversion: convert trained PyTorch ANNs to spiking networks."""

from .ann_to_snn import convert, ConvertedSNN
from .qcfs import QCFSActivation

__all__ = ["convert", "ConvertedSNN", "QCFSActivation"]
