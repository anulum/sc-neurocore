# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Public research import for the codec spike predictor

"""Expose the codec's NumPy spike predictor under the original research imports."""

from ..spike_codec._world_model import (
    SpikePredictor as SpikePredictor,
    predict_and_xor_world_model as predict_and_xor_world_model,
    xor_and_recover_world_model as xor_and_recover_world_model,
)

__all__ = ["SpikePredictor", "predict_and_xor_world_model", "xor_and_recover_world_model"]
