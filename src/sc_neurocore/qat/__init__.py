# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Quantization-aware training for SNNs

"""Train SNNs through quantization for hardware deployment."""

from .quantize import QuantizedSNNLayer, TernaryWeights, quantize_aware_train_step

__all__ = [
    "QuantizedSNNLayer",
    "quantize_aware_train_step",
    "TernaryWeights",
]

try:  # pragma: no cover — torch optional
    from .lsq import LSQLinear, LSQQuantizer  # pragma: no cover
    from .observers import (  # pragma: no cover
        MinMaxObserver,
        PerChannelMinMaxObserver,
        fake_quantize,
    )
    from .pact import PACTActivation  # pragma: no cover
    from .torch_qat import (  # pragma: no cover
        LSQPACTLIFNet,
        QuantizedLIFNet,
        QuantizedLinear,
        SCAwareLIFNet,
        SCAwareLinear,
        ste_quantize,
    )

    __all__ += [  # pragma: no cover
        "LSQLinear",
        "LSQQuantizer",
        "LSQPACTLIFNet",
        "MinMaxObserver",
        "PACTActivation",
        "PerChannelMinMaxObserver",
        "QuantizedLIFNet",
        "QuantizedLinear",
        "SCAwareLIFNet",
        "SCAwareLinear",
        "fake_quantize",
        "ste_quantize",
    ]
except ImportError:
    pass
