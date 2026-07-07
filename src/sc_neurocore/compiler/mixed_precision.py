# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Mixed-precision facade

"""Mixed-precision per-variable compilation engine."""

from __future__ import annotations

from .mixed_precision_spec import (
    MixedPrecisionSpec,
)
from .precision_config import (
    BlockFloatingPrecisionConfig,
    BlockFloatingScalarEncodingError,
    PrecisionConfig,
    PrecisionSpecLike,
    encode_scalar_value,
)
from .precision_presets import (
    PRECISION_PRESETS,
    from_preset,
)
from .precision_solver import (
    solve_precision,
)

__all__ = [
    "BlockFloatingPrecisionConfig",
    "BlockFloatingScalarEncodingError",
    "MixedPrecisionSpec",
    "PRECISION_PRESETS",
    "PrecisionConfig",
    "PrecisionSpecLike",
    "encode_scalar_value",
    "from_preset",
    "solve_precision",
]
