# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — sc_neurocore.stochastic_doctor -- Bitstream health diagnostics and correlation analysis

"""sc_neurocore.stochastic_doctor -- Bitstream health diagnostics and correlation analysis.

Tier: research.
"""

__tier__ = "research"

# Re-export the compiled Rust extension so downstream callers can spell
# `from sc_neurocore.stochastic_doctor import stochastic_doctor_core`.
# The extension may legitimately be absent (e.g. wheel-only install
# without maturin build); absorb the ImportError so module import
# never hard-fails, and callers gate runtime use on the attribute.
try:
    from . import stochastic_doctor_core  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover — .so not built
    stochastic_doctor_core = None

__all__ = ["stochastic_doctor_core"]
