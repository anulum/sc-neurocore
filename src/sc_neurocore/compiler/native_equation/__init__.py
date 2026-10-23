# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Typed equation lowering components

"""Typed numeric and expression components for equation-to-native graph lowering."""

from .arithmetic import EquationArithmetic
from .expressions import EquationExpressions

__all__ = ["EquationArithmetic", "EquationExpressions"]
