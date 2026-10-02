# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Source-authored equation admission and state failures

"""Distinguish deliberate equation reasons from interpreter diagnostics."""

from sc_neurocore.refusals import AuthoredRefusal


class EquationRefusal(AuthoredRefusal):
    """An authored equation or profile refusal, compatible with ValueError."""


class EquationStateFailure(FloatingPointError, AuthoredRefusal):
    """An authored state failure, compatible with FloatingPointError."""


class EquationEvaluationFailure(EquationRefusal):
    """An authored failure confined to evaluating a compiled equation value."""


class EquationDivisionFailure(ZeroDivisionError, EquationEvaluationFailure):
    """Preserve division-by-zero compatibility without generated diagnostics."""


class EquationOverflowFailure(OverflowError, EquationEvaluationFailure):
    """Preserve numeric-overflow compatibility without generated diagnostics."""


class EquationFloatingFailure(FloatingPointError, EquationEvaluationFailure):
    """Preserve floating-point-error compatibility without generated diagnostics."""


class EquationTypeFailure(TypeError, EquationEvaluationFailure):
    """Preserve invalid-value-type compatibility without generated diagnostics."""


class EquationIndexFailure(IndexError, EquationEvaluationFailure):
    """Preserve invalid-index compatibility without generated diagnostics."""


class EquationAttributeFailure(AttributeError, EquationEvaluationFailure):
    """Preserve unavailable-attribute compatibility without generated diagnostics."""


class EquationNameFailure(NameError, EquationEvaluationFailure):
    """Preserve unavailable-symbol compatibility without generated diagnostics."""
