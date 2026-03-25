# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Automated SNN Architecture Doctor

"""Automated SNN architecture diagnostics: coding efficiency, hardware fit,
spike health, and actionable fix recommendations."""

from .diagnose import diagnose, Diagnosis, DiagnosticReport

__all__ = ["diagnose", "Diagnosis", "DiagnosticReport"]
