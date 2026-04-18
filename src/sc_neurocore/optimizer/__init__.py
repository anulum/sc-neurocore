# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Automatic FPGA resource optimizer

"""Automatically compress an SNN to fit a target FPGA."""

from .resource_optimizer import fit_to_target, OptimizationResult

__all__ = ["fit_to_target", "OptimizationResult"]
