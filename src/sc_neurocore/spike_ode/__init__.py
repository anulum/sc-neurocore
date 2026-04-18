# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Spiking Neural ODEs

"""Continuous-depth SNNs: ODE solver + spiking dynamics + event handling."""

from .ode_layer import SpikingODELayer, ODELIFDynamics

__all__ = ["SpikingODELayer", "ODELIFDynamics"]
