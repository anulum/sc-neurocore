# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Mojo parity notes for connor_stevens

# Maintained contract note:
# The executable Connor-Stevens kernels are the Python reference, Rust engine,
# Go service, Julia mirror, and Rust safety surface. This Mojo file is kept as a
# parity note until the generated Mojo neuron-kernel lane is promoted to a build
# target. The required state order is:
#     v, m, h, n, a, b
# The required macro-step contract is candidate-first RK4 over
# Int(1.0 / max(dt, 0.001)) sub-steps with the Connor-Stevens 1971/1977 rate
# equations, finite current/state/parameter validation, gate-envelope checks,
# and no state mutation on invalid candidates.
