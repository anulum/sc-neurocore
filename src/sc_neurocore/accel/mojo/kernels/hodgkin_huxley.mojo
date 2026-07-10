# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Mojo parity notes for hodgkin_huxley

# Maintained contract note:
# The executable Hodgkin-Huxley kernels are the Python reference, the Rust engine
# RK4 binding, the Go service, the Julia mirror, and the Rust safety surface. This
# Mojo file is kept as a parity note until the generated Mojo neuron-kernel lane is
# promoted to a build target. The required state order is:
#     v, m, h, n
# The required macro-step contract is the historical baseline-Euler schedule over
# round(1.0 / dt) explicit sub-steps: within each sub-step the gating variables
# (m, h, n) advance first with the α/β rate equations, then the membrane voltage
# updates using the freshly-updated gates. The singular opening rates α_m and α_n
# take their analytic limit (1.0 and 0.1) when |v + shift| < 1e-7. Finite
# current/state/parameter validation and gate-envelope checks apply, with no state
# mutation on an invalid candidate. The spike-count parity contract against the
# Python golden is: silent at zero drive, six action potentials at I = 10 over
# 100 macro steps, and nine at I = 20.
