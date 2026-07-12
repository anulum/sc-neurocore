# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Mojo C ABI for the Medvedev 2005 first-return map

# Build:
#   mojo build --emit shared-lib -o libmedvedev.so medvedev_map.mojo
#
# The caller supplies n_steps+1 Float64 slots: the slow-calcium trace followed
# by final u. A negative result reports rejected input or a non-finite candidate.

from std.math import exp, isfinite, log
from std.memory import UnsafePointer


@export
def medvedev_map_simulate_c(
    u0: Float64,
    beta_0: Float64,
    beta_hc: Float64,
    beta_sn: Float64,
    delta: Float64,
    decay_t0: Float64,
    alpha_t0: Float64,
    f_0: Float64,
    f_1: Float64,
    homoclinic_exponent: Float64,
    d: Float64,
    input_gain: Float64,
    n_steps: Int,
    current: Float64,
    trace_addr: Int,
) -> Int64:
    if n_steps < 0 or trace_addr == 0:
        return -1
    if (
        not isfinite(u0)
        or not isfinite(beta_0)
        or not isfinite(beta_hc)
        or not isfinite(beta_sn)
        or not isfinite(delta)
        or not isfinite(decay_t0)
        or not isfinite(alpha_t0)
        or not isfinite(f_0)
        or not isfinite(f_1)
        or not isfinite(homoclinic_exponent)
        or not isfinite(d)
        or not isfinite(input_gain)
        or not isfinite(current)
        or not (0.0 < beta_0 and beta_0 < beta_sn and beta_sn < beta_hc and beta_hc < delta)
        or not (0.0 < decay_t0 and decay_t0 < 1.0)
        or not (0.0 < alpha_t0 and alpha_t0 < 1.0)
        or not (0.0 <= f_1 and f_1 < f_0)
        or homoclinic_exponent <= 0.0
        or d <= 0.0
        or input_gain < 0.0
    ):
        return -1

    var trace = UnsafePointer[Float64, MutAnyOrigin](unsafe_from_address=trace_addr)
    var u_0 = beta_0 / (delta - beta_0)
    var u_hc = beta_hc / (delta - beta_hc)
    var u_sn = beta_sn / (delta - beta_sn)
    var u = u0
    var events: Int64 = 0
    for index in range(n_steps):
        if u <= u_hc:
            events += 1

        var candidate: Float64
        if u <= u_0:
            candidate = decay_t0 * u + (1.0 - decay_t0) * f_0 + input_gain * current
        elif u <= u_hc:
            var u_1 = (1.0 - alpha_t0) * u + alpha_t0 * f_0
            var gap = beta_hc - delta * u_1 / (1.0 + u_1)
            var inner_return: Float64
            if gap <= 0.0:
                inner_return = f_1
            else:
                var log_argument = d * gap
                if not isfinite(log_argument) or log_argument <= 0.0:
                    return -1
                var scale = exp(homoclinic_exponent * log(log_argument))
                inner_return = scale * (u_1 - f_1) + f_1
            candidate = inner_return + input_gain * current
        else:
            candidate = u_sn

        if not isfinite(candidate):
            return -1
        u = candidate
        trace[index] = u

    trace[n_steps] = u
    return events
