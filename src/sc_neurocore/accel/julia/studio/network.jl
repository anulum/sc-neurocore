# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for studio/network

module NetworkAccel

using Statistics, LinearAlgebra

function simulate_ei_network(n_exc, n_inh, w_ee, w_ei, w_ie, w_ii, p_conn, ext_rate, duration, dt)
    n_exc: int = 80,
    n_inh: int = 20,
    w_ee: float = 0.1,
    w_ei: float = 0.4,
    w_ie: float = 0.1,
    w_ii: float = 0.4,
    p_conn: float = 0.2,
    ext_rate: float = 5.0,
    duration: float = 200.0,
    dt: float = 0.1,
    ) -> dict
    try
        return _simulate_rust(
            n_exc,
            n_inh,
            w_ee,
            w_ei,
            w_ie,
            w_ii,
            p_conn,
            ext_rate,
            duration,
            dt,
        )
    except ImportError
        return _simulate_numpy(
            n_exc,
            n_inh,
            w_ee,
            w_ei,
            w_ie,
            w_ii,
            p_conn,
            ext_rate,
            duration,
            dt,
        )
end

end # module NetworkAccel
